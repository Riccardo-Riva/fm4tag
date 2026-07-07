"""Optuna hyper-parameter optimisation on top of :func:`fm4tag.runner.run`.

Driven entirely by the ``hpo:`` section of the experiment config::

    hpo:
      study_name: fm4tag_hpo
      storage:    null            # e.g. sqlite:///hpo.db for parallel workers
      n_trials:   100
      timeout:    null            # seconds, per optimize() call
      phases:     [finetune]      # [pretrain], [finetune], or [pretrain, finetune]
      pretrain_encoder_ckpt: null # fixed pretrain ckpt for finetune-only HPO

      metric:     val/loss        # any logged metric name
      direction:  minimize        # minimize | maximize

      sampler: TPE                # TPE | random
      pruner:  median             # median | hyperband | none
      pruner_n_startup_trials: 5
      pruner_n_warmup_steps:   10
      pruner_interval_steps:   1

      max_epochs_pretrain: 100
      max_epochs_finetune: 100

      search_space:
        finetune:
          - {param: optimizer.lr, type: float, low: 1e-5, high: 1e-2, log: true}
          - {param: aggregator.depth, type: int, low: 3, high: 6}
          - {param: aggregator.ff_mult, type: categorical, choices: [2, 4]}

Each trial deep-copies the config **unresolved**, so interpolations like
``pretrain.lr: ${optimizer.lr}`` keep following the overridden value.  With
``phases: [pretrain, finetune]`` the finetune phase loads the pretrain
phase's best checkpoint; the pruning/metric callback is attached only to the
last phase.

Run with the installed entry point (same Hydra semantics as ``fm4tag``)::

    fm4tag-hpo --config-name=my_experiment hpo.n_trials=50

Notes
-----
* Use one device per trial; parallelise by launching several ``fm4tag-hpo``
  workers against a shared RDB ``storage``.
* The wandb logger is dropped per-trial by default (``hpo.disable_wandb``):
  one process cannot host many wandb runs, and ``log_model`` would upload
  every trial's checkpoints.  The CSV logger is kept.
"""

from __future__ import annotations

import copy

import hydra
import optuna
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig, OmegaConf, open_dict

from fm4tag.runner.run import run

# ---------------------------------------------------------------------------
# Search-space / study building blocks
# ---------------------------------------------------------------------------


def suggest_param(trial: optuna.Trial, spec: dict):
    """Suggest one value from a ``search_space`` entry.

    ``spec`` keys: ``param`` (dotted config path, also the Optuna name),
    ``type`` (``float`` | ``int`` | ``categorical``), then ``low``/``high``
    (+ optional ``log``) or ``choices``.
    """
    name, ptype = spec['param'], spec['type']
    if ptype == 'float':
        return trial.suggest_float(
            name, spec['low'], spec['high'], log=spec.get('log', False)
        )
    if ptype == 'int':
        return trial.suggest_int(
            name, spec['low'], spec['high'], log=spec.get('log', False)
        )
    if ptype == 'categorical':
        return trial.suggest_categorical(name, list(spec['choices']))
    raise ValueError(
        f"search_space type must be 'float', 'int' or 'categorical', got {ptype!r}"
    )


def build_sampler(hpo_cfg: DictConfig, seed: int) -> optuna.samplers.BaseSampler:
    name = str(hpo_cfg.get('sampler', 'TPE')).lower()
    if name == 'tpe':
        return optuna.samplers.TPESampler(seed=seed)
    if name == 'random':
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"sampler must be 'TPE' or 'random', got {name!r}")


def build_pruner(hpo_cfg: DictConfig) -> optuna.pruners.BasePruner:
    name = str(hpo_cfg.get('pruner', 'median') or 'none').lower()
    if name == 'median':
        return optuna.pruners.MedianPruner(
            n_startup_trials=hpo_cfg.get('pruner_n_startup_trials', 5),
            n_warmup_steps=hpo_cfg.get('pruner_n_warmup_steps', 10),
            interval_steps=hpo_cfg.get('pruner_interval_steps', 1),
        )
    if name == 'hyperband':
        return optuna.pruners.HyperbandPruner()
    if name == 'none':
        return optuna.pruners.NopPruner()
    raise ValueError(f"pruner must be 'median', 'hyperband' or 'none', got {name!r}")


# ---------------------------------------------------------------------------
# Metric reporting / pruning callback
# ---------------------------------------------------------------------------


class _OptunaMetricCallback(Callback):
    """Report ``metric`` to the trial after each validation epoch and prune.

    The last reported value is exposed as :attr:`last_value` — the objective
    returns it after the run finishes.
    """

    def __init__(self, trial: optuna.Trial, metric: str) -> None:
        self.trial = trial
        self.metric = metric
        self.last_value: float | None = None
        self._epoch = 0

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.metric)
        if value is None:
            raise ValueError(
                f'HPO metric {self.metric!r} not found in callback_metrics; '
                f'available: {sorted(trainer.callback_metrics)}'
            )
        self.last_value = float(value)
        self.trial.report(self.last_value, self._epoch)
        self._epoch += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f'Pruned at epoch {self._epoch - 1}: {self.metric}={self.last_value}'
            )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _trial_config(cfg: DictConfig, trial: optuna.Trial, phase: str) -> DictConfig:
    """Deep-copy ``cfg`` (unresolved, so interpolations keep following) and
    apply this trial's suggested overrides for ``phase``."""
    cfg_trial = copy.deepcopy(cfg)
    for spec in cfg.hpo.search_space.get(phase, []):
        value = suggest_param(trial, OmegaConf.to_container(spec, resolve=True))
        OmegaConf.update(cfg_trial, spec.param, value, merge=True)
    return cfg_trial


def _objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    hpo_cfg = cfg.hpo
    phases = list(hpo_cfg.get('phases', ['finetune']))
    metric = hpo_cfg.get('metric', 'val/loss')
    encoder_ckpt = hpo_cfg.get('pretrain_encoder_ckpt')

    cb: _OptunaMetricCallback | None = None
    for phase in phases:
        cfg_trial = _trial_config(cfg, trial, phase)
        max_epochs = hpo_cfg.get(f'max_epochs_{phase}')
        # open_dict: allow add/delete also on struct (Hydra) configs.
        with open_dict(cfg_trial):
            cfg_trial.experiment_name = (
                f'{cfg.get("experiment_name", "fm4tag")}-hpo-t{trial.number}-{phase}'
            )
            if max_epochs is not None:
                cfg_trial.trainer.max_epochs = max_epochs
            if hpo_cfg.get('disable_wandb', True) and OmegaConf.select(
                cfg_trial, 'loggers.wandb'
            ):
                del cfg_trial.loggers.wandb

        # Report/prune only on the last phase — its metric is the objective.
        is_last = phase == phases[-1]
        cb = _OptunaMetricCallback(trial, metric) if is_last else None

        trainer = run(
            cfg_trial,
            phase=phase,
            action='fit',
            encoder_ckpt=encoder_ckpt if phase == 'finetune' else None,
            extra_callbacks=[cb] if cb else None,
        )

        # Chain: finetune loads the pretrain phase's best checkpoint.
        if phase == 'pretrain' and 'finetune' in phases:
            encoder_ckpt = trainer.checkpoint_callback.best_model_path

    if cb is None or cb.last_value is None:
        raise RuntimeError(
            f'HPO metric {metric!r} was never reported — did validation run?'
        )
    return cb.last_value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_hpo(cfg: DictConfig) -> optuna.Study:
    """Create/load the study described by ``cfg.hpo`` and optimise it."""
    hpo_cfg = cfg.hpo

    devices = cfg.trainer.get('devices', 1)
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, tuple)) and len(devices) > 1
    ):
        rank_zero_warn(
            'HPO with multi-GPU trials is not supported (pruning + Optuna '
            'storage interact badly with DDP). Use devices: 1 and parallelise '
            'via multiple fm4tag-hpo workers on a shared RDB storage.'
        )

    study = optuna.create_study(
        study_name=hpo_cfg.get('study_name', 'fm4tag_hpo'),
        storage=hpo_cfg.get('storage'),
        sampler=build_sampler(hpo_cfg, seed=cfg.get('seed', 42)),
        pruner=build_pruner(hpo_cfg),
        direction=hpo_cfg.get('direction', 'minimize'),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: _objective(trial, cfg),
        n_trials=hpo_cfg.get('n_trials', 100),
        timeout=hpo_cfg.get('timeout'),
    )

    print(f'\nBest trial: #{study.best_trial.number}')
    print(f'  {hpo_cfg.get("metric", "val/loss")}: {study.best_value}')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    return study


@hydra.main(version_base=None, config_path='configs', config_name='default')
def main(cfg: DictConfig) -> None:
    run_hpo(cfg)
