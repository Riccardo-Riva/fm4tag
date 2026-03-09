"""Hyperparameter optimisation for fm4tag using Optuna.

Run a two-phase sequential study::

    # Full HPO: pretrain encoder, then finetune classifier
    fm4tag-hpo --config-name=default hpo.n_trials=50

    # Finetune-only HPO (skip pretrain, supply an existing encoder checkpoint)
    fm4tag-hpo --config-name=default \\
        hpo.phases=[finetune] \\
        hpo.pretrain_encoder_ckpt=/path/to/pretrain.ckpt \\
        hpo.n_trials=50

    # Persistent study — safe to interrupt and resume
    fm4tag-hpo --config-name=default hpo.storage="sqlite:///hpo.db" hpo.n_trials=100

All ``hpo.*`` keys (n_trials, max_epochs_pretrain/finetune, sampler, pruner, …)
can be overridden from the CLI exactly like any other config key.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

import lightning as L

from fm4tag.engine import run

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optuna pruning callback
# ---------------------------------------------------------------------------


class _OptunaMetricCallback(L.Callback):
    """Report val_loss to Optuna at every validation epoch and prune if needed.

    After training completes, read ``callback.best_value`` to get the best
    metric seen during the run.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = 'val_loss') -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.best_value: float | None = None

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        raw = trainer.callback_metrics.get(self.monitor)
        if raw is None:
            return
        value = float(raw)
        epoch = trainer.current_epoch

        if self.best_value is None or value < self.best_value:
            self.best_value = value

        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f'Trial pruned at epoch {epoch} ({self.monitor}={value:.6f})'
            )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _deep_copy_cfg(cfg: DictConfig) -> DictConfig:
    """Return a mutable deep copy of *cfg* with all interpolations resolved."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _find_best_ckpt(log_dir: str, monitor: str = 'val_loss') -> str | None:
    """Return the checkpoint path with the lowest *monitor* value, or None.

    Lightning saves checkpoints with filenames like::

        epoch=005-val_loss=0.1234.ckpt

    The metric key uses underscores (``/`` → ``_``).
    """
    ckpt_dir = Path(log_dir) / 'checkpoints'
    if not ckpt_dir.exists():
        return None
    metric_key = monitor.replace('/', '_')
    pattern = re.compile(rf'{re.escape(metric_key)}=(\d+\.\d+)\.ckpt$')
    best_val = float('inf')
    best_path: str | None = None
    for ckpt in ckpt_dir.glob('*.ckpt'):
        m = pattern.search(ckpt.name)
        if m:
            val = float(m.group(1))
            if val < best_val:
                best_val = val
                best_path = str(ckpt)
    return best_path


def _log_dir_for(cfg: DictConfig) -> str:
    """Reconstruct the TensorBoardLogger log dir for a config (version 0)."""
    output_dir = cfg.get('output_dir', 'outputs')
    name = cfg.get('experiment_name', 'fm4tag')
    return str(Path(output_dir) / name / 'version_0')


# ---------------------------------------------------------------------------
# Sampler / pruner builders
# ---------------------------------------------------------------------------


def _build_sampler(hpo_cfg: DictConfig) -> optuna.samplers.BaseSampler:
    name = str(hpo_cfg.get('sampler', 'TPE')).upper()
    if name == 'TPE':
        return optuna.samplers.TPESampler()
    if name == 'CMAES':
        return optuna.samplers.CmaEsSampler()
    if name == 'RANDOM':
        return optuna.samplers.RandomSampler()
    raise ValueError(f'Unknown sampler {name!r}. Choose TPE, CmaES, or Random.')


def _build_pruner(hpo_cfg: DictConfig) -> optuna.pruners.BasePruner:
    name = str(hpo_cfg.get('pruner', 'median')).lower()
    n_startup = int(hpo_cfg.get('pruner_n_startup_trials', 5))
    n_warmup = int(hpo_cfg.get('pruner_n_warmup_steps', 10))
    interval = int(hpo_cfg.get('pruner_interval_steps', 1))
    if name == 'median':
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup,
            n_warmup_steps=n_warmup,
            interval_steps=interval,
        )
    if name == 'hyperband':
        return optuna.pruners.HyperbandPruner()
    if name == 'nop':
        return optuna.pruners.NopPruner()
    raise ValueError(f'Unknown pruner {name!r}. Choose median, hyperband, or nop.')


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------


def _suggest_from_config(
    trial: optuna.Trial, cfg: DictConfig, group: list
) -> None:
    """Apply suggestions from a declarative search-space list to *cfg* in-place.

    Each entry in *group* must have ``param`` (dot-path into *cfg*) and
    ``type`` (``categorical`` | ``int`` | ``float``).  Float entries may
    carry a boolean ``log`` key for log-uniform sampling.
    """
    for spec in group:
        param = str(spec['param'])
        kind = str(spec['type'])
        if kind == 'categorical':
            value = trial.suggest_categorical(param, list(spec['choices']))
        elif kind == 'int':
            value = trial.suggest_int(param, int(spec['low']), int(spec['high']))
        elif kind == 'float':
            value = trial.suggest_float(
                param,
                float(spec['low']),
                float(spec['high']),
                log=bool(spec.get('log', False)),
            )
        else:
            raise ValueError(f'Unknown search space type {kind!r} for {param!r}')
        OmegaConf.update(cfg, param, value, merge=False)


def _suggest_encoder_params(trial: optuna.Trial, cfg: DictConfig) -> None:
    """Suggest encoder architecture from ``cfg.hpo.search_space.encoder``.

    If the ``encoder`` group is absent from the search space (e.g. commented
    out in the config), the encoder architecture is kept fixed and nothing is
    suggested.  All parameters to optimise — including ``col_heads`` and
    ``row_heads`` — must be listed explicitly in the config search space.
    """
    space = cfg.hpo.search_space
    if 'encoder' not in space:
        return  # encoder architecture fixed at config values — skip

    _suggest_from_config(trial, cfg, space.encoder)


def _suggest_pretrain_params(trial: optuna.Trial, base_cfg: DictConfig) -> DictConfig:
    """Return a per-trial config with pretrain hyperparameters suggested."""
    cfg = _deep_copy_cfg(base_cfg)
    _suggest_encoder_params(trial, cfg)
    _suggest_from_config(trial, cfg, cfg.hpo.search_space.pretrain)
    cfg.trainer.max_epochs = int(base_cfg.hpo.max_epochs_pretrain)
    cfg.seed = int(base_cfg.get('seed', 42)) + trial.number
    return cfg


def _suggest_finetune_params(
    trial: optuna.Trial,
    base_cfg: DictConfig,
    *,
    has_pretrained_encoder: bool,
) -> DictConfig:
    """Return a per-trial config with finetune hyperparameters suggested."""
    cfg = _deep_copy_cfg(base_cfg)
    # Only tune encoder architecture when there is no fixed pretrained encoder.
    if not has_pretrained_encoder:
        _suggest_encoder_params(trial, cfg)
    _suggest_from_config(trial, cfg, cfg.hpo.search_space.finetune)
    cfg.trainer.max_epochs = int(base_cfg.hpo.max_epochs_finetune)
    cfg.seed = int(base_cfg.get('seed', 42)) + trial.number
    return cfg


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_trial_params(trial: optuna.Trial, phase: str) -> None:
    """Log a table of all hyperparameters sampled for *trial*."""
    params = trial.params
    if not params:
        return
    col_w = max(len(k) for k in params) + 2
    val_w = max(len(str(v)) for v in params.values()) + 2
    sep = '─' * (col_w + val_w + 3)
    lines = [
        f'\n{phase.capitalize()} trial {trial.number} — sampled hyperparameters:',
        sep,
        f'  {"Parameter":<{col_w}}{"Value":>{val_w}}',
        sep,
    ]
    for k, v in params.items():
        lines.append(f'  {k:<{col_w}}{str(v):>{val_w}}')
    lines.append(sep)
    log.info('\n'.join(lines))


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------


def _pretrain_objective(trial: optuna.Trial, base_cfg: DictConfig) -> float:
    trial_cfg = _suggest_pretrain_params(trial, base_cfg)

    trial_cfg.experiment_name = (
        f'{base_cfg.experiment_name}_hpo_pretrain_{trial.number:04d}'
    )

    has_val = bool(trial_cfg.get('pretrain_val_file'))
    monitor = 'val_loss' if has_val else 'train_loss'
    cb = _OptunaMetricCallback(trial, monitor=monitor)

    _log_trial_params(trial, 'pretrain')

    run(trial_cfg, phase='pretrain', action='fit', extra_callbacks=[cb])

    log_dir = _log_dir_for(trial_cfg)
    best_ckpt = _find_best_ckpt(log_dir, monitor=monitor)
    if best_ckpt is not None:
        trial.set_user_attr('best_ckpt', best_ckpt)
    trial.set_user_attr('log_dir', log_dir)

    value = cb.best_value
    if value is None:
        log.warning(
            'Pretrain trial %d produced no metric — marking failed.', trial.number
        )
        return float('inf')

    log.info('Pretrain trial %d finished: %s=%.6f', trial.number, monitor, value)
    return value


def _finetune_objective(
    trial: optuna.Trial,
    base_cfg: DictConfig,
    encoder_ckpt: str | None,
) -> float:
    trial_cfg = _suggest_finetune_params(
        trial, base_cfg, has_pretrained_encoder=encoder_ckpt is not None
    )

    trial_cfg.experiment_name = (
        f'{base_cfg.experiment_name}_hpo_finetune_{trial.number:04d}'
    )

    cb = _OptunaMetricCallback(trial, monitor='val_loss')

    _log_trial_params(trial, 'finetune')

    run(
        trial_cfg,
        phase='finetune',
        action='fit',
        encoder_ckpt=encoder_ckpt,
        extra_callbacks=[cb],
    )

    log_dir = _log_dir_for(trial_cfg)
    best_ckpt = _find_best_ckpt(log_dir, monitor='val_loss')
    if best_ckpt is not None:
        trial.set_user_attr('best_ckpt', best_ckpt)
    trial.set_user_attr('log_dir', log_dir)

    value = cb.best_value
    if value is None:
        log.warning(
            'Finetune trial %d produced no metric — marking failed.', trial.number
        )
        return float('inf')

    log.info('Finetune trial %d finished: val_loss=%.6f', trial.number, value)
    return value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='configs', config_name='default')
def main(cfg: DictConfig) -> None:
    """Run the Optuna HPO study (entry point for ``fm4tag-hpo``)."""
    hpo_cfg = cfg.get('hpo', {})
    phases = list(hpo_cfg.get('phases', ['pretrain', 'finetune']))
    encoder_ckpt: str | None = hpo_cfg.get('pretrain_encoder_ckpt') or None
    n_trials = int(hpo_cfg.get('n_trials', 50))
    timeout = hpo_cfg.get('timeout') or None
    storage = hpo_cfg.get('storage') or None
    study_name: str = str(hpo_cfg.get('study_name', 'fm4tag_hpo'))

    # ── Pretrain study ────────────────────────────────────────────────────────
    if 'pretrain' in phases and encoder_ckpt is None:
        pretrain_study = optuna.create_study(
            study_name=f'{study_name}_pretrain',
            direction='minimize',
            storage=storage,
            sampler=_build_sampler(hpo_cfg),
            pruner=_build_pruner(hpo_cfg),
            load_if_exists=True,
        )
        log.info(
            'Starting pretrain study "%s_pretrain" — %d trials', study_name, n_trials
        )
        pretrain_study.optimize(
            lambda t: _pretrain_objective(t, cfg),
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,),
        )

        best_pretrain = pretrain_study.best_trial
        encoder_ckpt = best_pretrain.user_attrs.get('best_ckpt')
        log.info(
            'Best pretrain trial: #%d  val_loss=%.6f  ckpt=%s',
            best_pretrain.number,
            best_pretrain.value,
            encoder_ckpt,
        )
        log.info('Best pretrain params: %s', best_pretrain.params)

    # ── Finetune study ────────────────────────────────────────────────────────
    if 'finetune' in phases:
        if encoder_ckpt is None and 'pretrain' not in phases:
            log.warning(
                'No pretrained encoder checkpoint available. '
                'Finetune study will tune encoder architecture as well.'
            )

        finetune_study = optuna.create_study(
            study_name=f'{study_name}_finetune',
            direction='minimize',
            storage=storage,
            sampler=_build_sampler(hpo_cfg),
            pruner=_build_pruner(hpo_cfg),
            load_if_exists=True,
        )
        log.info(
            'Starting finetune study "%s_finetune" — %d trials', study_name, n_trials
        )
        finetune_study.optimize(
            lambda t: _finetune_objective(t, cfg, encoder_ckpt),
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,),
        )

        best_finetune = finetune_study.best_trial
        log.info(
            'Best finetune trial: #%d  val_loss=%.6f  ckpt=%s',
            best_finetune.number,
            best_finetune.value,
            best_finetune.user_attrs.get('best_ckpt'),
        )
        log.info('Best finetune params: %s', best_finetune.params)
