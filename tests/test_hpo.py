"""Tests for fm4tag.hpo (no real training — mocked trials/trainers)."""

from __future__ import annotations

from types import SimpleNamespace

import optuna
import pytest
import torch
from omegaconf import OmegaConf

from fm4tag.hpo import (
    _OptunaMetricCallback,
    _trial_config,
    build_pruner,
    build_sampler,
    suggest_param,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Search-space suggestion
# ---------------------------------------------------------------------------


def _fresh_trial(**study_kwargs) -> optuna.Trial:
    return optuna.create_study(**study_kwargs).ask()


def test_suggest_param_float_int_categorical():
    trial = _fresh_trial()
    v = suggest_param(trial, {'param': 'a', 'type': 'float', 'low': 0.1, 'high': 0.5})
    assert 0.1 <= v <= 0.5
    v = suggest_param(
        trial, {'param': 'b', 'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True}
    )
    assert 1e-5 <= v <= 1e-2
    v = suggest_param(trial, {'param': 'c', 'type': 'int', 'low': 1, 'high': 3})
    assert v in (1, 2, 3)
    v = suggest_param(trial, {'param': 'd', 'type': 'categorical', 'choices': [2, 4]})
    assert v in (2, 4)


def test_suggest_param_unknown_type_raises():
    with pytest.raises(ValueError, match='float'):
        suggest_param(_fresh_trial(), {'param': 'x', 'type': 'bool'})


# ---------------------------------------------------------------------------
# Sampler / pruner factories
# ---------------------------------------------------------------------------


def test_build_sampler():
    cfg = OmegaConf.create({'sampler': 'TPE'})
    assert isinstance(build_sampler(cfg, seed=1), optuna.samplers.TPESampler)
    cfg = OmegaConf.create({'sampler': 'random'})
    assert isinstance(build_sampler(cfg, seed=1), optuna.samplers.RandomSampler)
    with pytest.raises(ValueError, match='sampler'):
        build_sampler(OmegaConf.create({'sampler': 'cmaes'}), seed=1)


def test_build_pruner():
    cfg = OmegaConf.create(
        {
            'pruner': 'median',
            'pruner_n_startup_trials': 3,
            'pruner_n_warmup_steps': 7,
        }
    )
    pruner = build_pruner(cfg)
    assert isinstance(pruner, optuna.pruners.MedianPruner)
    assert isinstance(
        build_pruner(OmegaConf.create({'pruner': 'none'})), optuna.pruners.NopPruner
    )
    assert isinstance(
        build_pruner(OmegaConf.create({'pruner': 'hyperband'})),
        optuna.pruners.HyperbandPruner,
    )
    with pytest.raises(ValueError, match='pruner'):
        build_pruner(OmegaConf.create({'pruner': 'asha'}))


# ---------------------------------------------------------------------------
# Trial config overrides
# ---------------------------------------------------------------------------


def test_trial_config_applies_overrides_and_keeps_interpolation():
    cfg = OmegaConf.create(
        {
            'optimizer': {'lr': 1e-4},
            'pretrain': {'lr': '${optimizer.lr}'},
            'hpo': {
                'search_space': {
                    'finetune': [
                        # degenerate range → deterministic suggestion
                        {
                            'param': 'optimizer.lr',
                            'type': 'float',
                            'low': 0.5,
                            'high': 0.5,
                        },
                    ]
                }
            },
        }
    )
    cfg_trial = _trial_config(cfg, _fresh_trial(), 'finetune')
    assert cfg_trial.optimizer.lr == 0.5
    assert cfg_trial.pretrain.lr == 0.5  # interpolation follows the override
    assert cfg.optimizer.lr == 1e-4  # original config untouched


def test_trial_config_phase_without_search_space_is_noop():
    cfg = OmegaConf.create(
        {'optimizer': {'lr': 1e-4}, 'hpo': {'search_space': {'finetune': []}}}
    )
    cfg_trial = _trial_config(cfg, _fresh_trial(), 'pretrain')
    assert cfg_trial.optimizer.lr == 1e-4


# ---------------------------------------------------------------------------
# Metric callback
# ---------------------------------------------------------------------------


def _fake_trainer(metrics: dict, sanity: bool = False) -> SimpleNamespace:
    return SimpleNamespace(sanity_checking=sanity, callback_metrics=metrics)


def test_metric_callback_reports_value():
    cb = _OptunaMetricCallback(_fresh_trial(), 'val/loss')
    cb.on_validation_end(_fake_trainer({'val/loss': torch.tensor(0.5)}), None)
    assert cb.last_value == pytest.approx(0.5)


def test_metric_callback_skips_sanity_check():
    cb = _OptunaMetricCallback(_fresh_trial(), 'val/loss')
    cb.on_validation_end(
        _fake_trainer({'val/loss': torch.tensor(0.5)}, sanity=True), None
    )
    assert cb.last_value is None


def test_metric_callback_missing_metric_raises():
    cb = _OptunaMetricCallback(_fresh_trial(), 'val/head/auroc')
    with pytest.raises(ValueError, match='not found'):
        cb.on_validation_end(_fake_trainer({'val/loss': torch.tensor(0.5)}), None)


def test_metric_callback_prunes():
    class _AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    cb = _OptunaMetricCallback(_fresh_trial(pruner=_AlwaysPrune()), 'val/loss')
    with pytest.raises(optuna.TrialPruned):
        cb.on_validation_end(_fake_trainer({'val/loss': torch.tensor(0.5)}), None)


# ---------------------------------------------------------------------------
# Objective (mocked run)
# ---------------------------------------------------------------------------


def test_objective_chains_phases_and_returns_metric(monkeypatch):
    import fm4tag.hpo as hpo_mod

    calls = []

    def fake_run(cfg, *, phase, action, encoder_ckpt, extra_callbacks):
        calls.append((phase, encoder_ckpt, cfg.trainer.max_epochs))
        if extra_callbacks:
            extra_callbacks[0].on_validation_end(
                _fake_trainer({'val/loss': torch.tensor(0.3)}), None
            )
        return SimpleNamespace(
            checkpoint_callback=SimpleNamespace(best_model_path='/tmp/best.ckpt')
        )

    monkeypatch.setattr(hpo_mod, 'run', fake_run)
    cfg = OmegaConf.create(
        {
            'experiment_name': 'exp',
            'trainer': {'max_epochs': 5},
            'hpo': {
                'phases': ['pretrain', 'finetune'],
                'metric': 'val/loss',
                'max_epochs_pretrain': 2,
                'max_epochs_finetune': 3,
                'search_space': {'pretrain': [], 'finetune': []},
            },
        }
    )
    value = hpo_mod._objective(_fresh_trial(), cfg)
    assert value == pytest.approx(0.3)
    # pretrain first (no ckpt), then finetune with pretrain's best ckpt.
    assert calls[0] == ('pretrain', None, 2)
    assert calls[1] == ('finetune', '/tmp/best.ckpt', 3)
