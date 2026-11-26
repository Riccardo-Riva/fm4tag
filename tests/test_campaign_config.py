"""The runner and the arms, on the default base config: every arm composes and builds, and the runner wires the
config into the module.  The tests set every value they check themselves; they do not pin the values of the
shipped configs (data, batch size, loss weights, clipping, callbacks), which are experiment choices."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate as hydra_instantiate

from fm4tag.configs import __path__ as cfg_path
from fm4tag.utils.model_builders import build_aggregator, build_encoders, build_head


def _cfg(*overrides):
    with initialize_config_dir(config_dir=str(cfg_path[0]), version_base=None):
        return compose(config_name='default', overrides=list(overrides))


@pytest.mark.parametrize('arm', ['finetune', 'frozen'])
def test_pretrained_arms_require_a_checkpoint(arm):
    from fm4tag.runner.run import run

    cfg = _cfg(f'arm={arm}', 'encoder_ckpt=null', 'require_encoder_ckpt=true')
    with pytest.raises(ValueError, match='require_encoder_ckpt'):
        run(cfg)


@pytest.mark.parametrize('arm', ['scratch', 'finetune', 'frozen'])
def test_downstream_arms_build(arm):
    cfg = _cfg(f'arm={arm}', 'encoder_ckpt=/unused.ckpt')
    encoders = build_encoders(cfg)
    aggregator = build_aggregator(cfg, encoders)
    module = hydra_instantiate(
        cfg.finetune,
        encoders=encoders,
        aggregator=aggregator,
        head=build_head(cfg, aggregator),
        views=[hydra_instantiate(v) for v in cfg.views],
        n_classes=len(cfg.variables[cfg.global_object].unique_labels),
        _convert_='all',
    )
    assert module.freeze_encoders == cfg.finetune.freeze_encoders
    assert module.chunk_size == cfg.chunk_size


@pytest.mark.parametrize('arm', ['pretrain', 'scratch'])
def test_run_builds_the_module_with_the_optimizer_block(arm, tmp_path, monkeypatch):
    import lightning as L

    from fm4tag.runner.run import run

    built = {}
    monkeypatch.setattr(
        L.Trainer, 'fit', lambda self, module, *a, **k: built.update(module=module)
    )
    cfg = _cfg(
        f'arm={arm}',
        'logger=csv',
        'callbacks=smoke',
        f'output_dir={tmp_path}',
        'trainer.accelerator=cpu',
        'trainer.precision=32-true',
        'norm_dict_path=null',
        'class_dict_path=null',
        'optimizer.lr=0.123',
        'optimizer.warmup_epochs=1.5',
        'finetune.backbone_lr=null',
    )
    run(cfg)
    m = built['module']
    assert m.lr == 0.123
    for key in (
        'weight_decay',
        'schedule',
        'warmup_frac',
        'warmup_epochs',
        'decay_frac',
    ):
        assert getattr(m, key) == cfg.optimizer[key], key
    assert m.warmup_epochs == 1.5
    if arm == 'scratch':
        assert m.backbone_lr == 0.123  # null = optimizer.lr


def test_pretrain_arm_builds():
    cfg = _cfg('arm=pretrain', 'pretrain.loss_weights.jet_contrastive=1.0')
    encoders = build_encoders(cfg)
    module = hydra_instantiate(
        cfg.pretrain,
        encoders=encoders,
        aggregator=build_aggregator(cfg, encoders),
        views=[hydra_instantiate(v) for v in cfg.views],
        _convert_='all',
    )
    assert module.chunk_size == cfg.chunk_size
    assert any(p.requires_grad for p in module.aggregator.parameters())  # P1


@pytest.mark.parametrize('arm', ['pretrain', 'scratch', 'finetune', 'frozen'])
def test_the_qk_norm_switch_reaches_every_row_attention(arm):
    from fm4tag.models.attention import RowAttention

    attns = [
        m
        for m in build_encoders(_cfg(f'arm={arm}', 'row_qk_norm=true')).modules()
        if isinstance(m, RowAttention)
    ]
    assert attns and all(m.qk_norm for m in attns)


def test_qk_norm_is_off_for_run_configs_without_the_switch():
    """Run configs older than row_qk_norm rebuild without q_norm/k_norm, so their
    checkpoints load (the encoder files read it with a false default)."""
    from omegaconf import open_dict

    cfg = _cfg('arm=scratch')
    with open_dict(cfg):
        del cfg.row_qk_norm
    assert not any('q_norm' in k for k in build_encoders(cfg).state_dict())
