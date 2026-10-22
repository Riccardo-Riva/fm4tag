"""The SSL campaign config: every arm composes, builds, and means what it says."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate as hydra_instantiate

from fm4tag.configs import __path__ as cfg_path
from fm4tag.utils.model_builders import build_aggregator, build_encoders, build_head


def _cfg(*overrides):
    with initialize_config_dir(config_dir=str(cfg_path[0]), version_base=None):
        return compose(config_name='campaign_ssl', overrides=list(overrides))


def test_campaign_uses_the_disjoint_split():
    cfg = _cfg('arm=scratch')
    assert cfg.base_path.endswith('data_parallel_preprocess_251112_152051')
    assert cfg.pretrain_dataset_path.endswith('pretrain/output/pp_output_pretrain.h5')
    assert cfg.datamodule.batch_size % (8 * cfg.chunk_size) == 0


def test_pretrain_arm_is_contrastive_only():
    w = _cfg('arm=pretrain').pretrain.loss_weights
    assert (w.contrastive, w.denoising_cat, w.denoising_con) == (1.0, 0.0, 0.0)
    assert w.jet_contrastive == 0.0  # P0; P1 sets it on the command line


@pytest.mark.parametrize('arm', ['finetune', 'frozen'])
def test_pretrained_arms_require_a_checkpoint(arm):
    from fm4tag.runner.run import run

    cfg = _cfg(f'arm={arm}')
    assert cfg.encoder_ckpt is None and cfg.require_encoder_ckpt
    with pytest.raises(ValueError, match='require_encoder_ckpt'):
        run(cfg)


@pytest.mark.parametrize(
    'arm,frozen', [('scratch', False), ('finetune', False), ('frozen', True)]
)
def test_downstream_arms_build(arm, frozen):
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
    assert module.freeze_encoders is frozen
    assert module.chunk_size == cfg.chunk_size
    assert module.loss_weights['jet_contrastive'] == 0.0


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


def test_campaign_logs_gpu_memory():
    from fm4tag.callbacks import MemoryMonitorCallback
    from fm4tag.utils import instantiate_callbacks

    callbacks = instantiate_callbacks(_cfg('arm=scratch').callbacks)
    assert any(isinstance(cb, MemoryMonitorCallback) for cb in callbacks)


def test_campaign_logs_grad_norm():
    from fm4tag.callbacks import GradNormMonitorCallback
    from fm4tag.utils import instantiate_callbacks

    callbacks = instantiate_callbacks(_cfg('arm=scratch').callbacks)
    assert any(isinstance(cb, GradNormMonitorCallback) for cb in callbacks)


def test_only_pretraining_clips_gradients():
    assert _cfg('arm=pretrain').trainer.gradient_clip_val == 1.0
    for arm in ('scratch', 'finetune', 'frozen'):
        assert _cfg(f'arm={arm}').trainer.get('gradient_clip_val') is None


@pytest.mark.parametrize('arm', ['pretrain', 'scratch', 'finetune', 'frozen'])
def test_every_arm_has_qk_norm_in_its_row_attention(arm):
    from fm4tag.models.attention import RowAttention

    attns = [
        m
        for m in build_encoders(_cfg(f'arm={arm}')).modules()
        if isinstance(m, RowAttention)
    ]
    assert attns and all(m.qk_norm for m in attns)


def test_qk_norm_is_off_outside_the_campaign():
    """Older run configs rebuild without q_norm/k_norm, so their checkpoints load."""
    with initialize_config_dir(config_dir=str(cfg_path[0]), version_base=None):
        cfg = compose(config_name='default')
    assert cfg.row_qk_norm is False
    assert not any('q_norm' in k for k in build_encoders(cfg).state_dict())


def test_campaign_monitors_row_attention():
    from fm4tag.callbacks import RowAttentionMonitor
    from fm4tag.utils import instantiate_callbacks

    callbacks = instantiate_callbacks(_cfg('arm=pretrain').callbacks)
    assert any(isinstance(cb, RowAttentionMonitor) for cb in callbacks)
