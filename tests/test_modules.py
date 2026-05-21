"""Tests for BasePretrainModule and ContrastiveDenoisingModule."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from fm4tag.augmentations import Compose, CutMix, FeatureDropout, GaussianNoise
from fm4tag.models import Encoder, GlobalEncoder
from fm4tag.modules import ContrastiveDenoisingModule


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

_CATEGORIES = [3, 5, 2]   # three categorical features, cardinalities 3/5/2
_NUM_CONTINUOUS = 4
_DIM = 16


@pytest.fixture()
def minimal_cfg():
    return OmegaConf.create({
        'global_object': 'jets',
        'constituent_objects': ['tracks'],
        'pretrain': {
            '_target_': 'fm4tag.modules.ContrastiveDenoisingModule',
            'nce_temp': 0.1,
            'loss_type': 'out',
            'include_pos_in_denom': True,
            'lam_contrastive': 0.6,
            'lam_denoising_cat': 0.2,
            'lam_denoising_con': 0.2,
        },
        'eval': {
            'enabled': False,
            'splits': ['val'],
            'n_samples': 32,
            'metrics': ['uniformity'],
        },
        'optimizer': {'lr': 1e-3, 'weight_decay': 1e-5},
    })


@pytest.fixture()
def encoders():
    global_enc = GlobalEncoder(num_features=2, dim=_DIM)
    track_enc = Encoder(
        categories=_CATEGORIES,
        num_continuous=_NUM_CONTINUOUS,
        dim=_DIM,
        layers=[{'type': 'col', 'depth': 1, 'heads': 2, 'dim_head': 8,
                 'ff_mult': 1, 'attn_dropout': 0.0, 'ff_dropout': 0.0}],
    )
    return torch.nn.ModuleDict({'jets': global_enc, 'tracks': track_enc})


@pytest.fixture()
def two_views():
    return [
        Compose([CutMix(lam=0.7)]),
        Compose([FeatureDropout(corrupt_frac=0.3)]),
    ]


@pytest.fixture()
def three_views():
    return [
        Compose([CutMix(lam=0.7)]),
        Compose([FeatureDropout(corrupt_frac=0.3)]),
        Compose([GaussianNoise(sigma=0.1, space='embedding')]),
    ]


def _make_batch(B: int = 4, C: int = 8) -> dict:
    """Build a minimal batch dict matching the expected format."""
    return {
        'global': torch.randn(B, 2),
        'constituents': {
            'tracks': {
                'categorical': torch.randint(0, 2, (B, C, len(_CATEGORIES))),
                'continuous':  torch.randn(B, C, _NUM_CONTINUOUS),
                'valid':        torch.ones(B, C, dtype=torch.bool),
            }
        },
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    assert len(module.views) == 2


def test_requires_two_views(encoders, minimal_cfg):
    with pytest.raises(ValueError, match='at least 2'):
        ContrastiveDenoisingModule(
            encoders=encoders,
            views=[Compose([])],
            cfg=minimal_cfg,
        )


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------

def test_compute_loss_returns_scalar(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    loss, log_dict = module._compute_loss(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_compute_loss_log_dict_has_expected_keys(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    _, log_dict = module._compute_loss(batch)
    assert 'loss' in log_dict
    assert 'jets/loss_contrastive' in log_dict
    assert 'tracks/loss_contrastive' in log_dict


def test_compute_loss_denoising_keys_present(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    _, log_dict = module._compute_loss(batch)
    assert 'jets/loss_denoising_con' in log_dict
    assert 'tracks/loss_denoising_cat' in log_dict
    assert 'tracks/loss_denoising_con' in log_dict


def test_compute_loss_no_denoising_when_lam_zero(encoders, two_views):
    cfg = OmegaConf.create({
        'global_object': 'jets',
        'constituent_objects': ['tracks'],
        'pretrain': {
            '_target_': 'fm4tag.modules.ContrastiveDenoisingModule',
            'nce_temp': 0.1,
            'loss_type': 'out',
            'include_pos_in_denom': True,
            'lam_contrastive': 1.0,
            'lam_denoising_cat': 0.0,
            'lam_denoising_con': 0.0,
        },
        'eval': {'enabled': False, 'splits': ['val'], 'n_samples': 32,
                 'metrics': []},
        'optimizer': {'lr': 1e-3, 'weight_decay': 1e-5},
    })
    module = ContrastiveDenoisingModule(encoders=encoders, views=two_views, cfg=cfg)
    batch = _make_batch()
    _, log_dict = module._compute_loss(batch)
    assert 'jets/loss_denoising_con' not in log_dict
    assert 'tracks/loss_denoising_cat' not in log_dict


def test_three_views_loss(encoders, three_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=three_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    loss, _ = module._compute_loss(batch)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Gradient flow through the full loss
# ---------------------------------------------------------------------------

def test_gradients_flow_to_encoder(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    loss, _ = module._compute_loss(batch)
    loss.backward()

    # Check that at least some encoder parameters have non-zero gradients.
    grads = [p.grad for p in module.encoders.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


# ---------------------------------------------------------------------------
# _project_for_eval
# ---------------------------------------------------------------------------

def test_project_for_eval_global(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    z = module._project_for_eval(batch, 'jets')
    assert z is not None
    assert z.shape[0] == batch['global'].shape[0]


def test_project_for_eval_constituent(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch(B=4, C=8)
    z = module._project_for_eval(batch, 'tracks')
    assert z is not None
    assert z.ndim == 2   # (N_valid, proj_dim)


def test_project_for_eval_empty_valid(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    batch['constituents']['tracks']['valid'][:] = False
    z = module._project_for_eval(batch, 'tracks')
    assert z is None


# ---------------------------------------------------------------------------
# predict_step
# ---------------------------------------------------------------------------

def test_predict_step_structure(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    out = module.predict_step(batch, 0)

    assert 'jets' in out
    assert 'original' in out['jets']
    assert 'views' in out['jets']
    assert len(out['jets']['views']) == 2

    assert 'constituents' in out
    assert 'tracks' in out['constituents']
    assert 'original' in out['constituents']['tracks']
    assert 'views' in out['constituents']['tracks']
    assert len(out['constituents']['tracks']['views']) == 2


def test_predict_step_view_has_pre_flatten_and_raw(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    out = module.predict_step(batch, 0)
    view = out['constituents']['tracks']['views'][0]
    assert 'pre_flatten' in view
    assert 'raw' in view
    assert 'categorical' in view['raw']
    assert 'continuous' in view['raw']


def test_predict_step_output_is_cpu(encoders, two_views, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders, views=two_views, cfg=minimal_cfg
    )
    batch = _make_batch()
    out = module.predict_step(batch, 0)
    assert out['jets']['original'].device.type == 'cpu'
    view_raw = out['constituents']['tracks']['views'][0]['raw']
    assert view_raw['continuous'].device.type == 'cpu'
