"""Tests for BasePretrainModule and ContrastiveDenoisingModule."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from fm4tag.augmentations import Compose, CutMix, FeatureDropout, GaussianNoise
from fm4tag.models import Encoder, GlobalEncoder, JetAggregator
from fm4tag.modules import (
    ContrastiveDenoisingModule,
    ContrastiveTermAdapter,
    DenoisingTermAdapter,
    PretrainLoss,
)


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

_CATEGORIES = [3, 5, 2]  # three categorical features, cardinalities 3/5/2
_NUM_CONTINUOUS = 4
_DIM = 16


@pytest.fixture()
def minimal_cfg():
    return OmegaConf.create(
        {
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
        }
    )


@pytest.fixture()
def encoders():
    global_enc = GlobalEncoder(num_features=2, feature_dim=_DIM, dim=_DIM)
    track_enc = Encoder(
        categories=_CATEGORIES,
        num_continuous=_NUM_CONTINUOUS,
        dim=_DIM,
        layers=[
            {
                'type': 'col',
                'depth': 1,
                'heads': 2,
                'dim_head': 8,
                'ff_mult': 1,
                'attn_dropout': 0.0,
                'ff_dropout': 0.0,
            }
        ],
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


@pytest.fixture()
def aggregator(encoders):
    global_dim = encoders['jets'].projector.layers[-1].out_features
    const_dims = [encoders['tracks'].projector.layers[-1].out_features]
    return JetAggregator(
        global_dim=global_dim,
        const_dims=const_dims,
        depth=1,
        heads=2,
        dim_head=8,
        ff_mult=1,
    )


@pytest.fixture()
def loss():
    return PretrainLoss(
        terms=[
            ContrastiveTermAdapter(
                temperature=0.1, loss_type='out', include_pos_in_denom=True
            ),
            DenoisingTermAdapter(weight_cat=0.2, weight_con=0.2),
        ],
        weights=[0.6, 1.0],
    )


def _make_batch(B: int = 4, C: int = 8) -> dict:
    """Build a minimal batch dict matching the expected format."""
    return {
        'global': {
            'categorical': torch.zeros(B, 0, dtype=torch.long),
            'continuous': torch.randn(B, 2),
        },
        'constituents': {
            'tracks': {
                'categorical': torch.randint(0, 2, (B, C, len(_CATEGORIES))),
                'continuous': torch.randn(B, C, _NUM_CONTINUOUS),
                'valid': torch.ones(B, C, dtype=torch.bool),
            }
        },
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction(encoders, aggregator, two_views, loss, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    assert len(module.views) == 2


def test_requires_two_views(encoders, aggregator, loss, minimal_cfg):
    with pytest.raises(ValueError, match='at least 2'):
        ContrastiveDenoisingModule(
            encoders=encoders,
            aggregator=aggregator,
            views=[Compose([])],
            loss=loss,
            cfg=minimal_cfg,
        )


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


def test_compute_loss_returns_scalar(
    encoders, aggregator, two_views, loss, minimal_cfg
):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    batch = _make_batch()
    loss, log_dict = module._compute_loss(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_compute_loss_log_dict_has_expected_keys(
    encoders, aggregator, two_views, loss, minimal_cfg
):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    batch = _make_batch()
    _, log_dict = module._compute_loss(batch)
    assert 'loss' in log_dict
    assert 'jets/loss_contrastive' in log_dict
    assert 'tracks/loss_contrastive' in log_dict


def test_compute_loss_denoising_keys_present(
    encoders, aggregator, two_views, loss, minimal_cfg
):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    batch = _make_batch()
    _, log_dict = module._compute_loss(batch)
    assert 'jets/loss_denoising_con' in log_dict
    assert 'tracks/loss_denoising_cat' in log_dict
    assert 'tracks/loss_denoising_con' in log_dict


def test_categorical_jets_denoising_and_gradients():
    """Global object with categorical features: cat denoising fires + grads flow."""
    g_categories = [3, 4]  # two categorical jet features
    n_g_cont = 2
    global_enc = GlobalEncoder(
        num_features=n_g_cont, feature_dim=_DIM, dim=_DIM, categories=g_categories
    )
    track_enc = Encoder(
        categories=_CATEGORIES,
        num_continuous=_NUM_CONTINUOUS,
        dim=_DIM,
        layers=[{'type': 'col', 'depth': 1, 'heads': 2, 'dim_head': 8, 'ff_mult': 1}],
    )
    encoders = torch.nn.ModuleDict({'jets': global_enc, 'tracks': track_enc})

    aggregator = JetAggregator(
        global_dim=global_enc.projector.layers[-1].out_features,
        const_dims=[track_enc.projector.layers[-1].out_features],
        depth=1,
        heads=2,
        dim_head=8,
        ff_mult=1,
    )
    views = [Compose([CutMix(lam=0.7)]), Compose([FeatureDropout(corrupt_frac=0.3)])]
    loss = PretrainLoss(
        terms=[
            ContrastiveTermAdapter(temperature=0.1),
            DenoisingTermAdapter(weight_cat=0.2, weight_con=0.2),
        ],
        weights=[0.6, 1.0],
    )
    cfg = OmegaConf.create(
        {
            'global_object': 'jets',
            'constituent_objects': ['tracks'],
            'eval': {'enabled': False, 'splits': ['val'], 'n_samples': 32, 'metrics': []},
            'optimizer': {'lr': 1e-3, 'weight_decay': 1e-5},
        }
    )
    module = ContrastiveDenoisingModule(
        encoders=encoders, aggregator=aggregator, views=views, loss=loss, cfg=cfg
    )

    B, C = 4, 8
    batch = {
        'global': {
            'categorical': torch.stack(
                [torch.randint(0, n, (B,)) for n in g_categories], dim=1
            ),  # (B, 2) long, valid class indices
            'continuous': torch.randn(B, n_g_cont),
        },
        'constituents': {
            'tracks': {
                'categorical': torch.randint(0, 2, (B, C, len(_CATEGORIES))),
                'continuous': torch.randn(B, C, _NUM_CONTINUOUS),
                'valid': torch.ones(B, C, dtype=torch.bool),
            }
        },
    }

    total, log_dict = module._compute_loss(batch)
    assert torch.isfinite(total)
    # Categorical denoising now fires for the global object too.
    assert 'jets/loss_denoising_cat' in log_dict
    assert 'jets/loss_denoising_con' in log_dict

    total.backward()
    assert global_enc.embeds.weight.grad is not None
    assert global_enc.embeds.weight.grad.abs().sum() > 0


def test_compute_loss_no_denoising_when_lam_zero(encoders, aggregator, two_views):
    cfg = OmegaConf.create(
        {
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
            'eval': {
                'enabled': False,
                'splits': ['val'],
                'n_samples': 32,
                'metrics': [],
            },
            'optimizer': {'lr': 1e-3, 'weight_decay': 1e-5},
        }
    )
    loss = PretrainLoss(
        terms=[
            ContrastiveTermAdapter(
                temperature=0.1, loss_type='out', include_pos_in_denom=True
            )
        ],
        weights=[1.0],
    )
    module = ContrastiveDenoisingModule(
        encoders=encoders, aggregator=aggregator, views=two_views, loss=loss, cfg=cfg
    )
    batch = _make_batch()
    _, log_dict = module._compute_loss(batch)
    assert 'jets/loss_denoising_con' not in log_dict
    assert 'tracks/loss_denoising_cat' not in log_dict


def test_three_views_loss(encoders, aggregator, three_views, loss, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=three_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    batch = _make_batch()
    loss, _ = module._compute_loss(batch)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Gradient flow through the full loss
# ---------------------------------------------------------------------------


def test_gradients_flow_to_encoder(encoders, aggregator, two_views, loss, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
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


def test_project_for_eval_global(encoders, aggregator, two_views, loss, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    batch = _make_batch()
    z = module._project_for_eval(batch, 'jets')
    assert z is not None
    assert z.shape[0] == batch['global']['continuous'].shape[0]


def test_project_for_eval_constituent(
    encoders, aggregator, two_views, loss, minimal_cfg
):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    batch = _make_batch(B=4, C=8)
    z = module._project_for_eval(batch, 'tracks')
    assert z is not None
    assert z.ndim == 2  # (N_valid, proj_dim)


def test_project_for_eval_empty_valid(
    encoders, aggregator, two_views, loss, minimal_cfg
):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    batch = _make_batch()
    batch['constituents']['tracks']['valid'][:] = False
    z = module._project_for_eval(batch, 'tracks')
    assert z is None


# ---------------------------------------------------------------------------
# predict_step
# ---------------------------------------------------------------------------


def test_predict_step_structure(encoders, aggregator, two_views, loss, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
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


def test_predict_step_view_has_pre_flatten_and_raw(
    encoders, aggregator, two_views, loss, minimal_cfg
):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    batch = _make_batch()
    out = module.predict_step(batch, 0)
    view = out['constituents']['tracks']['views'][0]
    assert 'pre_flatten' in view
    assert 'raw' in view
    assert 'categorical' in view['raw']
    assert 'continuous' in view['raw']


def test_predict_step_output_is_cpu(encoders, aggregator, two_views, loss, minimal_cfg):
    module = ContrastiveDenoisingModule(
        encoders=encoders,
        aggregator=aggregator,
        views=two_views,
        loss=loss,
        cfg=minimal_cfg,
    )
    batch = _make_batch()
    out = module.predict_step(batch, 0)
    assert out['jets']['original']['continuous'].device.type == 'cpu'
    assert out['jets']['views'][0]['raw']['continuous'].device.type == 'cpu'
    view_raw = out['constituents']['tracks']['views'][0]['raw']
    assert view_raw['continuous'].device.type == 'cpu'
