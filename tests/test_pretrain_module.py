"""Tests for fm4tag.modules.PretrainModule (weights-driven loss composition)."""

from __future__ import annotations

import pytest
import torch

from fm4tag.augmentations import Compose, CutMix, FeatureDropout, GaussianNoise
from fm4tag.losses import MultiViewSupConLoss
from fm4tag.models import Encoder, GlobalEncoder, TransformerAggregator
from fm4tag.modules import PretrainModule

_CATEGORIES = [3, 5, 2]  # three categorical features, cardinalities 3/5/2
_NUM_CONTINUOUS = 4
_DIM = 16

_WEIGHTS = {
    'contrastive': 0.6,
    'denoising_cat': 0.2,
    'denoising_con': 0.2,
    'jet_contrastive': 1.0,
}


@pytest.fixture()
def encoders():
    global_enc = GlobalEncoder(num_features=2, dim=_DIM, proj_out=_DIM)
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
def aggregator(encoders):
    global_dim = encoders['jets'].projector.layers[-1].out_features
    const_dims = [encoders['tracks'].projector.layers[-1].out_features]
    return TransformerAggregator(
        global_dim=global_dim,
        const_dims=const_dims,
        depth=1,
        heads=2,
        dim_head=8,
        ff_mult=1,
    )


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


def _make_module(encoders, aggregator, views, loss_weights=None):
    return PretrainModule(
        encoders=encoders,
        aggregator=aggregator,
        views=views,
        global_object='jets',
        constituent_objects=['tracks'],
        loss_weights=dict(_WEIGHTS) if loss_weights is None else loss_weights,
        contrastive_loss=MultiViewSupConLoss(temperature=0.1),
        jet_contrastive_loss=MultiViewSupConLoss(temperature=0.1),
        lr=1e-3,
    )


def _make_batch(B: int = 4, C: int = 8) -> dict:
    return {
        'global': torch.randn(B, 2),
        'constituents': {
            'tracks': {
                'categorical': torch.randint(0, 2, (B, C, len(_CATEGORIES))),
                'continuous': torch.randn(B, C, _NUM_CONTINUOUS),
                'valid': torch.ones(B, C, dtype=torch.bool),
            }
        },
    }


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_construction_uses_weights_as_is(encoders, aggregator, two_views):
    module = _make_module(encoders, aggregator, two_views)
    assert module.loss_weights == _WEIGHTS


def test_missing_weight_keys_default_to_zero(encoders, aggregator, two_views):
    module = _make_module(
        encoders, aggregator, two_views, loss_weights={'contrastive': 1.0}
    )
    assert module.loss_weights['denoising_cat'] == 0.0
    assert module.loss_weights['jet_contrastive'] == 0.0


def test_unknown_weight_key_raises(encoders, aggregator, two_views):
    with pytest.raises(ValueError, match='Unknown loss_weights'):
        _make_module(encoders, aggregator, two_views, loss_weights={'contrastve': 1.0})


def test_all_zero_weights_raise(encoders, aggregator, two_views):
    with pytest.raises(ValueError, match='At least one'):
        _make_module(encoders, aggregator, two_views, loss_weights={'contrastive': 0.0})


def test_requires_two_views_when_contrastive(encoders, aggregator):
    with pytest.raises(ValueError, match='at least 2 views'):
        _make_module(encoders, aggregator, [Compose([])])


def test_single_view_ok_for_denoising_only(encoders, aggregator):
    module = _make_module(
        encoders,
        aggregator,
        [Compose([CutMix(lam=0.7)])],
        loss_weights={'denoising_cat': 1.0, 'denoising_con': 1.0},
    )
    loss, log_dict = module._compute_loss(_make_batch())
    assert torch.isfinite(loss)
    assert 'tracks_embedding/loss_denoising_cat' in log_dict
    assert 'jets_embedding/loss_contrastive' not in log_dict


# ---------------------------------------------------------------------------
# Loss composition and logging keys
# ---------------------------------------------------------------------------


def test_compute_loss_returns_finite_scalar(encoders, aggregator, two_views):
    module = _make_module(encoders, aggregator, two_views)
    loss, _ = module._compute_loss(_make_batch())
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_log_dict_uses_component_nomenclature(encoders, aggregator, two_views):
    module = _make_module(encoders, aggregator, two_views)
    _, log_dict = module._compute_loss(_make_batch())
    assert set(log_dict) == {
        'loss',
        'jets_embedding/loss_contrastive',
        'jets_embedding/loss_denoising_con',
        'tracks_embedding/loss_contrastive',
        'tracks_embedding/loss_denoising_cat',
        'tracks_embedding/loss_denoising_con',
        'aggregator/loss_contrastive',
    }


def test_zero_weight_components_not_computed_or_logged(encoders, aggregator, two_views):
    module = _make_module(
        encoders,
        aggregator,
        two_views,
        loss_weights={'contrastive': 1.0, 'jet_contrastive': 0.0},
    )
    _, log_dict = module._compute_loss(_make_batch())
    assert 'aggregator/loss_contrastive' not in log_dict
    assert 'tracks_embedding/loss_denoising_cat' not in log_dict
    assert 'jets_embedding/loss_denoising_con' not in log_dict


def test_scaling_weights_scales_loss(encoders, aggregator, two_views):
    """Weights are applied as-is: doubling every weight doubles the loss."""
    weights_a = {
        'contrastive': 2.0,
        'denoising_cat': 1.0,
        'denoising_con': 1.0,
        'jet_contrastive': 2.0,
    }
    weights_b = {k: w / 2 for k, w in weights_a.items()}
    module_a = _make_module(encoders, aggregator, two_views, loss_weights=weights_a)
    module_b = _make_module(encoders, aggregator, two_views, loss_weights=weights_b)

    batch = _make_batch()
    torch.manual_seed(0)  # augmentations are stochastic — align the RNG
    loss_a, _ = module_a._compute_loss(batch)
    torch.manual_seed(0)
    loss_b, _ = module_b._compute_loss(batch)
    assert torch.isclose(loss_a, 2 * loss_b, atol=1e-6)


def test_three_views_loss(encoders, aggregator, three_views):
    module = _make_module(encoders, aggregator, three_views)
    loss, _ = module._compute_loss(_make_batch())
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_to_encoders_and_aggregator(encoders, aggregator, two_views):
    module = _make_module(encoders, aggregator, two_views)
    loss, _ = module._compute_loss(_make_batch())
    loss.backward()

    enc_grads = [p.grad for p in module.encoders.parameters() if p.grad is not None]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads)

    agg_grads = [p.grad for p in module.aggregator.parameters() if p.grad is not None]
    assert agg_grads and any(g.abs().sum() > 0 for g in agg_grads)


def test_no_aggregator_gradient_when_jet_weight_zero(encoders, aggregator, two_views):
    module = _make_module(
        encoders, aggregator, two_views, loss_weights={'contrastive': 1.0}
    )
    loss, _ = module._compute_loss(_make_batch())
    loss.backward()
    assert all(p.grad is None for p in module.aggregator.parameters())


# ---------------------------------------------------------------------------
# forward(): clean embeddings for eval / downstream
# ---------------------------------------------------------------------------


def test_forward_returns_all_embeddings(encoders, aggregator, two_views):
    module = _make_module(encoders, aggregator, two_views)
    B, C = 4, 8
    out = module(_make_batch(B=B, C=C))
    assert set(out) == {'jets_embedding', 'tracks_embedding', 'aggregator'}
    assert out['jets_embedding'].shape[0] == B
    assert out['tracks_embedding'].shape[0] == B * C  # all valid in this batch
    assert out['aggregator'].shape[0] == B


# ---------------------------------------------------------------------------
# DDP safety: zero-weight heads are frozen at construction
# ---------------------------------------------------------------------------


def test_contrastive_only_freezes_unused_heads(encoders, aggregator, two_views):
    module = _make_module(
        encoders, aggregator, two_views, loss_weights={'contrastive': 1.0}
    )
    tracks = module.encoders['tracks']
    jets = module.encoders['jets']
    assert not any(p.requires_grad for p in tracks.cat_reconstructor.parameters())
    assert not any(p.requires_grad for p in tracks.con_reconstructor.parameters())
    assert not any(p.requires_grad for p in jets.reconstructor.parameters())
    assert not any(p.requires_grad for p in module.aggregator.parameters())
    # Projectors feed the active contrastive loss.
    assert all(p.requires_grad for p in tracks.projector.parameters())


def test_denoising_only_freezes_projectors_and_aggregator(
    encoders, aggregator, two_views
):
    module = _make_module(
        encoders,
        aggregator,
        two_views,
        loss_weights={'denoising_cat': 1.0, 'denoising_con': 1.0},
    )
    tracks = module.encoders['tracks']
    assert not any(p.requires_grad for p in tracks.projector.parameters())
    assert not any(p.requires_grad for p in module.aggregator.parameters())
    assert all(p.requires_grad for p in tracks.cat_reconstructor.parameters())
    assert all(p.requires_grad for p in tracks.con_reconstructor.parameters())


def test_all_weights_active_freezes_nothing(encoders, aggregator, two_views):
    module = _make_module(encoders, aggregator, two_views)
    assert all(p.requires_grad for p in module.parameters())
