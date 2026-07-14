"""Tests for fm4tag.modules.FinetuneModule (weights-driven loss composition)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fm4tag.augmentations import Compose, CutMix, FeatureDropout
from fm4tag.losses import MultiViewSupConLoss
from fm4tag.models import (
    Encoder,
    GlobalEncoder,
    MultiStreamClassifierHead,
    TransformerAggregator,
)
from fm4tag.modules import FinetuneModule

_CATEGORIES = [3, 5, 2]
_NUM_CONTINUOUS = 4
_DIM = 16
_N_CLASSES = 4


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
def head(aggregator):
    return MultiStreamClassifierHead(jet_dim=aggregator.out_dim, y_dim=_N_CLASSES)


@pytest.fixture()
def two_views():
    return [
        Compose([CutMix(lam=0.7)]),
        Compose([FeatureDropout(corrupt_frac=0.3)]),
    ]


def _make_module(encoders, aggregator, head, views, loss_weights=None, **kwargs):
    return FinetuneModule(
        encoders=encoders,
        aggregator=aggregator,
        head=head,
        views=views,
        global_object='jets',
        constituent_objects=['tracks'],
        loss_weights=(
            {'cross_entropy': 1.0, 'jet_contrastive': 0.3}
            if loss_weights is None
            else loss_weights
        ),
        jet_contrastive_loss=MultiViewSupConLoss(temperature=0.1),
        n_classes=_N_CLASSES,
        lr=1e-3,
        **kwargs,
    )


def _make_batch(B: int = 4, C: int = 8) -> dict:
    return {
        'label': torch.randint(0, _N_CLASSES, (B,)),
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


def test_construction_normalises_weights(encoders, aggregator, head, two_views):
    module = _make_module(encoders, aggregator, head, two_views)
    assert sum(module.loss_weights.values()) == pytest.approx(1.0)
    assert module.loss_weights['cross_entropy'] == pytest.approx(1.0 / 1.3)


def test_unknown_weight_key_raises(encoders, aggregator, head, two_views):
    with pytest.raises(ValueError, match='Unknown loss_weights'):
        _make_module(
            encoders, aggregator, head, two_views, loss_weights={'contrastive': 1.0}
        )


def test_all_zero_weights_raise(encoders, aggregator, head, two_views):
    with pytest.raises(ValueError, match='At least one'):
        _make_module(
            encoders, aggregator, head, two_views, loss_weights={'cross_entropy': 0.0}
        )


def test_jet_contrastive_requires_two_views(encoders, aggregator, head):
    with pytest.raises(ValueError, match='at least 2 views'):
        _make_module(encoders, aggregator, head, [])


def test_no_views_needed_for_pure_cross_entropy(encoders, aggregator, head):
    module = _make_module(
        encoders, aggregator, head, [], loss_weights={'cross_entropy': 1.0}
    )
    loss, _, log_dict = module._compute_loss(_make_batch())
    assert torch.isfinite(loss)
    assert 'aggregator/loss_contrastive' not in log_dict


def test_class_weights_registered_as_buffer(encoders, aggregator, head, two_views):
    module = _make_module(
        encoders, aggregator, head, two_views, class_weights=[1.0, 2.0, 0.5, 1.5]
    )
    assert isinstance(module.class_weights, torch.Tensor)
    assert 'class_weights' in dict(module.named_buffers())


# ---------------------------------------------------------------------------
# Loss composition and logging keys
# ---------------------------------------------------------------------------


def test_compute_loss_returns_finite_scalar_and_logits(
    encoders, aggregator, head, two_views
):
    module = _make_module(encoders, aggregator, head, two_views)
    B = 4
    loss, logits, _ = module._compute_loss(_make_batch(B=B))
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert logits.shape == (B, _N_CLASSES)


def test_log_dict_uses_component_nomenclature(encoders, aggregator, head, two_views):
    module = _make_module(encoders, aggregator, head, two_views)
    _, _, log_dict = module._compute_loss(_make_batch())
    assert set(log_dict) == {'loss', 'head/loss_ce', 'aggregator/loss_contrastive'}


def test_proportional_weights_give_identical_loss(
    encoders, aggregator, head, two_views
):
    weights_a = {'cross_entropy': 2.0, 'jet_contrastive': 0.6}
    weights_b = {'cross_entropy': 1.0, 'jet_contrastive': 0.3}
    module_a = _make_module(encoders, aggregator, head, two_views, weights_a)
    module_b = _make_module(encoders, aggregator, head, two_views, weights_b)

    batch = _make_batch()
    torch.manual_seed(0)  # augmentations are stochastic — align the RNG
    loss_a, _, _ = module_a._compute_loss(batch)
    torch.manual_seed(0)
    loss_b, _, _ = module_b._compute_loss(batch)
    assert torch.isclose(loss_a, loss_b, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_everywhere(encoders, aggregator, head, two_views):
    module = _make_module(encoders, aggregator, head, two_views)
    loss, _, _ = module._compute_loss(_make_batch())
    loss.backward()
    for name, part in (
        ('backbone', module.backbone),
        ('aggregator', module.aggregator),
        ('head', module.head),
    ):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), f'no grads in {name}'


# ---------------------------------------------------------------------------
# Optimizer / staged unfreeze
# ---------------------------------------------------------------------------


def _n_params(params) -> int:
    return sum(p.numel() for p in params)


def _attach_trainer(module, *, steps=100, max_epochs=10):
    module._trainer = SimpleNamespace(
        estimated_stepping_batches=steps, max_epochs=max_epochs
    )


def test_optimizer_has_backbone_and_head_groups(
    encoders, aggregator, head, two_views
):
    module = _make_module(encoders, aggregator, head, two_views, backbone_lr=1e-5)
    _attach_trainer(module)
    opt = module.configure_optimizers()['optimizer']
    assert len(opt.param_groups) == 2
    # Frozen reconstructor heads are excluded from the optimizer.
    n_pretrained = _n_params(
        p for p in module.backbone.parameters() if p.requires_grad
    ) + _n_params(p for p in module.aggregator.parameters() if p.requires_grad)
    assert _n_params(opt.param_groups[0]['params']) == n_pretrained
    assert _n_params(opt.param_groups[1]['params']) == _n_params(
        module.head.parameters()
    )
    # Group 0 = backbone (base backbone_lr), group 1 = head (base lr); the live
    # 'lr' is scaled by the schedule, the pristine value is kept in 'initial_lr'.
    assert opt.param_groups[0]['initial_lr'] == pytest.approx(1e-5)
    assert opt.param_groups[1]['initial_lr'] == pytest.approx(1e-3)


def test_staged_unfreeze_holds_backbone_at_zero_lr(
    encoders, aggregator, head, two_views
):
    # 100 steps over 10 epochs -> unfreeze at epoch 5 == step 50.
    module = _make_module(
        encoders, aggregator, head, two_views, backbone_lr=1e-5, unfreeze_at_epoch=5
    )
    _attach_trainer(module, steps=100, max_epochs=10)
    cfg = module.configure_optimizers()
    opt = cfg['optimizer']
    scheduler = cfg['lr_scheduler']['scheduler']

    assert opt.param_groups[0]['lr'] == 0.0    # backbone frozen at step 0
    assert opt.param_groups[1]['lr'] > 0.0     # head warming up
    for _ in range(49):
        scheduler.step()
    assert opt.param_groups[0]['lr'] == 0.0    # still frozen at step 49
    scheduler.step()                            # step 50 == unfreeze_at_epoch
    assert opt.param_groups[0]['lr'] > 0.0     # backbone now training


def test_unfreeze_at_epoch_zero_trains_backbone_from_start(
    encoders, aggregator, head, two_views
):
    module = _make_module(
        encoders, aggregator, head, two_views, backbone_lr=1e-3, unfreeze_at_epoch=0
    )
    _attach_trainer(module)
    opt = module.configure_optimizers()['optimizer']
    assert opt.param_groups[0]['lr'] > 0.0     # trained from step 0


# ---------------------------------------------------------------------------
# forward(): same embeddings contract as PretrainModule
# ---------------------------------------------------------------------------


def test_forward_returns_embeddings_contract(encoders, aggregator, head, two_views):
    module = _make_module(encoders, aggregator, head, two_views)
    B, C = 4, 8
    out = module(_make_batch(B=B, C=C))
    assert set(out) == {'jets_embedding', 'tracks_embedding', 'aggregator'}
    assert out['aggregator'].shape[0] == B


def test_predict_step_returns_probabilities(encoders, aggregator, head, two_views):
    module = _make_module(encoders, aggregator, head, two_views)
    B = 4
    probs = module.predict_step(_make_batch(B=B), 0)
    assert probs.shape == (B, _N_CLASSES)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# DDP safety: pretrain-only reconstruction heads are frozen
# ---------------------------------------------------------------------------


def test_reconstructor_heads_frozen(encoders, aggregator, head, two_views):
    module = _make_module(encoders, aggregator, head, two_views)
    for enc in module.backbone.values():
        for name in ('reconstructor', 'cat_reconstructor', 'con_reconstructor'):
            if hasattr(enc, name):
                heads = getattr(enc, name)
                assert not any(p.requires_grad for p in heads.parameters())
    assert all(p.requires_grad for p in module.head.parameters())
    assert all(p.requires_grad for p in module.aggregator.parameters())
