"""Tests for the EmbeddingMetrics callback (single process)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fm4tag.augmentations import Compose, CutMix, FeatureDropout
from fm4tag.callbacks import EmbeddingMetrics
from fm4tag.losses import MultiViewSupConLoss
from fm4tag.models import Encoder, GlobalEncoder, TransformerAggregator
from fm4tag.modules import PretrainModule

_CATEGORIES = [3, 5, 2]
_NUM_CONTINUOUS = 4
_DIM = 16


@pytest.fixture()
def module():
    encoders = torch.nn.ModuleDict(
        {
            'jets': GlobalEncoder(num_features=2, feature_dim=_DIM, dim=_DIM),
            'tracks': Encoder(
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
            ),
        }
    )
    aggregator = TransformerAggregator(
        global_dim=encoders['jets'].projector.layers[-1].out_features,
        const_dims=[encoders['tracks'].projector.layers[-1].out_features],
        depth=1,
        heads=2,
        dim_head=8,
        ff_mult=1,
    )
    return PretrainModule(
        encoders=encoders,
        aggregator=aggregator,
        views=[Compose([CutMix(lam=0.7)]), Compose([FeatureDropout(corrupt_frac=0.3)])],
        global_object='jets',
        constituent_objects=['tracks'],
        loss_weights={'contrastive': 1.0},
        contrastive_loss=MultiViewSupConLoss(temperature=0.1),
        jet_contrastive_loss=MultiViewSupConLoss(temperature=0.1),
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


def _trainer_stub(world_size: int = 1, sanity_checking: bool = False):
    return SimpleNamespace(world_size=world_size, sanity_checking=sanity_checking)


def _capture_logs(module) -> dict[str, float]:
    logged: dict[str, float] = {}
    module.log = lambda name, value, **_kw: logged.update({name: float(value)})
    return logged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_split_raises():
    with pytest.raises(ValueError, match='subset'):
        EmbeddingMetrics(splits=['test'])


def test_unknown_embedding_name_raises(module):
    cb = EmbeddingMetrics(embeddings=['does_not_exist'])
    with pytest.raises(ValueError, match='does_not_exist'):
        cb.on_validation_batch_end(_trainer_stub(), module, None, _make_batch(), 0)


# ---------------------------------------------------------------------------
# End-to-end: accumulate over batches, log at epoch end
# ---------------------------------------------------------------------------


def test_logs_all_embeddings_and_metrics(module):
    cb = EmbeddingMetrics(metrics=['uniformity', 'effective_rank'], splits=['val'])
    trainer = _trainer_stub()
    logged = _capture_logs(module)

    for _ in range(2):
        cb.on_validation_batch_end(trainer, module, None, _make_batch(), 0)
    cb.on_validation_epoch_end(trainer, module)

    expected = {
        f'val/{name}/{metric}'
        for name in ('jets_embedding', 'tracks_embedding', 'aggregator')
        for metric in ('uniformity', 'effective_rank')
    }
    assert set(logged) == expected
    assert all(torch.isfinite(torch.tensor(v)) for v in logged.values())
    # Buffers must be reset for the next epoch.
    assert len(cb._store['val']) == 0
    assert 'val' not in cb._names


def test_embeddings_filter(module):
    cb = EmbeddingMetrics(metrics=['uniformity'], embeddings=['aggregator'])
    trainer = _trainer_stub()
    logged = _capture_logs(module)

    cb.on_validation_batch_end(trainer, module, None, _make_batch(), 0)
    cb.on_validation_epoch_end(trainer, module)

    assert set(logged) == {'val/aggregator/uniformity'}


def test_quota_caps_accumulation(module):
    cb = EmbeddingMetrics(n_samples=6)
    trainer = _trainer_stub()

    for _ in range(3):  # 3 batches of B=4 jets → would be 12 without the cap
        cb.on_validation_batch_end(trainer, module, None, _make_batch(B=4), 0)

    for name in cb._names['val']:
        assert cb._count(cb._store['val'][name]) <= 6


def test_sanity_checking_logs_nothing(module):
    cb = EmbeddingMetrics()
    trainer = _trainer_stub(sanity_checking=True)
    logged = _capture_logs(module)

    cb.on_validation_batch_end(trainer, module, None, _make_batch(), 0)
    cb.on_validation_epoch_end(trainer, module)

    assert logged == {}


def test_train_split_restores_training_mode(module):
    cb = EmbeddingMetrics(splits=['train'])
    trainer = _trainer_stub()
    logged = _capture_logs(module)

    module.train()
    cb.on_train_batch_end(trainer, module, None, _make_batch(), 0)
    assert module.training, 'module must be back in train mode after accumulation'

    cb.on_train_epoch_end(trainer, module)
    assert any(k.startswith('train/') for k in logged)
