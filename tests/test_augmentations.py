"""Tests for fm4tag.augmentations."""

from __future__ import annotations

import pytest
import torch

from fm4tag.augmentations import (
    Compose,
    CutMix,
    FeatureDropout,
    GaussianNoise,
    Identity,
    Mixup,
    Stage,
    TrackDropout,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, C, F_cat, F_con, D = 4, 10, 3, 5, 16


@pytest.fixture()
def pre_flatten_data():
    return {
        'categorical': torch.randint(0, 5, (B, C, F_cat)),
        'continuous': torch.randn(B, C, F_con),
        'valid': torch.ones(B, C, dtype=torch.bool),
    }


@pytest.fixture()
def raw_data():
    return {
        'categorical': torch.randint(0, 5, (B * C, F_cat)),
        'continuous': torch.randn(B * C, F_con),
    }


@pytest.fixture()
def embedding_data():
    return {
        'categorical': torch.randn(B * C, F_cat, D),
        'continuous': torch.randn(B * C, F_con, D),
    }


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_pre_flatten(pre_flatten_data):
    out = Identity()(pre_flatten_data)
    assert torch.equal(out['categorical'], pre_flatten_data['categorical'])
    assert torch.equal(out['continuous'], pre_flatten_data['continuous'])


def test_identity_stage_default():
    assert Identity.stage == Stage.RAW


def test_identity_stage_override():
    assert Identity(stage=Stage.PRE_FLATTEN).stage == Stage.PRE_FLATTEN


# ---------------------------------------------------------------------------
# TrackDropout
# ---------------------------------------------------------------------------


def test_track_dropout_reduces_valid(pre_flatten_data):
    aug = TrackDropout(drop_prob=0.9, min_valid=1)
    out = aug(pre_flatten_data)
    assert out['valid'].shape == pre_flatten_data['valid'].shape
    assert out['valid'].sum() <= pre_flatten_data['valid'].sum()


def test_track_dropout_min_valid(pre_flatten_data):
    aug = TrackDropout(drop_prob=1.0, min_valid=2)
    out = aug(pre_flatten_data)
    assert (out['valid'].sum(dim=1) >= 2).all()


def test_track_dropout_does_not_change_features(pre_flatten_data):
    aug = TrackDropout(drop_prob=0.5)
    out = aug(pre_flatten_data)
    assert torch.equal(out['categorical'], pre_flatten_data['categorical'])
    assert torch.equal(out['continuous'], pre_flatten_data['continuous'])


def test_track_dropout_no_valid_key():
    # Should return input unchanged when 'valid' is absent.
    data = {'categorical': torch.randint(0, 3, (B, C, F_cat))}
    out = TrackDropout()(data)
    assert torch.equal(out['categorical'], data['categorical'])


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------


def test_cutmix_output_shape(raw_data):
    aug = CutMix(lam=0.5)
    out = aug(raw_data)
    assert out['categorical'].shape == raw_data['categorical'].shape
    assert out['continuous'].shape == raw_data['continuous'].shape


def test_cutmix_stage():
    assert CutMix.stage == Stage.RAW


def test_cutmix_values_are_mix(raw_data):
    torch.manual_seed(0)
    aug = CutMix(lam=0.0)  # all values replaced → output ≠ input
    out = aug(raw_data)
    # lam=0 means keep-mask is all False, so all values come from a permuted copy.
    # They can coincidentally equal input, but on average they should differ.
    assert not torch.equal(out['continuous'], raw_data['continuous'])


# ---------------------------------------------------------------------------
# FeatureDropout
# ---------------------------------------------------------------------------


def test_feature_dropout_shape(raw_data):
    aug = FeatureDropout(corrupt_frac=0.5)
    out = aug(raw_data)
    assert out['continuous'].shape == raw_data['continuous'].shape


def test_feature_dropout_stage():
    assert FeatureDropout.stage == Stage.RAW


# ---------------------------------------------------------------------------
# GaussianNoise
# ---------------------------------------------------------------------------


def test_gaussian_noise_embedding_shape(embedding_data):
    aug = GaussianNoise(sigma=0.1, space='embedding')
    out = aug(embedding_data)
    assert out['categorical'].shape == embedding_data['categorical'].shape
    assert out['continuous'].shape == embedding_data['continuous'].shape


def test_gaussian_noise_stage_embedding():
    assert GaussianNoise(sigma=0.1, space='embedding').stage == Stage.EMBEDDING


def test_gaussian_noise_raw_shape(raw_data):
    aug = GaussianNoise(sigma=0.1, space='raw')
    out = aug(raw_data)
    assert out['continuous'].shape == raw_data['continuous'].shape


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------


def test_mixup_shape(embedding_data):
    aug = Mixup(lam=0.5)
    out = aug(embedding_data)
    assert out['categorical'].shape == embedding_data['categorical'].shape
    assert out['continuous'].shape == embedding_data['continuous'].shape


def test_mixup_stage():
    assert Mixup.stage == Stage.EMBEDDING


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def test_compose_stages_are_separated():
    compose = Compose(
        [
            TrackDropout(drop_prob=0.1),
            CutMix(lam=0.7),
            GaussianNoise(sigma=0.05, space='embedding'),
        ]
    )
    assert len(compose.pre_flatten) == 1
    assert len(compose.raw) == 1
    assert len(compose.embedding) == 1


def test_compose_empty_is_identity(pre_flatten_data, raw_data, embedding_data):
    compose = Compose([])
    out = compose.apply_pre_flatten(pre_flatten_data)
    assert torch.equal(out['categorical'], pre_flatten_data['categorical'])
    out = compose.apply_raw(raw_data)
    assert torch.equal(out['continuous'], raw_data['continuous'])
    out = compose.apply_embedding(embedding_data)
    assert torch.equal(out['continuous'], embedding_data['continuous'])


def test_compose_applies_in_order(raw_data):
    # Two CutMix at raw stage: both should be applied sequentially.
    compose = Compose([CutMix(lam=0.5), CutMix(lam=0.5)])
    assert len(compose.raw) == 2
    out = compose.apply_raw(raw_data)
    assert out['continuous'].shape == raw_data['continuous'].shape


def test_compose_pre_flatten_passes_through_raw(pre_flatten_data, raw_data):
    compose = Compose([TrackDropout(drop_prob=0.0)])
    # apply_raw should not be affected by a pre_flatten-only compose.
    out = compose.apply_raw(raw_data)
    assert torch.equal(out['continuous'], raw_data['continuous'])


def test_compose_repr_contains_stage_names():
    compose = Compose([TrackDropout(), CutMix(lam=0.5)])
    r = repr(compose)
    assert 'pre_flatten' in r
    assert 'raw' in r
