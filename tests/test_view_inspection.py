"""Tests for fm4tag.augmentations.collect_view_outputs (predict_step successor)."""

from __future__ import annotations

import pytest
import torch

from fm4tag.augmentations import (
    Compose,
    CutMix,
    FeatureDropout,
    TrackDropout,
    collect_view_outputs,
)

_F_CAT = 3
_F_CON = 4


@pytest.fixture()
def two_views():
    return [
        Compose([CutMix(lam=0.7)]),
        Compose([FeatureDropout(corrupt_frac=0.3)]),
    ]


def _make_batch(B: int = 4, C: int = 8) -> dict:
    return {
        'global': torch.randn(B, 2),
        'constituents': {
            'tracks': {
                'categorical': torch.randint(0, 2, (B, C, _F_CAT)),
                'continuous': torch.randn(B, C, _F_CON),
                'valid': torch.ones(B, C, dtype=torch.bool),
            }
        },
    }


def _collect(views, batch):
    return collect_view_outputs(
        views, batch, global_object='jets', constituent_objects=['tracks']
    )


def test_structure(two_views):
    out = _collect(two_views, _make_batch())

    assert 'jets' in out
    assert 'original' in out['jets']
    assert len(out['jets']['views']) == 2

    tracks = out['constituents']['tracks']
    assert 'original' in tracks
    assert len(tracks['views']) == 2


def test_view_has_pre_flatten_and_raw(two_views):
    out = _collect(two_views, _make_batch())
    view = out['constituents']['tracks']['views'][0]
    assert 'pre_flatten' in view
    assert 'raw' in view
    assert 'categorical' in view['raw']
    assert 'continuous' in view['raw']


def test_output_is_cpu(two_views):
    out = _collect(two_views, _make_batch())
    assert out['jets']['original'].device.type == 'cpu'
    view_raw = out['constituents']['tracks']['views'][0]['raw']
    assert view_raw['continuous'].device.type == 'cpu'


def test_input_batch_not_mutated(two_views):
    batch = _make_batch()
    snapshot = {
        'global': batch['global'].clone(),
        'cat': batch['constituents']['tracks']['categorical'].clone(),
        'con': batch['constituents']['tracks']['continuous'].clone(),
        'valid': batch['constituents']['tracks']['valid'].clone(),
    }
    _collect(two_views, batch)
    assert torch.equal(batch['global'], snapshot['global'])
    assert torch.equal(batch['constituents']['tracks']['categorical'], snapshot['cat'])
    assert torch.equal(batch['constituents']['tracks']['continuous'], snapshot['con'])
    assert torch.equal(batch['constituents']['tracks']['valid'], snapshot['valid'])


def test_mask_modifying_view_changes_constituent_count():
    """Each view uses its own valid mask, so TrackDropout is visible here."""
    torch.manual_seed(0)
    views = [Compose([]), Compose([TrackDropout(drop_prob=0.5)])]
    out = _collect(views, _make_batch(B=8, C=16))
    tracks = out['constituents']['tracks']

    n_clean = tracks['views'][0]['pre_flatten']['continuous'].shape[0]
    n_dropped = tracks['views'][1]['pre_flatten']['continuous'].shape[0]
    assert n_clean == 8 * 16
    assert n_dropped < n_clean
    # The valid mask reported for the dropped view must match its row count.
    assert int(tracks['views'][1]['pre_flatten']['valid'].sum()) == n_dropped
