"""Tests for :class:`~fm4tag.datasets.samplers.BatchSliceSampler`.

The sampler exists to keep HDF5 reads contiguous — the training file is chunked
and lzf compressed, so a single-row read decompresses the whole enclosing chunk
(~485 MB per 1024-jet batch, against ~3 MB for span reads).  What has to hold is
that yielding spans instead of indices does not quietly change *which* samples
the model sees: full coverage in sequential mode, correct batch sizes, a fresh
draw every epoch, and disjoint equal-length shards under DDP.
"""

from __future__ import annotations

import numpy as np
import pytest

from fm4tag.datasets import BatchSliceSampler


def _rows(batches) -> np.ndarray:
    """Flatten a list of span-lists into the row indices they select."""
    return np.concatenate([np.arange(s, e) for b in batches for (s, e) in b])


# ---------------------------------------------------------------------------
# Sequential mode (validation / test): must be an exact, ordered cover
# ---------------------------------------------------------------------------


def test_sequential_covers_every_row_exactly_once():
    sampler = BatchSliceSampler(10_000, 1000, shuffle=False)
    batches = list(sampler)

    assert np.array_equal(_rows(batches), np.arange(10_000))
    assert all(len(b) == 1 for b in batches), 'sequential batches must be one span'
    assert len(sampler) == len(batches)


def test_sequential_keeps_a_ragged_tail():
    sampler = BatchSliceSampler(10_500, 1000, shuffle=False, drop_last=False)
    assert np.array_equal(_rows(list(sampler)), np.arange(10_500))


def test_drop_last_discards_the_partial_batch():
    sampler = BatchSliceSampler(10_500, 1000, shuffle=False, drop_last=True)
    batches = list(sampler)
    assert len(batches) == 10
    assert np.array_equal(_rows(batches), np.arange(10_000))


def test_reads_per_batch_ignored_when_sequential():
    # Splitting a sequential read would fragment it for no benefit.
    sampler = BatchSliceSampler(10_000, 1000, shuffle=False, reads_per_batch=8)
    assert all(len(b) == 1 for b in sampler)


# ---------------------------------------------------------------------------
# Shuffled mode (training)
# ---------------------------------------------------------------------------


def test_shuffled_batches_have_the_right_shape():
    sampler = BatchSliceSampler(
        10_000, 1000, shuffle=True, reads_per_batch=8, seed=0
    )
    batches = list(sampler)

    assert len(sampler) == len(batches)
    for b in batches:
        assert len(b) == 8, 'one span per read'
        assert sum(e - s for s, e in b) == 1000, 'spans must total batch_size'
        assert all(0 <= s and e <= 10_000 for s, e in b), 'spans must stay in range'


def test_same_epoch_is_reproducible_and_different_epochs_differ():
    sampler = BatchSliceSampler(10_000, 1000, shuffle=True, seed=0)

    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == first, 'same epoch must replay identically'

    sampler.set_epoch(1)
    assert list(sampler) != first, 'a new epoch must redraw'


def test_batch_composition_is_redrawn_each_epoch():
    """The point of reads_per_batch > 1.

    Row (intersample) attention makes the batch a model *input*, so a batch
    composition frozen across epochs is something the model can memorise.
    """
    sampler = BatchSliceSampler(
        1_000_000, 1024, shuffle=True, reads_per_batch=8, seed=0
    )

    def first_batch(epoch):
        sampler.set_epoch(epoch)
        return set(_rows([next(iter(sampler))]))

    overlap = len(first_batch(0) & first_batch(1)) / 1024
    assert overlap < 0.05, f'batches repeat across epochs (overlap={overlap:.3f})'


def test_reads_per_batch_one_is_a_single_slice():
    sampler = BatchSliceSampler(10_000, 1000, shuffle=True, reads_per_batch=1)
    assert all(len(b) == 1 for b in sampler)


def test_spans_within_a_batch_are_ordered():
    # The reads of one batch walk the file forwards; only which regions land
    # together is randomised.
    sampler = BatchSliceSampler(
        1_000_000, 1024, shuffle=True, reads_per_batch=8, seed=3
    )
    for b in sampler:
        starts = [s for s, _ in b]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'kwargs, match',
    [
        (dict(reads_per_batch=0), 'reads_per_batch'),
        (dict(reads_per_batch=7), 'divisible'),  # 1000 % 7 != 0
    ],
)
def test_invalid_reads_per_batch_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BatchSliceSampler(10_000, 1000, shuffle=True, **kwargs)


def test_batch_larger_than_dataset_rejected():
    with pytest.raises(ValueError, match='exceeds dataset length'):
        BatchSliceSampler(500, 1000)


# ---------------------------------------------------------------------------
# DDP: real process group, since the sampler shards itself
# ---------------------------------------------------------------------------


def _worker_disjoint_shards(rank, world_size):
    import torch.distributed as dist

    sampler = BatchSliceSampler(
        100_000, 1000, shuffle=True, reads_per_batch=8, seed=0
    )
    batches = list(sampler)

    # Every rank must run the same number of steps or DDP deadlocks at the
    # first collective past the shortest rank.
    assert len(batches) == len(sampler), 'len() must match what __iter__ yields'
    counts = [None] * world_size
    dist.all_gather_object(counts, len(batches))
    assert len(set(counts)) == 1, f'ranks disagree on step count: {counts}'

    # And no two ranks may read the same batch.
    mine = {tuple(sorted(tuple(sp) for sp in b)) for b in batches}
    gathered = [None] * world_size
    dist.all_gather_object(gathered, mine)
    for other in range(world_size):
        if other != rank:
            assert not (mine & gathered[other]), (
                f'rank {rank} and {other} share batches'
            )


@pytest.mark.ddp
def test_ddp_shards_are_disjoint_and_equal_length():
    from tests.conftest import run_ddp_test

    run_ddp_test(_worker_disjoint_shards, world_size=2)
