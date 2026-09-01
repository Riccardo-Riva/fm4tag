"""Batch samplers that keep HDF5 reads contiguous.

The training file is chunked and compressed (``jets`` 3174 rows/chunk, ``tracks``
100 rows/chunk, both lzf), so h5py cannot read a single row — it fetches and
decompresses the whole enclosing chunk.  Sampling one random index at a time
therefore costs ~240 kB of decompression per 76-byte row, and with a shuffled
52M-row dataset the chunk cache never hits: measured 485 MB decompressed to
build a 2.45 MB batch, 13.4 s/batch against 26.6 ms for a contiguous slice.

:class:`BatchSliceSampler` yields *spans* instead of indices, so the dataset can
read whole runs of rows at a time.
"""

from __future__ import annotations

import math

import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler


def _ddp_context() -> tuple[int, int]:
    """``(rank, world_size)``, falling back to single-process values."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class BatchSliceSampler(Sampler):
    """Yield each batch as a list of contiguous ``(start, stop)`` spans.

    A batch of ``batch_size`` rows is assembled from ``reads_per_batch``
    equal-length runs taken at independent random offsets, so one batch costs
    ``reads_per_batch`` HDF5 reads instead of ``batch_size`` of them.

    ``reads_per_batch`` trades I/O against how freely samples mix:

    * ``1`` — one contiguous slice per batch, the cheapest possible read (this
      is what salt's ``RandomBatchSampler`` does).  Only the *order* of batches
      is shuffled, so a given jet sits with the same neighbours in every epoch.
    * ``>1`` (default ``8``) — the batch is stitched from that many unrelated
      regions of the file, redrawn every epoch, so co-occurrence is randomised
      while the read count stays ~2 orders of magnitude below per-row sampling.

    That distinction matters here in a way it does not for salt: the row
    (intersample) attention makes the batch a model *input*, so a batch
    composition frozen across epochs is something the model can learn.

    Weak shuffling is only sound if the file is already shuffled on disk — this
    one is (class fractions and kinematics agree to ~1% across slices 40M rows
    apart), which is also the assumption salt's sampler makes.

    Note that with ``shuffle=True`` the run offsets are drawn *with replacement*,
    so an epoch is ``len(self) * batch_size`` samples drawn from the file rather
    than an exact permutation of it.  For a 52M-row file that distinction is
    statistical noise; ``shuffle=False`` (validation, test) is always an exact,
    ordered, non-overlapping cover.

    Under DDP each rank takes a disjoint stride of the batch list, so no two
    ranks read the same rows.  This replaces Lightning's automatic sampler
    injection, so the trainer must run with ``use_distributed_sampler: false``.

    Args:
        dataset_len:     Number of rows in the file.
        batch_size:      Rows per batch, per rank.
        shuffle:         Randomise offsets and batch order (training).
        drop_last:       Drop the trailing partial batch (``shuffle=False`` only;
                         shuffled epochs are always whole batches).
        reads_per_batch: Contiguous runs stitched into one batch.
        seed:            Base seed; the epoch is mixed in via :meth:`set_epoch`.
    """

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        reads_per_batch: int = 8,
        seed: int = 0,
    ) -> None:
        if reads_per_batch < 1:
            raise ValueError(f'reads_per_batch must be >= 1, got {reads_per_batch}')
        if shuffle and batch_size % reads_per_batch != 0:
            raise ValueError(
                f'batch_size ({batch_size}) must be divisible by '
                f'reads_per_batch ({reads_per_batch})'
            )
        if batch_size > dataset_len:
            raise ValueError(
                f'batch_size ({batch_size}) exceeds dataset length ({dataset_len})'
            )

        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        # Sequential reading is already one contiguous run per batch; splitting
        # it would fragment the read for no benefit.
        self.reads_per_batch = reads_per_batch if shuffle else 1
        self.seed = seed
        self.epoch = 0

        if shuffle:
            n_batches = dataset_len // batch_size
        elif drop_last:
            n_batches = dataset_len // batch_size
        else:
            n_batches = math.ceil(dataset_len / batch_size)
        self._n_batches = n_batches

    def __len__(self) -> int:
        # Every rank must run the same number of steps or DDP deadlocks at the
        # first collective past the shortest rank, so round DOWN to a common count.
        _, world_size = _ddp_context()
        return self._n_batches // world_size

    def set_epoch(self, epoch: int) -> None:
        """Reseed the shuffle. Lightning calls this at each epoch boundary."""
        self.epoch = epoch

    def _batch_spans(self, rng: np.random.Generator) -> list[list[tuple[int, int]]]:
        """Build this epoch's batches as lists of ``(start, stop)`` spans."""
        if not self.shuffle:
            spans = []
            for b in range(self._n_batches):
                start = b * self.batch_size
                stop = min(start + self.batch_size, self.dataset_len)
                spans.append([(start, stop)])
            return spans

        run = self.batch_size // self.reads_per_batch
        # One independent uniform offset per run. Drawing them all at once and
        # reshaping keeps each batch's runs independent of each other, so a batch
        # is stitched from regions spread across the whole file.
        offsets = rng.integers(
            0,
            self.dataset_len - run + 1,
            size=(self._n_batches, self.reads_per_batch),
        )
        # Sort within a batch only: the reads of one batch then walk the file
        # forwards, without correlating which regions land in the same batch.
        offsets.sort(axis=1)
        return [
            [(int(o), int(o) + run) for o in batch_offsets] for batch_offsets in offsets
        ]

    def __iter__(self):
        # Read the DDP context here, not in __init__: the sampler can be built
        # before the process group exists.
        rank, world_size = _ddp_context()

        rng = np.random.default_rng(self.seed + self.epoch)
        spans = self._batch_spans(rng)

        if self.shuffle:
            spans = [spans[i] for i in rng.permutation(len(spans))]

        n = (len(spans) // world_size) * world_size
        for i in range(rank, n, world_size):
            yield spans[i]


__all__ = ['BatchSliceSampler']
