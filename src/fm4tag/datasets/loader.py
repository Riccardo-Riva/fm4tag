"""DataLoader construction for the span-reading :class:`DatasetCatCon`.

One builder shared by the datamodule, ``eval_encoder`` and
``scripts/roc_comparison`` so the three cannot drift apart in how they drive the
dataset.  The important part is the trio ``batch_size=None`` /
``sampler=BatchSliceSampler`` / ``collate_fn=None``: automatic batching is off,
each sampler item is a list of row spans, and the dataset returns the collated
batch itself.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from .datasets import DatasetCatCon
from .samplers import BatchSliceSampler


def make_batch_dataloader(
    dataset: DatasetCatCon,
    *,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    reads_per_batch: int = 8,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    seed: int = 0,
    max_rows: int | None = None,
) -> DataLoader:
    """Build a DataLoader that reads whole batches as contiguous spans.

    Args:
        dataset:         The dataset to read from.
        batch_size:      Rows per batch, per rank.
        shuffle:         Randomise offsets and batch order (training only).
        drop_last:       Drop a trailing partial batch.
        reads_per_batch: Contiguous runs stitched into one batch; see
                         :class:`~fm4tag.datasets.samplers.BatchSliceSampler`.
        max_rows:        Use only the first ``max_rows`` rows of the file.
                         Replaces wrapping the dataset in ``torch.utils.data.Subset``,
                         which cannot work here — ``Subset`` remaps *integer*
                         indices, and this dataset is indexed by row spans.
        seed:            Base shuffling seed; the epoch is mixed in by Lightning
                         via ``sampler.set_epoch``.
    """
    n_rows = len(dataset) if max_rows is None else min(max_rows, len(dataset))

    sampler = BatchSliceSampler(
        n_rows,
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        reads_per_batch=reads_per_batch,
        seed=seed,
    )

    return DataLoader(
        dataset,
        # Automatic batching off: the sampler yields one batch's worth of spans
        # and the dataset returns the assembled batch.
        batch_size=None,
        sampler=sampler,
        collate_fn=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=pin_memory,
        # Start workers from a clean process instead of forking the parent.
        # With the default 'fork', workers inherit the parent's initialized
        # CUDA context; when an inherited CUDA tensor's destructor later runs
        # in the worker it aborts with cudaErrorInitializationError (CUDA
        # cannot be re-initialized in a fork child), killing the worker. The
        # finetune path hits this reliably because of its GPU-resident
        # torchmetrics/head state. 'forkserver' avoids inheriting CUDA.
        multiprocessing_context='forkserver' if num_workers > 0 else None,
    )


__all__ = ['make_batch_dataloader']
