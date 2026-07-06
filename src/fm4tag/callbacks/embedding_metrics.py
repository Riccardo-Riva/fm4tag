"""Lightning callback logging representation-quality metrics on clean embeddings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import lightning as L
import torch

from ..metrics import compute_metric
from ..utils.ddp import gather_embeddings_sized


class EmbeddingMetrics(L.Callback):
    """Log representation-quality metrics (uniformity, effective rank, …).

    During each tracked split's epoch, clean (no augmentation) embeddings are
    accumulated by calling the module's ``forward(batch)``, which must return
    ``{'<name>': (N, D) tensor, ...}`` — e.g. :class:`PretrainModule` returns
    one entry per object (``jets_embedding``, ``tracks_embedding``, …) plus
    the ``aggregator`` output.  At epoch end the embeddings are gathered
    across DDP ranks and every metric is logged as ``<split>/<name>/<metric>``.

    Phase-agnostic: works with any module exposing that ``forward`` contract
    (pretraining and fine-tuning alike).  The extra forward pass runs in eval
    mode under ``no_grad`` and stops as soon as every quota is filled.

    Args:
        metrics:    Metric names registered in :mod:`fm4tag.metrics`.
        embeddings: Embedding names to track; ``None`` tracks everything the
                    module's ``forward`` returns.
        splits:     Subset of ``('train', 'val')``.
        n_samples:  Global embedding cap per name, summed across DDP ranks
                    (each rank keeps ``n_samples // world_size``), so metric
                    values do not drift with the number of GPUs.
    """

    def __init__(
        self,
        metrics: Sequence[str] = ('uniformity', 'effective_rank'),
        embeddings: Sequence[str] | None = None,
        splits: Sequence[str] = ('val',),
        n_samples: int = 8192,
    ) -> None:
        unknown_splits = set(splits) - {'train', 'val'}
        if unknown_splits:
            raise ValueError(
                f"splits must be a subset of ('train', 'val'), got {sorted(unknown_splits)}."
            )
        if n_samples < 1:
            raise ValueError(f'n_samples must be positive, got {n_samples}.')

        self.metrics = list(metrics)
        self.embeddings = list(embeddings) if embeddings is not None else None
        self.splits = list(splits)
        self.n_samples = n_samples

        # split → name → list of (n, D) CPU chunks.
        self._store: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # split → tracked names, fixed at the first batch so every DDP rank
        # iterates the gather collectives in the same order.
        self._names: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def _quota(self, trainer: L.Trainer) -> int:
        return max(1, self.n_samples // max(1, trainer.world_size))

    @staticmethod
    def _count(chunks: list[torch.Tensor]) -> int:
        return sum(t.size(0) for t in chunks)

    @torch.no_grad()
    def _accumulate(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: dict, split: str
    ) -> None:
        quota = self._quota(trainer)
        store = self._store[split]

        names = self._names.get(split)
        if names is not None and all(self._count(store[n]) >= quota for n in names):
            return  # every quota filled — skip the extra forward entirely

        # Clean embeddings: eval mode (dropout off, BN running stats), no grad.
        was_training = pl_module.training
        pl_module.eval()
        try:
            out = pl_module(batch)
        finally:
            pl_module.train(was_training)

        if names is None:
            if self.embeddings is not None:
                missing = set(self.embeddings) - set(out)
                if missing:
                    raise ValueError(
                        f'Embeddings {sorted(missing)} not returned by '
                        f'{type(pl_module).__name__}.forward; available: {sorted(out)}.'
                    )
                names = list(self.embeddings)
            else:
                names = sorted(out)
            self._names[split] = names

        for name in names:
            have = self._count(store[name])
            z = out[name]
            if have >= quota or z.size(0) == 0:
                continue
            store[name].append(z[: quota - have].detach().float().cpu())

    # ------------------------------------------------------------------
    # Epoch-end gather + logging
    # ------------------------------------------------------------------

    def _log_metrics(
        self, trainer: L.Trainer, pl_module: L.LightningModule, split: str
    ) -> None:
        store = self._store[split]

        # Same name order on all ranks keeps the gather collectives aligned.
        for name in self._names.get(split, []):
            chunks = store.get(name, [])
            z_local = torch.cat(chunks, dim=0) if chunks else None
            z = gather_embeddings_sized(
                z_local, trainer.world_size, pl_module.all_gather, pl_module.device
            )
            if z is None:
                continue
            for metric in self.metrics:
                value = compute_metric(metric, z)
                pl_module.log(
                    f'{split}/{name}/{metric}',
                    value if isinstance(value, torch.Tensor) else torch.tensor(value),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,  # identical on all ranks after the gather
                )

        store.clear()
        self._names.pop(split, None)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if 'train' in self.splits:
            self._accumulate(trainer, pl_module, batch, 'train')

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if 'val' in self.splits and not trainer.sanity_checking:
            self._accumulate(trainer, pl_module, batch, 'val')

    def on_train_epoch_end(self, trainer, pl_module):
        if 'train' in self.splits:
            self._log_metrics(trainer, pl_module, 'train')

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if 'val' in self.splits:
            self._log_metrics(trainer, pl_module, 'val')
