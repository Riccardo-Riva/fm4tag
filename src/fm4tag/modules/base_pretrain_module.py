"""Abstract base class for all self-supervised pretraining Lightning modules."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict

import lightning as L
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..metrics import compute_metric
from ..utils.ddp import gather_embeddings_sized


class BasePretrainModule(L.LightningModule):
    """Abstract Lightning module for self-supervised pretraining.

    Subclasses must implement :meth:`_compute_loss` and
    :meth:`_project_for_eval`.  Everything else — training loop, logging,
    epoch-end summaries, embedding metric accumulation, and the optimizer —
    is handled here.

    Args:
        encoders: :class:`~torch.nn.ModuleDict` mapping object name → encoder.
        cfg:      Full Hydra config.
    """

    def __init__(self, encoders: torch.nn.ModuleDict, cfg: DictConfig) -> None:
        super().__init__()
        # Ignore nn.Module constructor args (already saved via state_dict) so
        # they are not duplicated into the checkpoint's hyper_parameters.  Names
        # absent on a given subclass are harmless to list here.
        self.save_hyperparameters(
            ignore=['encoders', 'aggregator', 'views', 'loss']
        )

        self.encoders = encoders
        self.cfg = cfg
        self.global_object = cfg.global_object
        self.constituent_objects = list(cfg.constituent_objects)

        # Per-epoch metric buffers (key → list of scalar tensors).
        self._train_acc: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._val_acc: dict[str, list[torch.Tensor]] = defaultdict(list)

        # Per-epoch embedding buffers for uniformity / effective-rank.
        self._train_emb_acc: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._val_emb_acc: dict[str, list[torch.Tensor]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total pretraining loss and a per-object breakdown.

        Returns:
            ``(total_loss, log_dict)`` where ``log_dict`` contains scalar
            tensors keyed as ``'<obj_name>/<metric>'`` plus a top-level
            ``'loss'`` entry for the batch total.
        """

    @abstractmethod
    def _project_for_eval(
        self, batch: dict, obj_name: str
    ) -> torch.Tensor | None:
        """Return ``(N, D)`` projected embeddings for *obj_name* without grad.

        Used to accumulate embeddings for online representation-quality metrics
        (uniformity, effective rank).  Return ``None`` to skip this object.
        """

    # ------------------------------------------------------------------
    # Embedding metric helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _accumulate_embeddings(
        self,
        batch: dict,
        store: dict[str, list[torch.Tensor]],
    ) -> None:
        eval_cfg = self.cfg.get('eval', {})
        n_max: int = eval_cfg.get('n_samples', 8192)
        objects_filter = eval_cfg.get('objects', None)

        for obj_name in list(self.encoders.keys()):
            if objects_filter is not None and obj_name not in objects_filter:
                continue
            already = sum(t.size(0) for t in store[obj_name])
            if already >= n_max:
                continue

            z = self._project_for_eval(batch, obj_name)
            if z is None:
                continue

            remaining = n_max - already
            store[obj_name].append(z[:remaining].detach().cpu())

    def _compute_and_log_embedding_metrics(
        self,
        store: dict[str, list[torch.Tensor]],
        split: str,
    ) -> None:
        """Gather embeddings (DDP-safe), compute registered metrics, and log."""
        eval_cfg = self.cfg.get('eval', {})
        metric_names: list[str] = list(
            eval_cfg.get('metrics', ['uniformity', 'effective_rank'])
        )

        # Iterate over ALL encoder objects so every DDP rank participates in
        # the same collectives in the same order.
        for obj_name in list(self.encoders.keys()):
            chunks = store.get(obj_name, [])
            z_local = torch.cat(chunks, dim=0) if chunks else None

            z = gather_embeddings_sized(
                z_local, self.trainer.world_size, self.all_gather, self.device
            )
            if z is None:
                continue

            for name in metric_names:
                try:
                    val = compute_metric(name, z)
                except Exception:
                    continue
                self.log(
                    f'{split}_{obj_name}/{name}',
                    val if isinstance(val, torch.Tensor) else torch.tensor(val),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
        store.clear()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_metrics(
        self, log_dict: dict[str, torch.Tensor], split: str
    ) -> None:
        on_step = split == 'train'
        self.log(
            f'{split}_loss',
            log_dict['loss'],
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for k, v in log_dict.items():
            if k == 'loss':
                continue
            self.log(
                f'{split}_{k}',
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def _accumulate(
        self,
        store: dict[str, list[torch.Tensor]],
        log_dict: dict[str, torch.Tensor],
    ) -> None:
        for k, v in log_dict.items():
            store[k].append(v.detach().cpu().float())

    def _format_epoch_table(self, split: str, avgs: dict[str, float]) -> str:
        objects = [self.global_object] + self.constituent_objects

        # Discover sub-metric columns from available keys (everything after '/').
        all_subkeys = sorted({
            k.split('/', 1)[1]
            for k in avgs
            if '/' in k and k.split('/', 1)[1] != 'loss'
        })

        obj_w = max(len('TOTAL'), *(len(o) for o in objects)) + 2
        num_w = 12

        def fmt(val: float | None) -> str:
            return f'{val:{num_w}.4f}' if val is not None else f'{"—":>{num_w}}'

        header = f'{"Object":<{obj_w}}{"Loss":>{num_w}}' + ''.join(
            f'{k:>{num_w}}' for k in all_subkeys
        )
        sep = '─' * len(header)
        title = f'Epoch {self.current_epoch} | {split}'

        lines = ['', sep, title, sep, header, sep]

        for obj in objects:
            row = (
                f'{obj:<{obj_w}}'
                + fmt(avgs.get(f'{obj}/loss'))
                + ''.join(fmt(avgs.get(f'{obj}/{k}')) for k in all_subkeys)
            )
            lines.append(row)

        lines += [
            sep,
            f'{"TOTAL":<{obj_w}}' + fmt(avgs.get('loss')),
            sep,
            '',
        ]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # LightningModule hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self._compute_loss(batch)
        self._log_metrics(log_dict, 'train')
        self._accumulate(self._train_acc, log_dict)
        eval_cfg = self.cfg.get('eval', {})
        if eval_cfg.get('enabled', False) and 'train' in eval_cfg.get(
            'splits', ['val']
        ):
            self._accumulate_embeddings(batch, self._train_emb_acc)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self._compute_loss(batch)
        self._log_metrics(log_dict, 'val')
        self._accumulate(self._val_acc, log_dict)
        eval_cfg = self.cfg.get('eval', {})
        if eval_cfg.get('enabled', False) and 'val' in eval_cfg.get('splits', ['val']):
            self._accumulate_embeddings(batch, self._val_emb_acc)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self._compute_loss(batch)
        self._log_metrics(log_dict, 'test')
        return loss

    def predict_step(self, batch: dict, batch_idx: int):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement predict_step."
        )

    def on_train_epoch_end(self) -> None:
        avgs = {k: torch.stack(v).mean().item() for k, v in self._train_acc.items()}
        self.print(self._format_epoch_table('Train', avgs))
        self._train_acc.clear()
        eval_cfg = self.cfg.get('eval', {})
        if eval_cfg.get('enabled', False) and 'train' in eval_cfg.get(
            'splits', ['val']
        ):
            self._compute_and_log_embedding_metrics(self._train_emb_acc, 'train')

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._val_acc.clear()
            self._val_emb_acc.clear()
            return
        avgs = {k: torch.stack(v).mean().item() for k, v in self._val_acc.items()}
        self.print(self._format_epoch_table('Val', avgs))
        self._val_acc.clear()
        eval_cfg = self.cfg.get('eval', {})
        if eval_cfg.get('enabled', False) and 'val' in eval_cfg.get('splits', ['val']):
            self._compute_and_log_embedding_metrics(self._val_emb_acc, 'val')

    def configure_optimizers(self):  # type: ignore[override]
        opt_cfg = self.cfg.optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get('weight_decay', 1e-5),
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(0.1 * total_steps))

        warmup_sched = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps)
        )
        scheduler = SequentialLR(
            optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }
