"""Self-supervised pretraining Lightning module.

The pretraining objective combines two complementary tasks applied **independently
to every object** in the dataset (global object + all constituent types):

* **Contrastive** (InfoNCE): two augmented views of the same sample are pushed
  together in a projected embedding space while views from different samples are
  pushed apart.

* **Denoising**: a corrupted view is reconstructed by the encoder, supervised
  with cross-entropy (categorical) + MSE (continuous) losses.
  For the global object (continuous-only) only the MSE term is used.

The total loss is the sum of per-object losses.  Each encoder's parameters
only receive gradients from its own loss, making the training equivalent to
independent per-encoder pretraining.

At the end of every epoch a formatted table is printed with per-object loss
breakdowns for both the train and (when available) validation split.  The same
values are also written to the CSV log by Lightning.
"""

from __future__ import annotations

from collections import defaultdict

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .components.encoder import Encoder, GlobalEncoder
from .components.losses import DenoisingLoss, InfoNCELoss
from ..data.augmentations import add_noise, embed_data, mixup_data


class PretrainModule(L.LightningModule):
    """Lightning module that self-supervisedly pretrains all object encoders.

    Args:
        encoders: :class:`~torch.nn.ModuleDict` mapping each object name to its
                  encoder — a :class:`GlobalEncoder` for the global object and an
                  :class:`Encoder` for each constituent type.
        cfg:      Full Hydra config.  Relevant sub-keys:
                  ``cfg.pretrain``, ``cfg.optimizer``,
                  ``cfg.global_object``, ``cfg.constituent_objects``.
    """

    def __init__(self, encoders: torch.nn.ModuleDict, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['encoders'])

        self.encoders = encoders
        self.cfg = cfg
        self.global_object = cfg.global_object
        self.constituent_objects = list(cfg.constituent_objects)

        self.contrastive_loss = InfoNCELoss(temperature=cfg.pretrain.nce_temp)
        self.denoising_loss = DenoisingLoss()

        # Per-epoch metric buffers for the epoch-end summary table.
        # Each entry is a list of scalar tensors accumulated during the epoch.
        self._train_acc: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._val_acc: dict[str, list[torch.Tensor]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Per-object loss helpers
    # ------------------------------------------------------------------

    def _compute_loss_for_global(
        self,
        batch: dict,
        encoder: GlobalEncoder,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Contrastive + MSE denoising for the global (flat continuous) object."""
        cfg_pt = self.cfg.pretrain
        x_global = batch['global']  # (B, F_g)

        # ── Corrupted view 2 ────────────────────────────────────────────────
        if 'cutmix' in cfg_pt.aug:
            _, x_global_2 = add_noise(None, x_global, lam=cfg_pt.aug_lambda)
        else:
            x_global_2 = x_global

        # ── Embed both views ─────────────────────────────────────────────────
        X_1 = encoder(x_global)  # (B, F_g, dim)
        X_2 = encoder(x_global_2)  # (B, F_g, dim)

        # ── Mixup in embedding space (applied to view 2) ─────────────────────
        if 'mixup' in cfg_pt.aug:
            idx = torch.randperm(X_2.size(0), device=X_2.device)
            X_2 = cfg_pt.aug_lambda * X_2 + (1.0 - cfg_pt.aug_lambda) * X_2[idx]

        # ── Losses ────────────────────────────────────────────────────────────
        total_loss = X_1.new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        if 'contrastive' in cfg_pt.tasks:
            proj1 = encoder.pt_mlp1
            proj2 = (
                encoder.pt_mlp2 if cfg_pt.projhead_style == 'diff' else encoder.pt_mlp1
            )
            z1 = proj1(X_1.flatten(1))  # (B, proj_dim)
            z2 = proj2(X_2.flatten(1))
            l_cont = self.contrastive_loss(z1, z2)
            log_dict['loss_contrastive'] = l_cont
            total_loss = total_loss + cfg_pt.lam0 * l_cont

        if 'denoising' in cfg_pt.tasks:
            con_pred = torch.cat(encoder.mlp_recon(X_2), dim=1)  # (B, F_g)
            l_con = F.mse_loss(con_pred, x_global)
            log_dict['loss_denoising_con'] = l_con
            total_loss = total_loss + cfg_pt.lam2 * l_con

        log_dict['loss'] = total_loss
        return total_loss, log_dict

    def _compute_loss_for_constituent(
        self,
        batch: dict,
        obj_name: str,
        encoder: Encoder,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Contrastive + categorical/continuous denoising for one constituent type."""
        cfg_pt = self.cfg.pretrain

        # ── Flatten valid constituents ───────────────────────────────────────
        const = batch['constituents'][obj_name]
        valids_flat = rearrange(const['valid'], 'b c -> (b c)')
        x_categ = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_flat]
        x_cont = rearrange(const['continuous'], 'b c f -> (b c) f')[valids_flat]

        # ── Corrupted view 2 ────────────────────────────────────────────────
        if 'cutmix' in cfg_pt.aug:
            x_categ_2, x_cont_2 = add_noise(x_categ, x_cont, lam=cfg_pt.aug_lambda)
        else:
            x_categ_2, x_cont_2 = x_categ, x_cont

        # ── Embed both views ─────────────────────────────────────────────────
        x_cat_enc_1, x_con_enc_1 = embed_data(x_categ, x_cont, encoder)
        x_cat_enc_2, x_con_enc_2 = embed_data(x_categ_2, x_cont_2, encoder)

        # ── Mixup in embedding space (applied to view 2) ─────────────────────
        if 'mixup' in cfg_pt.aug:
            x_cat_enc_2, x_con_enc_2 = mixup_data(
                x_cat_enc_2, x_con_enc_2, lam=cfg_pt.aug_lambda
            )

        # ── Encode both views ─────────────────────────────────────────────────
        X_1 = encoder(x_cat_enc_1, x_con_enc_1)  # (N_valid, F, dim)
        X_2 = encoder(x_cat_enc_2, x_con_enc_2)

        # ── Losses ────────────────────────────────────────────────────────────
        total_loss = X_1.new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        if 'contrastive' in cfg_pt.tasks:
            proj1 = encoder.pt_mlp1
            proj2 = (
                encoder.pt_mlp2 if cfg_pt.projhead_style == 'diff' else encoder.pt_mlp1
            )
            z1 = proj1(X_1.flatten(1, 2))
            z2 = proj2(X_2.flatten(1, 2))
            l_cont = self.contrastive_loss(z1, z2)
            log_dict['loss_contrastive'] = l_cont
            total_loss = total_loss + cfg_pt.lam0 * l_cont

        if 'denoising' in cfg_pt.tasks:
            cat_outs = encoder.mlp1(X_2[:, : encoder.num_categories, :])
            con_outs = encoder.mlp2(X_2[:, encoder.num_categories :, :])
            l_cat, l_con = self.denoising_loss(cat_outs, x_categ, con_outs, x_cont)
            log_dict['loss_denoising_cat'] = l_cat
            log_dict['loss_denoising_con'] = l_con
            total_loss = total_loss + cfg_pt.lam1 * l_cat + cfg_pt.lam2 * l_con

        log_dict['loss'] = total_loss
        return total_loss, log_dict

    def _compute_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sum per-object losses and collect per-object metric breakdown."""
        total_loss = batch['global'].new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        for obj_name, encoder in self.encoders.items():
            if obj_name == self.global_object:
                obj_loss, obj_logs = self._compute_loss_for_global(batch, encoder)
            else:
                obj_loss, obj_logs = self._compute_loss_for_constituent(
                    batch, obj_name, encoder
                )
            total_loss = total_loss + obj_loss
            for k, v in obj_logs.items():
                log_dict[f'{obj_name}/{k}'] = v

        log_dict['loss'] = total_loss
        return total_loss, log_dict

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_metrics(self, log_dict: dict[str, torch.Tensor], split: str) -> None:
        """Write all metrics to the Lightning logger (CSV file).

        The total loss is logged both on-step (during train) and on-epoch.
        Per-object breakdowns are epoch-only to keep the progress bar clean.
        """
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
        self, store: dict[str, list[torch.Tensor]], log_dict: dict[str, torch.Tensor]
    ) -> None:
        """Append detached scalar tensors to the per-epoch accumulator."""
        for k, v in log_dict.items():
            store[k].append(v.detach().cpu().float())

    def _format_epoch_table(self, split: str, avgs: dict[str, float]) -> str:
        """Return a formatted multi-column table string for the epoch summary."""
        objects = [self.global_object] + self.constituent_objects

        # Determine which sub-metric columns have data for at least one object.
        all_subkeys = ['loss_contrastive', 'loss_denoising_cat', 'loss_denoising_con']
        col_labels = {
            'loss_contrastive': 'Contrastive',
            'loss_denoising_cat': 'Denois.Cat',
            'loss_denoising_con': 'Denois.Con',
        }
        active_cols = [
            c for c in all_subkeys if any(f'{o}/{c}' in avgs for o in objects)
        ]

        obj_w = max(len('TOTAL'), *(len(o) for o in objects)) + 2
        num_w = 12

        def fmt(val: float | None) -> str:
            return f'{val:{num_w}.4f}' if val is not None else f'{"—":>{num_w}}'

        # Header row
        header = f'{"Object":<{obj_w}}{"Loss":>{num_w}}' + ''.join(
            f'{col_labels[c]:>{num_w}}' for c in active_cols
        )
        sep = '─' * len(header)
        title = f'Epoch {self.current_epoch} | {split}'

        lines = ['', sep, title, sep, header, sep]

        for obj in objects:
            row = (
                f'{obj:<{obj_w}}'
                + fmt(avgs.get(f'{obj}/loss'))
                + ''.join(fmt(avgs.get(f'{obj}/{c}')) for c in active_cols)
            )
            lines.append(row)

        total = avgs.get('loss')
        lines += [
            sep,
            f'{"TOTAL":<{obj_w}}' + fmt(total),
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
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self._compute_loss(batch)
        self._log_metrics(log_dict, 'val')
        self._accumulate(self._val_acc, log_dict)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self._compute_loss(batch)
        self._log_metrics(log_dict, 'test')
        return loss

    def on_train_epoch_end(self) -> None:
        avgs = {k: torch.stack(v).mean().item() for k, v in self._train_acc.items()}
        self.print(self._format_epoch_table('Train', avgs))
        self._train_acc.clear()

    def on_validation_epoch_end(self) -> None:
        # Skip the sanity-check validation that runs before training starts.
        if self.trainer.sanity_checking:
            self._val_acc.clear()
            return
        avgs = {k: torch.stack(v).mean().item() for k, v in self._val_acc.items()}
        self.print(self._format_epoch_table('Val', avgs))
        self._val_acc.clear()

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
