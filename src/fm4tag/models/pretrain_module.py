"""Self-supervised pretraining Lightning module for the SAINT encoder.

The pretraining objective combines two complementary tasks:

* **Contrastive** (InfoNCE): two augmented views of the same constituent are
  pushed together in a projected embedding space, while views from different
  constituents are pushed apart.

* **Denoising**: a corrupted view is reconstructed by the encoder, supervised
  with cross-entropy (categorical) and MSE (continuous) losses.

Each constituent (track, hit, …) is treated as an independent tabular sample
during pretraining, so the jet-level hierarchy is intentionally ignored here.
"""

from __future__ import annotations

import lightning as L
import torch
from einops import rearrange
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .components.encoder import saint_encoder
from .components.losses import DenoisingLoss, InfoNCELoss
from ..data.augmentations import add_noise, embed_data, mixup_data


class PretrainModule(L.LightningModule):
    """Lightning module that self-supervisedly pretrains a :class:`saint_encoder`.

    The module is deliberately thin: it delegates all model logic to the
    encoder and loss classes, and only orchestrates the data flow.

    Args:
        encoder: An uninitialised or randomly initialised :class:`saint_encoder`.
        cfg:     Full Hydra config.  Relevant sub-keys:
                 ``cfg.pretrain``, ``cfg.optimizer``, ``cfg.constituent_objects``.
    """

    def __init__(self, encoder: saint_encoder, cfg: DictConfig) -> None:
        super().__init__()
        # Do not save encoder as hyperparameter (non-serialisable nn.Module).
        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.cfg = cfg

        self.contrastive_loss = InfoNCELoss(temperature=cfg.pretrain.nce_temp)
        self.denoising_loss = DenoisingLoss()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_valid_constituents(
        self, batch: dict, obj_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten (B, C, F) tensors and keep only valid entries.

        Returns:
            ``x_categ`` – ``(N_valid, F_cat)`` long
            ``x_cont``  – ``(N_valid, F_con)`` float
        """
        const = batch['constituents'][obj_name]
        x_categ = const['categorical']  # (B, C, F_cat)
        x_cont = const['continuous']  # (B, C, F_con)
        valids = const['valid']  # (B, C)

        valids_flat = rearrange(valids, 'b c -> (b c)')
        x_categ_flat = rearrange(x_categ, 'b c f -> (b c) f')
        x_cont_flat = rearrange(x_cont, 'b c f -> (b c) f')

        return x_categ_flat[valids_flat], x_cont_flat[valids_flat]

    def _compute_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg_pt = self.cfg.pretrain
        obj_name = list(self.cfg.constituent_objects)[0]  # primary constituent type

        # ── Raw (uncorrupted) constituents ──────────────────────────────────
        x_categ, x_cont = self._get_valid_constituents(batch, obj_name)

        # ── Build corrupted view 2 (before embedding) ───────────────────────
        if 'cutmix' in cfg_pt.aug:
            x_categ_2, x_cont_2 = add_noise(x_categ, x_cont, lam=cfg_pt.aug_lambda)
        else:
            x_categ_2, x_cont_2 = x_categ, x_cont

        # ── Embed both views ─────────────────────────────────────────────────
        x_cat_enc_1, x_con_enc_1 = embed_data(x_categ, x_cont, self.encoder)
        x_cat_enc_2, x_con_enc_2 = embed_data(x_categ_2, x_cont_2, self.encoder)

        # ── Mixup in embedding space (applied to view 2) ─────────────────────
        if 'mixup' in cfg_pt.aug:
            x_cat_enc_2, x_con_enc_2 = mixup_data(
                x_cat_enc_2, x_con_enc_2, lam=cfg_pt.aug_lambda
            )

        # ── Encode both views ─────────────────────────────────────────────────
        # Encoder forward receives pre-embedded tensors.
        # Output: (N_valid, F_cat+F_con, dim)
        X_1 = self.encoder(x_cat_enc_1, x_con_enc_1)
        X_2 = self.encoder(x_cat_enc_2, x_con_enc_2)

        # ── Losses ────────────────────────────────────────────────────────────
        total_loss = X_1.new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        if 'contrastive' in cfg_pt.tasks:
            # Flatten feature tokens and project.
            aug_1 = X_1.flatten(1, 2)  # (N, F*dim)
            aug_2 = X_2.flatten(1, 2)

            proj1 = self.encoder.pt_mlp1
            proj2 = (
                self.encoder.pt_mlp2
                if cfg_pt.projhead_style == 'diff'
                else self.encoder.pt_mlp1
            )

            z1 = proj1(aug_1)  # (N, proj_dim)
            z2 = proj2(aug_2)

            l_cont = self.contrastive_loss(z1, z2)
            log_dict['loss_contrastive'] = l_cont
            total_loss = total_loss + cfg_pt.lam0 * l_cont

        if 'denoising' in cfg_pt.tasks:
            # Reconstruct original features from the corrupted view.
            cat_outs = self.encoder.mlp1(X_2[:, : self.encoder.num_categories, :])
            con_outs = self.encoder.mlp2(X_2[:, self.encoder.num_categories :, :])

            l_cat, l_con = self.denoising_loss(cat_outs, x_categ, con_outs, x_cont)
            log_dict['loss_denoising_cat'] = l_cat
            log_dict['loss_denoising_con'] = l_con
            total_loss = total_loss + cfg_pt.lam1 * l_cat + cfg_pt.lam2 * l_con

        log_dict['loss'] = total_loss
        return total_loss, log_dict

    # ------------------------------------------------------------------
    # LightningModule hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self._compute_loss(batch)
        self.log(
            'train_loss', log_dict['loss'], on_step=True, on_epoch=True, prog_bar=True
        )
        for k, v in log_dict.items():
            if k != 'loss':
                self.log(f'train_{k}', v, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self._compute_loss(batch)
        self.log(
            'val_loss', log_dict['loss'], on_step=False, on_epoch=True, prog_bar=True
        )
        for k, v in log_dict.items():
            if k != 'loss':
                self.log(f'val_{k}', v, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        opt_cfg = self.cfg.optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get('weight_decay', 1e-5),
        )

        # Cosine annealing with a short linear warm-up (10 % of total steps).
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
