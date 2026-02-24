"""Supervised fine-tuning Lightning module.

Wraps a pretrained (or randomly initialised) :class:`Encoder` backbone
and a :class:`ClassifierHead` into a full Lightning module that supports:

* ``fit``     — supervised training (with optional encoder freezing /
                progressive unfreezing via the ``BackboneFinetuning`` callback)
* ``test``    — evaluation on a labelled test set
* ``predict`` — inference returning class probabilities (softmax)

The backbone is stored as ``self.backbone`` so that Lightning's built-in
:class:`~lightning.pytorch.callbacks.BackboneFinetuning` callback can locate
and unfreeze it automatically when ``freeze_encoder=True`` in the config.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .components.encoder import Encoder
from .components.heads import ClassifierHead
from ..data.augmentations import embed_data


class FinetuneModule(L.LightningModule):
    """Lightning module for supervised fine-tuning of the encoder + classifier.

    Args:
        backbone: :class:`Encoder` — the feature extractor.
                  Named ``backbone`` (not ``encoder``) so the
                  ``BackboneFinetuning`` callback finds it automatically.
        head:     :class:`ClassifierHead` — the classification head.
        cfg:      Full Hydra config.  Relevant sub-keys:
                  ``cfg.optimizer``, ``cfg.freeze_encoder``,
                  ``cfg.class_weights``, ``cfg.constituent_objects``.
    """

    def __init__(
        self,
        backbone: Encoder,
        head: ClassifierHead,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'head'])

        # Named 'backbone' for BackboneFinetuning callback compatibility.
        self.backbone = backbone
        self.head = head
        self.cfg = cfg

        class_weights = cfg.get('class_weights')
        if class_weights is not None:
            self.register_buffer(
                'class_weights', torch.tensor(class_weights, dtype=torch.float)
            )
        else:
            self.class_weights: torch.Tensor | None = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        """Embed → encode → classify one batch.

        Args:
            batch: Dict from the datamodule collate function.

        Returns:
            ``(B, y_dim)`` class logits.
        """
        obj_name = list(self.cfg.constituent_objects)[0]
        const = batch['constituents'][obj_name]

        x_categ = const['categorical']  # (B, C, F_cat)
        x_cont = const['continuous']  # (B, C, F_con)
        valids = const['valid']  # (B, C)

        B, C, _ = x_categ.shape

        valids_flat = rearrange(valids, 'b c -> (b c)')  # (B*C,)
        x_categ_flat = rearrange(x_categ, 'b c f -> (b c) f')  # (B*C, F_cat)
        x_cont_flat = rearrange(x_cont, 'b c f -> (b c) f')  # (B*C, F_con)

        # Embed and encode only valid constituents for efficiency.
        x_cat_enc, x_con_enc = embed_data(
            x_categ_flat[valids_flat], x_cont_flat[valids_flat], self.backbone
        )
        x_valid_encoded = self.backbone(x_cat_enc, x_con_enc)  # (N_valid, F, dim)

        # Scatter back into a full (B*C, F, dim) tensor (zeros for padding).
        F_feat, dim = x_valid_encoded.shape[1], x_valid_encoded.shape[2]
        x_encoded = x_valid_encoded.new_zeros(B * C, F_feat, dim)
        x_encoded[valids_flat] = x_valid_encoded
        x_encoded = x_encoded.reshape(B, C, F_feat, dim)  # (B, C, F, dim)

        return self.head(x_encoded, valids)  # (B, y_dim)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward + compute loss and accuracy.

        Returns:
            ``(loss, preds, labels)``
        """
        logits = self(batch)
        labels = batch['label']
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = logits.argmax(dim=-1)
        return loss, preds, labels

    # ------------------------------------------------------------------
    # LightningModule hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, labels = self._shared_step(batch)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        return self(batch).softmax(dim=-1)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):  # type: ignore[override]
        opt_cfg = self.cfg.optimizer
        freeze = self.cfg.get('freeze_encoder', False)

        if freeze:
            # Backbone starts frozen; the BackboneFinetuning callback will
            # add backbone parameters to the optimiser when unfreezing.
            params = list(self.head.parameters())
            optimizer = torch.optim.AdamW(
                params,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.get('weight_decay', 1e-5),
            )
        else:
            # Both backbone and head are optimised from epoch 0.
            # The backbone can have a lower learning rate to reduce the risk
            # of catastrophic forgetting of pretrained representations.
            backbone_lr = opt_cfg.get('backbone_lr', opt_cfg.lr)
            optimizer = torch.optim.AdamW(
                [
                    {'params': list(self.backbone.parameters()), 'lr': backbone_lr},
                    {'params': list(self.head.parameters()), 'lr': opt_cfg.lr},
                ],
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
