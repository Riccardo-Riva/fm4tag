"""Supervised fine-tuning Lightning module.

Wraps a :class:`~torch.nn.ModuleDict` of pretrained (or randomly initialised)
encoders and a :class:`~fm4tag.models.components.heads.MultiStreamClassifierHead`
into a full Lightning module that supports:

* ``fit``     — supervised training (with optional encoder freezing /
                progressive unfreezing via the ``BackboneFinetuning`` callback)
* ``test``    — evaluation on a labelled test set
* ``predict`` — inference returning class probabilities (softmax)

The encoders are stored as ``self.backbone`` (an :class:`~torch.nn.ModuleDict`)
so that Lightning's built-in :class:`~lightning.pytorch.callbacks.BackboneFinetuning`
callback can locate and unfreeze them automatically when ``freeze_encoder=True``
in the config.
"""

from __future__ import annotations

from collections import defaultdict

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.classification import MulticlassAUROC

from .components.eval_metrics import effective_rank, uniformity
from .components.heads import MultiStreamClassifierHead
from ..data.augmentations import embed_data


class FinetuneModule(L.LightningModule):
    """Lightning module for supervised fine-tuning of all encoders + classifier.

    Args:
        encoders: :class:`~torch.nn.ModuleDict` mapping each object name to its
                  encoder — a :class:`GlobalEncoder` for the global object and
                  an :class:`Encoder` for each constituent type.  Stored as
                  ``self.backbone`` for ``BackboneFinetuning`` compatibility.
        head:     :class:`MultiStreamClassifierHead` — the classification head.
        cfg:      Full Hydra config.  Relevant sub-keys:
                  ``cfg.optimizer``, ``cfg.freeze_encoder``,
                  ``cfg.class_dict``, ``cfg.global_object``,
                  ``cfg.constituent_objects``.
    """

    def __init__(
        self,
        encoders: torch.nn.ModuleDict,
        head: MultiStreamClassifierHead,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['encoders', 'head'])

        # Named 'backbone' for BackboneFinetuning callback compatibility.
        self.backbone = encoders
        self.head = head
        self.cfg = cfg

        class_weights = None
        if cfg.get('class_dict'):
            from omegaconf import OmegaConf
            cd = OmegaConf.to_container(OmegaConf.load(cfg.class_dict), resolve=True)
            global_obj = cfg.global_object
            label_var = cfg.variables[global_obj].label
            class_weights = cd.get(global_obj, {}).get(label_var)

        if class_weights is not None:
            self.register_buffer(
                'class_weights', torch.tensor(class_weights, dtype=torch.float)
            )
        else:
            self.class_weights: torch.Tensor | None = None  # type: ignore[assignment]

        n_classes = len(cfg.variables[cfg.global_object].unique_labels)
        self.val_auroc = MulticlassAUROC(num_classes=n_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=n_classes, average='macro')

        # Per-epoch embedding buffers for online uniformity / effective-rank.
        # Keys: object name → list of (N, D) CPU tensors.
        self._val_emb_acc: dict[str, list[torch.Tensor]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Encoding pipeline
    # ------------------------------------------------------------------

    def _encode_all(
        self,
        batch: dict,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Run all encoders and apply pt_mlp1 projections.

        Returns:
            z_global:      ``(B, g_dim)`` — global projected embedding.
            z_consts:      List of ``(B, C, c_dim_i)`` — constituent projected
                           embeddings scattered back over the padded grid.
            valids_list:   List of ``(B, C)`` bool masks.
            z_valid_flat:  List of ``(N_valid_i, c_dim_i)`` — flat valid
                           constituent embeddings (before scattering), used for
                           uniformity / effective-rank monitoring.
        """
        # ── Global ───────────────────────────────────────────────────────────
        global_name = self.cfg.global_object
        enc_global = self.backbone[global_name]
        X_global = enc_global(batch['global'])                  # (B, F_g, dim)
        z_global = enc_global.pt_mlp1(X_global.flatten(1))     # (B, g_dim)

        # ── Constituents ─────────────────────────────────────────────────────
        z_consts: list[torch.Tensor] = []
        valids_list: list[torch.Tensor] = []
        z_valid_flat: list[torch.Tensor] = []

        for obj_name in self.cfg.constituent_objects:
            encoder = self.backbone[obj_name]
            const = batch['constituents'][obj_name]

            x_categ = const['categorical']   # (B, C, F_cat)
            x_cont = const['continuous']     # (B, C, F_con)
            valids = const['valid']          # (B, C)

            B, C, _ = x_categ.shape
            valids_flat = rearrange(valids, 'b c -> (b c)')
            x_categ_flat = rearrange(x_categ, 'b c f -> (b c) f')
            x_cont_flat = rearrange(x_cont, 'b c f -> (b c) f')

            # Encode only valid constituents for efficiency.
            x_cat_enc, x_con_enc = embed_data(
                x_categ_flat[valids_flat], x_cont_flat[valids_flat], encoder
            )
            X_valid = encoder(x_cat_enc, x_con_enc)             # (N_valid, F, dim)
            z_valid = encoder.pt_mlp1(X_valid.flatten(1, 2))   # (N_valid, c_dim)

            # Scatter projected embeddings back into (B, C, c_dim).
            c_dim = z_valid.shape[1]
            z_all = z_valid.new_zeros(B * C, c_dim)
            z_all[valids_flat] = z_valid
            z_all = z_all.reshape(B, C, c_dim)

            z_consts.append(z_all)
            valids_list.append(valids)
            z_valid_flat.append(z_valid)

        return z_global, z_consts, valids_list, z_valid_flat

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        """Encode all objects and classify.

        Args:
            batch: Dict from the datamodule collate function.

        Returns:
            ``(B, y_dim)`` class logits.
        """
        z_global, z_consts, valids_list, _ = self._encode_all(batch)
        return self.head(z_global, z_consts, valids_list)

    # ------------------------------------------------------------------
    # Online embedding metric helpers
    # ------------------------------------------------------------------

    def _accumulate_embeddings(
        self,
        z_global: torch.Tensor,
        z_valid_flat: list[torch.Tensor],
    ) -> None:
        """Append pre-computed pt_mlp1 projections to the embedding store.

        Uses the embeddings already computed during the forward pass —
        no extra encoder run needed.
        """
        eval_cfg = self.cfg.get('eval', {})
        n_max: int = eval_cfg.get('n_samples', 8192)
        objects_filter = eval_cfg.get('objects', None)

        global_name = self.cfg.global_object
        if objects_filter is None or global_name in objects_filter:
            already = sum(t.size(0) for t in self._val_emb_acc[global_name])
            remaining = n_max - already
            if remaining > 0:
                self._val_emb_acc[global_name].append(
                    z_global[:remaining].detach().cpu()
                )

        for obj_name, z_valid in zip(self.cfg.constituent_objects, z_valid_flat):
            if objects_filter is not None and obj_name not in objects_filter:
                continue
            already = sum(t.size(0) for t in self._val_emb_acc[obj_name])
            remaining = n_max - already
            if remaining > 0:
                self._val_emb_acc[obj_name].append(
                    z_valid[:remaining].detach().cpu()
                )

    def _compute_and_log_embedding_metrics(self) -> None:
        """Concatenate accumulated embeddings, gather (DDP), compute and log metrics."""
        eval_cfg = self.cfg.get('eval', {})
        for obj_name, chunks in self._val_emb_acc.items():
            if not chunks:
                continue
            z = torch.cat(chunks, dim=0)    # (N_local, D) on CPU

            if self.trainer.world_size > 1:
                z = self.all_gather(z.to(self.device)).flatten(0, 1).cpu()

            if eval_cfg.get('log_uniformity', True):
                self.log(
                    f'val_{obj_name}/uniformity',
                    uniformity(z),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
            if eval_cfg.get('log_effective_rank', True):
                self.log(
                    f'val_{obj_name}/effective_rank',
                    torch.tensor(effective_rank(z)),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
        self._val_emb_acc.clear()

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward + compute loss, accuracy, and class probabilities.

        Returns:
            ``(loss, preds, probs, labels)``
        """
        logits = self(batch)
        labels = batch['label']
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        return loss, preds, probs, labels

    # ------------------------------------------------------------------
    # LightningModule hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, _, labels = self._shared_step(batch)
        acc = (preds == labels).float().mean()
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log('train_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        eval_cfg = self.cfg.get('eval', {})
        do_eval = eval_cfg.get('enabled', False) and 'val' in eval_cfg.get('splits', ['val'])

        if do_eval:
            # Run encode_all to get both logits and pt_mlp1 projections.
            z_global, z_consts, valids_list, z_valid_flat = self._encode_all(batch)
            logits = self.head(z_global, z_consts, valids_list)
            self._accumulate_embeddings(z_global, z_valid_flat)
        else:
            logits = self(batch)

        labels = batch['label']
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log(
            'val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        self.val_auroc.update(probs, labels)

    def on_validation_epoch_end(self) -> None:
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()
        eval_cfg = self.cfg.get('eval', {})
        if eval_cfg.get('enabled', False) and 'val' in eval_cfg.get('splits', ['val']):
            self._compute_and_log_embedding_metrics()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, probs, labels = self._shared_step(batch)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', acc, sync_dist=True)
        self.test_auroc.update(probs, labels)

    def on_test_epoch_end(self) -> None:
        self.log('test_auroc', self.test_auroc.compute())
        self.test_auroc.reset()

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        return self(batch).softmax(dim=-1)

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Remap state-dict keys from older checkpoint formats.

        Drops stale head projection keys (``head.global_proj.*``,
        ``head.const_proj.*``, ``head.global_agg.*``, ``head.const_phi.*``)
        from checkpoints saved before the projection was moved into the
        encoder's ``pt_mlp1`` pipeline.  Also handles size mismatches in
        ``backbone.*.pt_mlp*`` layers.
        """
        sd = checkpoint['state_dict']
        model_sd = self.state_dict()

        # Drop stale head projection keys — the head no longer has its own
        # projections; the encoder's pt_mlp1 is used directly in forward.
        _stale_prefixes = (
            'head.global_proj.',
            'head.const_proj.',
            'head.global_agg.',
            'head.const_phi.',
        )
        new_sd = {k: v for k, v in sd.items() if not k.startswith(_stale_prefixes)}

        # For any key whose shape no longer matches (e.g. pt_mlp hidden-dim
        # formula changed), fall back to the current model's initialisation.
        for k in list(new_sd):
            if k in model_sd and new_sd[k].shape != model_sd[k].shape:
                new_sd[k] = model_sd[k]

        checkpoint['state_dict'] = new_sd

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):  # type: ignore[override]
        opt_cfg = self.cfg.optimizer
        freeze = self.cfg.get('freeze_encoder', False)

        if freeze:
            # Backbone starts frozen; the BackboneFinetuning callback will
            # add backbone parameters to the optimiser when unfreezing.
            optimizer = torch.optim.AdamW(
                list(self.head.parameters()),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.get('weight_decay', 1e-5),
            )
        else:
            # Both backbone and head are optimised from epoch 0.
            backbone_lr = opt_cfg.get('backbone_lr', opt_cfg.lr)
            optimizer = torch.optim.AdamW(
                [
                    {'params': list(self.backbone.parameters()), 'lr': backbone_lr},
                    {'params': list(self.head.parameters()), 'lr': opt_cfg.lr},
                ],
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
