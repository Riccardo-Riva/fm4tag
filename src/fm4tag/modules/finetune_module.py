"""Supervised fine-tuning Lightning module.

Wraps a :class:`~torch.nn.ModuleDict` of pretrained (or randomly initialised)
encoders, a shared :class:`~fm4tag.models.aggregator.JetAggregator`, and a
:class:`~fm4tag.models.heads.MultiStreamClassifierHead` into a full Lightning
module that supports:

* ``fit``     — supervised training (with optional encoder freezing /
                progressive unfreezing via the ``BackboneFinetuning`` callback)
* ``test``    — evaluation on a labelled test set
* ``predict`` — inference returning class probabilities (softmax)

The forward pass is::

    z_global, z_consts, valids = _encode_all(batch)   # POINT A (encoder.projector)
    z_jet  = aggregator(z_global, z_consts, valids)    # POINT B
    logits = head(z_jet)                               # POINT C

The loss is a composable :class:`~fm4tag.modules.losses.FinetuneLoss`.  If it
contains a jet-level contrastive term (one consuming ``z_jet_list``), the module
re-encodes the batch under each of ``views`` — exactly as in pretraining — to
build one ``z_jet`` per view and feeds the list to the loss.

The encoders are stored as ``self.backbone`` (an :class:`~torch.nn.ModuleDict`)
so that Lightning's built-in :class:`~lightning.pytorch.callbacks.BackboneFinetuning`
callback can locate and unfreeze them automatically when ``freeze_encoder=True``
in the config.
"""

from __future__ import annotations

from collections import defaultdict

import lightning as L
import torch
from einops import rearrange
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.classification import MulticlassAUROC

from ..augmentations import Compose
from ..metrics import effective_rank, uniformity
from ..models import embed_data
from ..models.aggregator import JetAggregator
from ..models.heads import MultiStreamClassifierHead
from ..utils.ddp import gather_embeddings_sized
from .losses import FinetuneLoss, loss_wants
from .view_encoding import (
    encode_constituent_view,
    encode_global_view,
    scatter_valid,
)


class FinetuneModule(L.LightningModule):
    """Lightning module for supervised fine-tuning of all encoders + classifier.

    Args:
        encoders:   :class:`~torch.nn.ModuleDict` mapping each object name to its
                    encoder — a :class:`GlobalEncoder` for the global object and
                    an :class:`Encoder` for each constituent type.  Stored as
                    ``self.backbone`` for ``BackboneFinetuning`` compatibility.
        aggregator: :class:`JetAggregator` mapping per-object projections to a
                    single jet embedding ``z_jet`` (POINT B).  Shared (same
                    weights) with pretraining.
        head:       :class:`MultiStreamClassifierHead` — receives ``z_jet``.
        loss:       :class:`~fm4tag.modules.losses.FinetuneLoss` — composable,
                    weighted sum of loss terms (cross-entropy, optional jet
                    contrastive).
        views:      List of :class:`~fm4tag.augmentations.Compose` pipelines used
                    only when the loss contains a jet-contrastive term; one
                    ``z_jet`` is produced per view.  May be empty otherwise.
        cfg:        Full Hydra config.  Relevant sub-keys:
                    ``cfg.optimizer``, ``cfg.freeze_encoder``,
                    ``cfg.class_dict``, ``cfg.global_object``,
                    ``cfg.constituent_objects``.
    """

    def __init__(
        self,
        encoders: torch.nn.ModuleDict,
        aggregator: JetAggregator,
        head: MultiStreamClassifierHead,
        loss: FinetuneLoss,
        views: list[Compose],
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=['encoders', 'aggregator', 'head', 'loss', 'views']
        )

        # Named 'backbone' for BackboneFinetuning callback compatibility.
        self.backbone = encoders
        self.aggregator = aggregator
        self.head = head
        self.loss = loss
        self.views = torch.nn.ModuleList(views)
        self.cfg = cfg

        # Whether the loss needs per-view jet embeddings (jet contrastive).
        self._needs_jet_views = loss_wants(self.loss, 'z_jet_list')
        if self._needs_jet_views and len(self.views) < 2:
            raise ValueError(
                "FinetuneModule needs at least 2 views when the loss contains a "
                "jet-contrastive term (z_jet_list); got "
                f"{len(self.views)}."
            )

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
        """Run all encoders and apply ``projector`` projections (POINT A).

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
        X_global = enc_global(batch['global'])  # (B, F_g, dim)
        z_global = enc_global.projector(X_global.flatten(1))  # (B, g_dim)

        # ── Constituents ─────────────────────────────────────────────────────
        z_consts: list[torch.Tensor] = []
        valids_list: list[torch.Tensor] = []
        z_valid_flat: list[torch.Tensor] = []

        for obj_name in self.cfg.constituent_objects:
            encoder = self.backbone[obj_name]
            const = batch['constituents'][obj_name]

            x_categ = const['categorical']  # (B, C, F_cat)
            x_cont = const['continuous']  # (B, C, F_con)
            valids = const['valid']  # (B, C)

            B, C, _ = x_categ.shape
            valids_flat = rearrange(valids, 'b c -> (b c)')
            x_categ_flat = rearrange(x_categ, 'b c f -> (b c) f')
            x_cont_flat = rearrange(x_cont, 'b c f -> (b c) f')

            # Encode only valid constituents for efficiency.
            x_cat_enc, x_con_enc = embed_data(
                x_categ_flat[valids_flat], x_cont_flat[valids_flat], encoder
            )
            X_valid = encoder(x_cat_enc, x_con_enc)  # (N_valid, F, dim)
            z_valid = encoder.projector(X_valid.flatten(1, 2))  # (N_valid, c_dim)

            # Scatter projected embeddings back into (B, C, c_dim).
            z_all = scatter_valid(z_valid, valids_flat, B, C)

            z_consts.append(z_all)
            valids_list.append(valids)
            z_valid_flat.append(z_valid)

        return z_global, z_consts, valids_list, z_valid_flat

    def _encode_jet_views(self, batch: dict) -> list[torch.Tensor]:
        """Build one ``z_jet`` per view (POINT B) for the jet-contrastive term.

        Re-encodes the batch under each augmentation view exactly as in
        pretraining, projects (POINT A), aggregates across all objects, and
        returns the per-view list of ``(B, jet_dim)`` jet embeddings.
        """
        global_name = self.cfg.global_object
        enc_global = self.backbone[global_name]
        x_orig = batch['global']

        # Per-view global projections.
        z_global_views = [
            encode_global_view(enc_global, view, x_orig)[0] for view in self.views
        ]

        # Per-view constituent projections (scattered) + valids.
        z_consts_per_obj: list[list[torch.Tensor]] = []
        valids_per_obj: list[torch.Tensor] = []
        for obj_name in self.cfg.constituent_objects:
            encoder = self.backbone[obj_name]
            const = batch['constituents'][obj_name]
            valids = const['valid']
            B, C = valids.shape
            valids_flat = rearrange(valids, 'b c -> (b c)')

            z_views = []
            for view in self.views:
                z, _ = encode_constituent_view(encoder, view, const, valids_flat)
                z_views.append(scatter_valid(z, valids_flat, B, C))
            z_consts_per_obj.append(z_views)
            valids_per_obj.append(valids)

        # Aggregate per view (POINT B).
        z_jet_list: list[torch.Tensor] = []
        for v in range(len(self.views)):
            z_consts_v = [z_views[v] for z_views in z_consts_per_obj]
            z_jet_list.append(
                self.aggregator(z_global_views[v], z_consts_v, valids_per_obj)
            )
        return z_jet_list

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        """Encode all objects, aggregate into a jet embedding, and classify.

        Args:
            batch: Dict from the datamodule collate function.

        Returns:
            ``(B, y_dim)`` class logits.
        """
        z_global, z_consts, valids_list, _ = self._encode_all(batch)
        z_jet = self.aggregator(z_global, z_consts, valids_list)  # POINT B
        return self.head(z_jet)  # POINT C

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self, batch: dict, logits: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate the composable finetune loss for a batch.

        Always passes ``logits``/``labels``/``class_weights``; additionally
        passes per-view ``z_jet_list`` when the loss contains a jet-contrastive
        term.
        """
        kwargs: dict = {
            'logits': logits,
            'labels': batch['label'],
            'class_weights': self.class_weights,
        }
        if self._needs_jet_views:
            kwargs['z_jet_list'] = self._encode_jet_views(batch)
        return self.loss(**kwargs)

    def _log_loss_components(
        self, loss_log: dict[str, torch.Tensor], split: str
    ) -> None:
        """Log each loss sub-component (everything except the top-level total)."""
        for k, v in loss_log.items():
            if k == 'loss':
                continue
            self.log(
                f'{split}_{k}', v, on_step=False, on_epoch=True, sync_dist=True
            )

    # ------------------------------------------------------------------
    # Online embedding metric helpers
    # ------------------------------------------------------------------

    def _accumulate_embeddings(
        self,
        z_global: torch.Tensor,
        z_valid_flat: list[torch.Tensor],
    ) -> None:
        """Append pre-computed ``projector`` projections to the embedding store.

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
                self._val_emb_acc[obj_name].append(z_valid[:remaining].detach().cpu())

    def _compute_and_log_embedding_metrics(self) -> None:
        """Concatenate accumulated embeddings, gather (DDP), compute and log metrics."""
        eval_cfg = self.cfg.get('eval', {})
        # Iterate over ALL backbone keys so every DDP rank participates in the
        # same collectives in the same order.  Skipping objects based on a local
        # condition (e.g. empty chunks) would cause other ranks to block forever
        # on the collective, producing an NCCL watchdog hang.
        for obj_name in list(self.backbone.keys()):
            chunks = self._val_emb_acc.get(obj_name, [])
            z_local = torch.cat(chunks, dim=0) if chunks else None  # (N_local, D) CPU

            z = gather_embeddings_sized(
                z_local, self.trainer.world_size, self.all_gather, self.device
            )
            if z is None:
                continue

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
                    torch.as_tensor(effective_rank(z)),
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Run forward + compute loss, predictions, and class probabilities.

        Returns:
            ``(loss, preds, probs, labels, loss_log)``
        """
        logits = self(batch)
        loss, loss_log = self._compute_loss(batch, logits)
        labels = batch['label']
        preds = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        return loss, preds, probs, labels, loss_log

    # ------------------------------------------------------------------
    # LightningModule hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, _, labels, loss_log = self._shared_step(batch)
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
        self._log_loss_components(loss_log, 'train')
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        eval_cfg = self.cfg.get('eval', {})
        do_eval = eval_cfg.get('enabled', False) and 'val' in eval_cfg.get(
            'splits', ['val']
        )

        if do_eval:
            # Run encode_all to get both logits and projector embeddings.
            z_global, z_consts, valids_list, z_valid_flat = self._encode_all(batch)
            z_jet = self.aggregator(z_global, z_consts, valids_list)
            logits = self.head(z_jet)
            self._accumulate_embeddings(z_global, z_valid_flat)
        else:
            logits = self(batch)

        loss, loss_log = self._compute_loss(batch, logits)
        labels = batch['label']
        preds = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        self._log_loss_components(loss_log, 'val')
        self.val_auroc.update(probs, labels)

    def on_validation_epoch_end(self) -> None:
        self.log(
            'val_auroc',
            self.val_auroc.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.val_auroc.reset()
        eval_cfg = self.cfg.get('eval', {})
        if eval_cfg.get('enabled', False) and 'val' in eval_cfg.get('splits', ['val']):
            self._compute_and_log_embedding_metrics()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, probs, labels, _ = self._shared_step(batch)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', acc, sync_dist=True)
        self.test_auroc.update(probs, labels)

    def on_test_epoch_end(self) -> None:
        self.log(
            'test_auroc',
            self.test_auroc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
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
        encoder's ``projector`` pipeline.  Also handles size mismatches in
        ``backbone.*.projector.*`` layers.
        """
        sd = checkpoint['state_dict']
        model_sd = self.state_dict()

        # Drop stale head projection keys — the head no longer has its own
        # projections; the encoder's projector is used directly in forward.
        _stale_prefixes = (
            'head.global_proj.',
            'head.const_proj.',
            'head.global_agg.',
            'head.const_phi.',
        )
        new_sd = {k: v for k, v in sd.items() if not k.startswith(_stale_prefixes)}

        # For any key whose shape no longer matches (e.g. projector hidden-dim
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

        # The aggregator + head are always trained from epoch 0; only the
        # backbone may start frozen (the BackboneFinetuning callback adds it
        # to the optimiser when it unfreezes).
        head_params = list(self.aggregator.parameters()) + list(self.head.parameters())

        if freeze:
            optimizer = torch.optim.AdamW(
                head_params,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.get('weight_decay', 1e-5),
            )
        else:
            backbone_lr = opt_cfg.get('backbone_lr', opt_cfg.lr)
            optimizer = torch.optim.AdamW(
                [
                    {'params': list(self.backbone.parameters()), 'lr': backbone_lr},
                    {'params': head_params, 'lr': opt_cfg.lr},
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
