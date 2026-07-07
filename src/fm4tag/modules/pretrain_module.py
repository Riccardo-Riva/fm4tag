"""Multi-view contrastive + denoising pretraining LightningModule.

For each object (global + constituents) and each batch:

1. **Contrastive** — all ``V`` views are encoded and projected
   (``encoder.projector``); ``contrastive_loss`` treats all views of the same
   sample as positives.
2. **Denoising** — the *first* view's encoded representation reconstructs the
   original (pre-augmentation) features: cross-entropy for categorical,
   MSE for continuous.
3. **Jet contrastive** — the per-view projections of *all* objects are
   aggregated into one embedding per view and ``jet_contrastive_loss`` is
   applied once per batch.

Which components run is controlled entirely by ``loss_weights``: a weight of
0 means that component is neither computed nor logged.  Logged keys follow
``<split>/<component>/<metric>``, e.g. ``train/tracks_embedding/loss_contrastive``
and ``train/aggregator/loss_contrastive``; ``<split>/loss`` is the total.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import lightning as L
import torch
from einops import rearrange
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..augmentations import Compose
from ..losses import denoising_cat_loss, denoising_con_loss, normalize_loss_weights
from .view_encoding import (
    encode_clean,
    encode_constituent_view,
    encode_global_view,
    scatter_valid,
)

LOSS_COMPONENTS = ('contrastive', 'denoising_cat', 'denoising_con', 'jet_contrastive')


class PretrainModule(L.LightningModule):
    """Multi-view contrastive + denoising pretraining module.

    Args:
        encoders:    :class:`~torch.nn.ModuleDict` mapping object name → encoder.
        aggregator:  Module mapping per-object projections to a single
                     collective embedding (shared weights with fine-tuning).
                     Receives no gradient unless ``jet_contrastive`` is active.
        views:       Augmentation :class:`~fm4tag.augmentations.Compose`
                     pipelines, one per view.  The first view is the denoising
                     input; the original batch is always the reconstruction
                     target.  At least 2 views are required when a contrastive
                     component is active, 1 otherwise.
        global_object:       Name of the global object (e.g. ``'jets'``).
        constituent_objects: Names of the constituent objects.
        loss_weights: Mapping with keys among ``contrastive``,
                     ``denoising_cat``, ``denoising_con``, ``jet_contrastive``
                     (missing keys default to 0).  Normalised to unit sum, so
                     only the relative balance matters; a weight of 0 disables
                     that component entirely (not computed, not logged).
        contrastive_loss:     Loss on per-object projections; called with the
                              list of per-view ``(N, D)`` tensors.
        jet_contrastive_loss: Loss on aggregator outputs; called with the list
                              of per-view ``(B, D)`` tensors.
        lr:           AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_frac:  Fraction of total steps used for linear LR warmup before
                      cosine annealing.

    Valid-mask consistency
    ----------------------
    All views share the **original** ``valid`` mask when flattening
    constituent tensors, so the constituent count ``N`` is identical across
    views — required by the flat per-object losses.  A future view that
    changes the mask (variable constituent count) is only compatible with the
    aggregator-level loss; thread a per-view mask through
    :func:`encode_constituent_view` when that lands.
    """

    def __init__(
        self,
        encoders: torch.nn.ModuleDict,
        aggregator: nn.Module,
        views: Sequence[Compose],
        global_object: str,
        constituent_objects: Sequence[str],
        loss_weights: Mapping[str, float],
        contrastive_loss: nn.Module,
        jet_contrastive_loss: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_frac: float = 0.1,
    ) -> None:
        super().__init__()

        unknown = set(loss_weights) - set(LOSS_COMPONENTS)
        if unknown:
            raise ValueError(
                f'Unknown loss_weights keys {sorted(unknown)}; '
                f'expected a subset of {LOSS_COMPONENTS}.'
            )
        weights = {k: float(loss_weights.get(k, 0.0)) for k in LOSS_COMPONENTS}
        self.loss_weights = normalize_loss_weights(weights)

        needs_views = (
            self.loss_weights['contrastive'] > 0
            or self.loss_weights['jet_contrastive'] > 0
        )
        if needs_views and len(views) < 2:
            raise ValueError(
                'PretrainModule requires at least 2 views when a contrastive '
                f'component is active, got {len(views)}.'
            )
        if len(views) < 1:
            raise ValueError('PretrainModule requires at least 1 view.')

        self.encoders = encoders
        self.aggregator = aggregator
        self.views = nn.ModuleList(views)
        self.contrastive_loss = contrastive_loss
        self.jet_contrastive_loss = jet_contrastive_loss

        # Freeze heads that can never receive gradients under these
        # loss_weights — otherwise DDP needs find_unused_parameters.
        w = self.loss_weights
        heads_off = []
        if w['denoising_cat'] == 0:
            heads_off.append('cat_reconstructor')
        if w['denoising_con'] == 0:
            heads_off += ['con_reconstructor', 'reconstructor']
        for enc in encoders.values():
            for name in heads_off:
                if hasattr(enc, name):
                    getattr(enc, name).requires_grad_(False)
            if w['contrastive'] == 0 and w['jet_contrastive'] == 0:
                enc.projector.requires_grad_(False)
        if w['jet_contrastive'] == 0:
            aggregator.requires_grad_(False)

        self.global_object = global_object
        self.constituent_objects = list(constituent_objects)

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac

        # nn.Module args are excluded: their weights live in the checkpoint's
        # state_dict and they are rebuilt from config on load.  Everything
        # else is a plain scalar/dict/list, so the checkpoint stays clean
        # (loadable with torch's weights_only=True default).
        self.save_hyperparameters(
            ignore=[
                'encoders',
                'aggregator',
                'views',
                'contrastive_loss',
                'jet_contrastive_loss',
                'constituent_objects', #TODO: remove when hparams conflicct with DataModule is resolved
            ]
        )

    # ------------------------------------------------------------------
    # Per-object losses
    # ------------------------------------------------------------------

    def _n_encode_views(self) -> int:
        """How many views need encoding: all for contrastive, 1 for denoising."""
        w = self.loss_weights
        if w['contrastive'] > 0 or w['jet_contrastive'] > 0:
            return len(self.views)
        return 1

    def _global_object_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]:
        """Loss components for the global object.

        Returns ``(obj_loss, obj_logs, z_views)`` where ``z_views`` is the list
        of per-view ``(B, d)`` projections, reused for jet aggregation.
        """
        w = self.loss_weights
        encoder = self.encoders[self.global_object]
        x_orig = batch['global']  # (B, F_g)

        zs: list[torch.Tensor] = []
        X_first: torch.Tensor | None = None
        for i in range(self._n_encode_views()):
            z, X = encode_global_view(encoder, self.views[i], x_orig)
            zs.append(z)
            if i == 0:
                X_first = X

        obj_loss = x_orig.new_zeros(())
        logs: dict[str, torch.Tensor] = {}

        if w['contrastive'] > 0:
            l_con = self.contrastive_loss(zs)
            obj_loss = obj_loss + w['contrastive'] * l_con
            logs['loss_contrastive'] = l_con

        # The global object has only continuous features.
        if w['denoising_con'] > 0:
            con_outs = encoder.reconstructor(X_first)  # list of (B, 1)
            l_den = denoising_con_loss(con_outs, x_orig)
            obj_loss = obj_loss + w['denoising_con'] * l_den
            logs['loss_denoising_con'] = l_den

        return obj_loss, logs, zs

    def _constituent_loss(
        self, batch: dict, obj_name: str, needs_jet: bool
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        list[torch.Tensor] | None,
        torch.Tensor,
    ]:
        """Loss components for one constituent type.

        Returns ``(obj_loss, obj_logs, z_views, valids)`` where ``z_views`` —
        present only when ``needs_jet`` — is the list of per-view ``(B, C, d)``
        projections scattered back onto the padded grid, and ``valids`` is the
        ``(B, C)`` mask.
        """
        w = self.loss_weights
        encoder = self.encoders[obj_name]
        const = batch['constituents'][obj_name]
        valids = const['valid']  # (B, C)
        valids_flat = rearrange(valids, 'b c -> (b c)')

        zs: list[torch.Tensor] = []
        X_first: torch.Tensor | None = None
        for i in range(self._n_encode_views()):
            z, X = encode_constituent_view(encoder, self.views[i], const, valids_flat)
            zs.append(z)
            if i == 0:
                X_first = X

        obj_loss = zs[0].new_zeros(())
        logs: dict[str, torch.Tensor] = {}

        if w['contrastive'] > 0:
            l_con = self.contrastive_loss(zs)
            obj_loss = obj_loss + w['contrastive'] * l_con
            logs['loss_contrastive'] = l_con

        if w['denoising_cat'] > 0:
            cat_outs = encoder.cat_reconstructor(
                X_first[:, : encoder.num_categories, :]
            )
            x_categ = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_flat]
            l_cat = denoising_cat_loss(cat_outs, x_categ)
            obj_loss = obj_loss + w['denoising_cat'] * l_cat
            logs['loss_denoising_cat'] = l_cat

        if w['denoising_con'] > 0:
            con_outs = encoder.con_reconstructor(
                X_first[:, encoder.num_categories :, :]
            )
            x_cont = rearrange(const['continuous'], 'b c f -> (b c) f')[valids_flat]
            l_den = denoising_con_loss(con_outs, x_cont)
            obj_loss = obj_loss + w['denoising_con'] * l_den
            logs['loss_denoising_con'] = l_den

        z_views: list[torch.Tensor] | None = None
        if needs_jet:
            B, C = valids.shape
            z_views = [scatter_valid(z, valids_flat, B, C) for z in zs]

        return obj_loss, logs, z_views, valids

    # ------------------------------------------------------------------
    # Total loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        w = self.loss_weights
        needs_jet = w['jet_contrastive'] > 0

        total = batch['global'].new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        obj_loss, obj_logs, z_global_views = self._global_object_loss(batch)
        total = total + obj_loss
        for k, v in obj_logs.items():
            log_dict[f'{self.global_object}_embedding/{k}'] = v

        z_consts_per_obj: list[list[torch.Tensor]] = []  # [obj][view] → (B,C,d)
        valids_per_obj: list[torch.Tensor] = []
        for obj_name in self.constituent_objects:
            obj_loss, obj_logs, z_views, valids = self._constituent_loss(
                batch, obj_name, needs_jet
            )
            total = total + obj_loss
            for k, v in obj_logs.items():
                log_dict[f'{obj_name}_embedding/{k}'] = v
            if needs_jet:
                z_consts_per_obj.append(z_views)
                valids_per_obj.append(valids)

        if needs_jet:
            z_jet_list = [
                self.aggregator(
                    z_global_views[v],
                    [z_views[v] for z_views in z_consts_per_obj],
                    valids_per_obj,
                )
                for v in range(len(self.views))
            ]
            jet_loss = self.jet_contrastive_loss(z_jet_list)
            total = total + w['jet_contrastive'] * jet_loss
            log_dict['aggregator/loss_contrastive'] = jet_loss

        log_dict['loss'] = total
        return total, log_dict

    # ------------------------------------------------------------------
    # Inference: clean embeddings for downstream use / eval metrics
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Clean (no augmentation) embeddings for every object and the aggregator.

        Returns:
            ``{'<obj>_embedding': (B or N_valid, d), ..., 'aggregator': (B, d)}``
        """
        return encode_clean(
            self.encoders,
            self.aggregator,
            batch,
            self.global_object,
            self.constituent_objects,
        )

    # ------------------------------------------------------------------
    # LightningModule hooks
    # ------------------------------------------------------------------

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        loss, log_dict = self._compute_loss(batch)
        on_step = split == 'train'
        for name, value in log_dict.items():
            self.log(
                f'{split}/{name}',
                value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=(name == 'loss'),
                sync_dist=True,
            )
        return loss
    
#    def on_after_backward(self):
#        for name, p in self.named_parameters():
#            if p.requires_grad and p.grad is None:
#                print(f"NO GRAD: {name}")
#
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'val')

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'test')

    def configure_optimizers(self):  # type: ignore[override]
        # Skip frozen (zero-weight) heads.
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(self.warmup_frac * total_steps))

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
