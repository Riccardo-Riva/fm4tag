"""Supervised fine-tuning LightningModule.

The classification pass is::

    out    = forward(batch)            # clean embeddings incl. 'aggregator'
    logits = head(out['aggregator'])   # classifier head

Loss composition is controlled by ``loss_weights`` (normalised to unit sum;
a weight of 0 disables that component entirely — not computed, not logged):

* ``cross_entropy``   — on the head's class logits.
* ``jet_contrastive`` — the batch is re-encoded under each augmentation view
  (exactly as in pretraining) and ``jet_contrastive_loss`` is applied to the
  per-view aggregator outputs.

Logged keys follow ``<split>/<component>/<metric>``:
``train/head/loss_ce``, ``train/aggregator/loss_contrastive``,
``val/head/acc``, ``val/head/auroc``; ``<split>/loss`` is the total.

``forward`` returns the same embeddings dict as
:class:`~fm4tag.modules.PretrainModule` (via
:func:`~fm4tag.modules.view_encoding.encode_clean`), so the
:class:`~fm4tag.callbacks.EmbeddingMetrics` callback works during fine-tuning
unchanged.

Staged unfreeze of the pretrained parts (encoders **and** aggregator) is
handled inside :meth:`configure_optimizers` via ``unfreeze_at_epoch``: every
parameter is in the optimiser from step 0 (so DDP registers a gradient-sync
hook for each), but the pretrained param group is held at ``lr=0`` until
``unfreeze_at_epoch`` and then ramps onto the head's cosine schedule.
``unfreeze_at_epoch=0`` trains everything from step 0; ``>= max_epochs`` keeps
the pretrained parts frozen for the whole run.  The encoders are stored as
``self.backbone``.  (This replaces the old ``PretrainedFinetuning`` callback,
whose ``requires_grad`` freeze happened before the DDP wrap and left the
unfrozen params without sync hooks → silent rank divergence.)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import MulticlassAUROC

from ..augmentations import Compose
from ..losses import normalize_loss_weights
from .view_encoding import (
    encode_clean,
    encode_constituent_view,
    encode_global_view,
    scatter_valid,
)

LOSS_COMPONENTS = ('cross_entropy', 'jet_contrastive')


class FinetuneModule(L.LightningModule):
    """Supervised fine-tuning of encoders + aggregator + classifier head.

    Args:
        encoders:   :class:`~torch.nn.ModuleDict` mapping object name → encoder.
                    Stored as ``self.backbone`` for ``BackboneFinetuning``
                    compatibility.
        aggregator: Module mapping per-object projections to the collective
                    embedding.  Shared (same weights) with pretraining.
        head:       Classifier head — receives the aggregator output.
        views:      Augmentation pipelines for the jet-contrastive component;
                    at least 2 are required when its weight is > 0 (may be
                    empty otherwise).
        global_object:       Name of the global object (e.g. ``'jets'``).
        constituent_objects: Names of the constituent objects.
        loss_weights: Mapping with keys among ``cross_entropy`` and
                    ``jet_contrastive`` (missing keys default to 0).
                    Normalised to unit sum; 0 disables the component.
        jet_contrastive_loss: Loss on aggregator outputs; called with the list
                    of per-view ``(B, D)`` tensors.
        n_classes:  Number of classes (for the AUROC metric).
        class_weights: Optional per-class weights for the cross-entropy.
        lr:           AdamW learning rate (classifier head).
        backbone_lr:  Base learning rate of the pretrained parts (encoders +
                      aggregator); ``None`` → same as ``lr``.  Held at 0 until
                      ``unfreeze_at_epoch``, then follows the cosine schedule.
        weight_decay: AdamW weight decay.
        warmup_frac:  Fraction of total steps for linear LR warmup before
                      cosine annealing.
        unfreeze_at_epoch: Epochs to hold the pretrained parts at ``lr=0``
                      before ramping them onto the head's schedule.  ``0``
                      trains everything from step 0; ``>= max_epochs`` keeps
                      them frozen for the whole run.
    """

    def __init__(
        self,
        encoders: torch.nn.ModuleDict,
        aggregator: nn.Module,
        head: nn.Module,
        views: Sequence[Compose],
        global_object: str,
        constituent_objects: Sequence[str],
        loss_weights: Mapping[str, float],
        jet_contrastive_loss: nn.Module,
        n_classes: int,
        class_weights: Sequence[float] | None = None,
        lr: float = 1e-4,
        backbone_lr: float | None = None,
        weight_decay: float = 1e-5,
        warmup_frac: float = 0.1,
        unfreeze_at_epoch: int = 0,
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

        if self.loss_weights['jet_contrastive'] > 0 and len(views) < 2:
            raise ValueError(
                'FinetuneModule requires at least 2 views when jet_contrastive '
                f'is active, got {len(views)}.'
            )

        # Named 'backbone' for BackboneFinetuning callback compatibility.
        self.backbone = encoders
        self.aggregator = aggregator
        self.head = head
        self.views = nn.ModuleList(views)
        self.jet_contrastive_loss = jet_contrastive_loss

        # Reconstruction heads are pretrain-only: freeze them so DDP does
        # not need find_unused_parameters.
        for enc in encoders.values():
            for name in ('reconstructor', 'cat_reconstructor', 'con_reconstructor'):
                if hasattr(enc, name):
                    getattr(enc, name).requires_grad_(False)

        self.global_object = global_object
        self.constituent_objects = list(constituent_objects)

        self.lr = lr
        self.backbone_lr = backbone_lr if backbone_lr is not None else lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac
        self.unfreeze_at_epoch = unfreeze_at_epoch

        if class_weights is not None:
            self.register_buffer(
                'class_weights', torch.tensor(list(class_weights), dtype=torch.float)
            )
        else:
            self.class_weights: torch.Tensor | None = None  # type: ignore[assignment]

        self.val_auroc = MulticlassAUROC(num_classes=n_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=n_classes, average='macro')

        # nn.Module args are excluded: their weights live in the checkpoint's
        # state_dict and they are rebuilt from config on load.
        self.save_hyperparameters(
            ignore=['encoders', 'aggregator', 'head', 'views', 'jet_contrastive_loss']
        )

    # ------------------------------------------------------------------
    # Forward: clean embeddings (same contract as PretrainModule)
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Clean (no augmentation) embeddings for every object and the aggregator.

        Returns:
            ``{'<obj>_embedding': (B or N_valid, d), ..., 'aggregator': (B, d)}``
        """
        return encode_clean(
            self.backbone,
            self.aggregator,
            batch,
            self.global_object,
            self.constituent_objects,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _encode_jet_views(self, batch: dict) -> list[torch.Tensor]:
        """One aggregator output per augmentation view (as in pretraining)."""
        enc_global = self.backbone[self.global_object]
        z_global_views = [
            encode_global_view(enc_global, view, batch['global'])[0]
            for view in self.views
        ]

        z_consts_per_obj: list[list[torch.Tensor]] = []
        valids_per_obj: list[torch.Tensor] = []
        for obj_name in self.constituent_objects:
            encoder = self.backbone[obj_name]
            const = batch['constituents'][obj_name]
            valids = const['valid']
            B, C = valids.shape
            valids_flat = valids.reshape(-1)

            z_views = []
            for view in self.views:
                z, _ = encode_constituent_view(encoder, view, const, valids_flat)
                z_views.append(scatter_valid(z, valids_flat, B, C))
            z_consts_per_obj.append(z_views)
            valids_per_obj.append(valids)

        return [
            self.aggregator(
                z_global_views[v],
                [z_views[v] for z_views in z_consts_per_obj],
                valids_per_obj,
            )
            for v in range(len(self.views))
        ]

    def _compute_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Returns ``(total_loss, logits, log_dict)``."""
        w = self.loss_weights

        out = self(batch)
        logits = self.head(out['aggregator'])

        total = logits.new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        if w['cross_entropy'] > 0:
            l_ce = F.cross_entropy(logits, batch['label'], weight=self.class_weights)
            total = total + w['cross_entropy'] * l_ce
            log_dict['head/loss_ce'] = l_ce

        if w['jet_contrastive'] > 0:
            # Costs V extra full encoder passes on top of the clean one above.
            z_jet_list = self._encode_jet_views(batch)
            l_jet = self.jet_contrastive_loss(z_jet_list)
            total = total + w['jet_contrastive'] * l_jet
            log_dict['aggregator/loss_contrastive'] = l_jet

        log_dict['loss'] = total
        return total, logits, log_dict

    # ------------------------------------------------------------------
    # LightningModule hooks
    # ------------------------------------------------------------------

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        loss, logits, log_dict = self._compute_loss(batch)
        labels = batch['label']
        on_step = split == 'train'

        for name, value in log_dict.items():
            # sync_dist=True costs one cross-rank reduce per metric per log.
            self.log(
                f'{split}/{name}',
                value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=(name == 'loss'),
                sync_dist=True,
            )

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log(
            f'{split}/head/acc', acc, on_step=False, on_epoch=True, sync_dist=True
        )

        if split == 'val':
            self.val_auroc.update(logits.softmax(dim=-1), labels)
        elif split == 'test':
            self.test_auroc.update(logits.softmax(dim=-1), labels)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'val')

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'test')

    def on_validation_epoch_end(self) -> None:
        self.log(
            'val/head/auroc',
            self.val_auroc.compute(),
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.val_auroc.reset()

    def on_test_epoch_end(self) -> None:
        self.log(
            'test/head/auroc',
            self.test_auroc.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.test_auroc.reset()

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        return self.head(self(batch)['aggregator']).softmax(dim=-1)

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
        # DDP-safe staged unfreeze: every parameter is in the optimiser from
        # step 0, so DDP registers a gradient-sync hook for each — a
        # requires_grad freeze skips hooks and the ranks silently diverge when
        # params later unfreeze.  The pretrained parts sit at lr=0 until
        # `unfreeze_at_epoch` via a per-group LR multiplier, then ramp onto the
        # head's cosine schedule.  (AdamW at lr=0 is a true freeze — both the
        # gradient step and decoupled weight decay scale by lr.)
        pretrained_params = [
            p
            for m in (self.backbone, self.aggregator)
            for p in m.parameters()
            if p.requires_grad  # skip frozen reconstructor heads
        ]
        optimizer = torch.optim.AdamW(
            [
                {'params': pretrained_params, 'lr': self.backbone_lr},  # group 0
                {'params': list(self.head.parameters()), 'lr': self.lr},  # group 1
            ],
            weight_decay=self.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(self.warmup_frac * total_steps))
        max_epochs = max(1, self.trainer.max_epochs or 1)
        unfreeze_step = round(self.unfreeze_at_epoch * total_steps / max_epochs)

        def head_factor(step: int) -> float:
            if step < warmup_steps:
                return 0.01 + 0.99 * step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        def backbone_factor(step: int) -> float:
            # Frozen (0 LR) until the unfreeze epoch, then joins the head curve.
            return 0.0 if step < unfreeze_step else head_factor(step)

        scheduler = LambdaLR(optimizer, lr_lambda=[backbone_factor, head_factor])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }
