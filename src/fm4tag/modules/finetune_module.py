"""Supervised fine-tuning LightningModule.

The classification pass is::

    out    = forward(batch)            # clean embeddings incl. 'aggregator'
    logits = head(out['aggregator'])   # classifier head

Loss composition is controlled by ``loss_weights`` (used as-is, not
normalised; a weight of 0 disables that component entirely — not computed,
not logged):

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

Staged unfreeze of the pretrained parts (``pretrained_parts``: the encoders
and/or the aggregator actually loaded from a checkpoint) is handled inside
:meth:`configure_optimizers` via ``unfreeze_at_epoch``: every
parameter is in the optimiser from step 0 (so DDP registers a gradient-sync
hook for each), but the pretrained param group is held at ``lr=0`` until
``unfreeze_at_epoch`` and then ramps onto the head's cosine schedule.
``unfreeze_at_epoch=0`` trains everything from step 0; ``>= max_epochs`` keeps
the pretrained parts at lr=0 for the whole run.  Parts that start from random
init train in the head group.  ``freeze_encoders`` is the real freeze of the
encoders (+ projectors): no gradient, no backward, eval mode.  The encoders are
stored as
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
from ..losses import validate_loss_weights
from .view_encoding import (
    batch_row_groups,
    encode_clean,
    encode_constituent_view,
    encode_global_view,
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
                    Used as-is (not normalised); 0 disables the component.
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
        warmup_epochs: Warmup length in epochs (may be fractional); when set
                      it replaces ``warmup_frac``, so runs of different
                      ``max_epochs`` can share the same warmup.
        schedule:     ``'cosine'`` (warmup, then cosine annealing to 0 over the
                      rest of the run) or ``'wsd'`` (warmup-stable-decay:
                      warmup, constant at the base lr, then a linear decay to
                      0 over the last ``decay_frac`` of the steps).
        decay_frac:   Fraction of total steps of the final linear decay
                      (``schedule='wsd'`` only).
        unfreeze_at_epoch: Epochs to hold the pretrained parts at ``lr=0``
                      before ramping them onto the head's schedule.  ``0``
                      trains everything from step 0; ``>= max_epochs`` keeps
                      them frozen for the whole run.
        freeze_encoders: Freeze the encoders and their projectors for the whole
                      run (``requires_grad=False`` before the DDP wrap — safe,
                      as they are never unfrozen — and eval mode).  Nothing
                      upstream of the aggregator then requires grad, so the
                      encoders run without autograd graph or backward.
        pretrained_parts: Which of ``'encoders'``, ``'aggregator'`` start from
                      pretrained weights (set by the runner from what it
                      loaded): those train at ``backbone_lr`` with the staged
                      unfreeze, the rest with the head at ``lr``.
        chunk_size:   Jets per intersample-attention chunk: each jet (and each
                      of its constituents) attends only within its chunk of
                      ``chunk_size`` consecutive jets.  ``None`` = the whole
                      batch is one chunk.
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
        warmup_epochs: float | None = None,
        schedule: str = 'cosine',
        decay_frac: float = 0.2,
        unfreeze_at_epoch: int = 0,
        chunk_size: int | None = None,
        freeze_encoders: bool = False,
        pretrained_parts: Sequence[str] = ('encoders', 'aggregator'),
    ) -> None:
        super().__init__()

        unknown_parts = set(pretrained_parts) - {'encoders', 'aggregator'}
        if unknown_parts:
            raise ValueError(f'Unknown pretrained_parts {sorted(unknown_parts)}.')

        unknown = set(loss_weights) - set(LOSS_COMPONENTS)
        if unknown:
            raise ValueError(
                f'Unknown loss_weights keys {sorted(unknown)}; '
                f'expected a subset of {LOSS_COMPONENTS}.'
            )
        weights = {k: float(loss_weights.get(k, 0.0)) for k in LOSS_COMPONENTS}
        self.loss_weights = validate_loss_weights(weights)

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
        self.freeze_encoders = freeze_encoders
        if freeze_encoders:
            encoders.requires_grad_(False)
        self.pretrained_parts = tuple(pretrained_parts)

        self.global_object = global_object
        self.constituent_objects = list(constituent_objects)

        self.lr = lr
        self.backbone_lr = backbone_lr if backbone_lr is not None else lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac
        self.warmup_epochs = warmup_epochs
        if schedule not in ('cosine', 'wsd'):
            raise ValueError(f"schedule must be 'cosine' or 'wsd', got {schedule!r}")
        if not 0.0 < decay_frac < 1.0:
            raise ValueError(f'decay_frac must be in (0, 1), got {decay_frac}')
        self.schedule = schedule
        self.decay_frac = decay_frac
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.chunk_size = chunk_size

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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoders:
            self.backbone.eval()  # frozen features: no dropout
        return self

    # ------------------------------------------------------------------
    # Forward: clean embeddings (same contract as PretrainModule)
    # ------------------------------------------------------------------

    def _row_groups(self, batch: dict) -> dict:
        return batch_row_groups(
            batch, self.global_object, self.constituent_objects, self.chunk_size
        )

    def forward(
        self, batch: dict, groups: dict | None = None
    ) -> dict[str, torch.Tensor]:
        """Clean (no augmentation) embeddings for every object and the aggregator.

        ``groups`` (from :meth:`_row_groups`) is rebuilt when not given.

        Returns:
            ``{'<obj>_embedding': (B or N_valid, d), ..., 'aggregator': (B, d)}``
        """
        return encode_clean(
            self.backbone,
            self.aggregator,
            batch,
            self.global_object,
            self.constituent_objects,
            groups if groups is not None else self._row_groups(batch),
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _encode_jet_views(
        self,
        batch: dict,
        z_clean: torch.Tensor | None = None,
        groups: dict | None = None,
    ) -> list[torch.Tensor]:
        """One aggregator output per augmentation view (as in pretraining).

        An identity view runs the same encoders, projector and aggregator over
        the same inputs as the clean pass, so it reproduces it exactly.  When
        ``z_clean`` (the clean ``aggregator`` output) is supplied it is reused
        for those views instead of being recomputed — with the default
        ``[CutMix+Mixup, Identity]`` pair this turns three full encoder passes
        per step into two, bit-for-bit unchanged.
        """
        if groups is None:
            groups = self._row_groups(batch)
        views = list(self.views)
        todo = [
            v for v, view in enumerate(views) if z_clean is None or not view.is_identity
        ]
        if not todo:
            return [z_clean for _ in views]

        enc_global = self.backbone[self.global_object]
        z_global_views = {
            v: encode_global_view(
                enc_global, views[v], batch['global'], groups[self.global_object]
            )[0]
            for v in todo
        }

        z_consts_per_obj: list[dict[int, torch.Tensor]] = []
        valids_per_obj: list[torch.Tensor] = []
        for obj_name in self.constituent_objects:
            encoder = self.backbone[obj_name]
            const = batch['constituents'][obj_name]
            valids = const['valid']
            valids_flat = valids.reshape(-1)

            z_views = {}
            for v in todo:
                z_views[v], _ = encode_constituent_view(
                    encoder, views[v], const, valids_flat, groups[obj_name]
                )
            z_consts_per_obj.append(z_views)
            valids_per_obj.append(valids)

        return [
            z_clean
            if v not in z_global_views
            else self.aggregator(
                z_global_views[v],
                [z_views[v] for z_views in z_consts_per_obj],
                valids_per_obj,
            )
            for v in range(len(views))
        ]

    def _compute_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Returns ``(total_loss, logits, log_dict)``."""
        w = self.loss_weights
        groups = self._row_groups(batch)  # shared by the clean pass and every view

        out = self(batch, groups)
        logits = self.head(out['aggregator'])

        total = logits.new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        if w['cross_entropy'] > 0:
            l_ce = F.cross_entropy(logits, batch['label'], weight=self.class_weights)
            total = total + w['cross_entropy'] * l_ce
            log_dict['head/loss_ce'] = l_ce

        if w['jet_contrastive'] > 0:
            # Costs one extra full encoder pass per NON-identity view; identity
            # views reuse the clean aggregator output computed above.
            z_jet_list = self._encode_jet_views(
                batch, z_clean=out['aggregator'], groups=groups
            )
            l_jet = self.jet_contrastive_loss(
                z_jet_list, groups=groups[self.global_object]
            )
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
        self.log(f'{split}/head/acc', acc, on_step=False, on_epoch=True, sync_dist=True)

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
        # Pretrained parts → 'backbone' group; randomly initialised ones train
        # with the head.  Frozen params (reconstructors, frozen encoders) are
        # left out.
        pretrained_params, fresh_params = [], list(self.head.parameters())
        for name, part in (
            ('encoders', self.backbone),
            ('aggregator', self.aggregator),
        ):
            params = [p for p in part.parameters() if p.requires_grad]
            if name in self.pretrained_parts:
                pretrained_params += params
            else:
                fresh_params += params
        # 'name' labels the LearningRateMonitor series (lr-AdamW/backbone,
        # lr-AdamW/head) so the staged unfreeze is visible in the logs.
        groups = [{'params': fresh_params, 'lr': self.lr, 'name': 'head'}]
        if pretrained_params:
            groups.insert(
                0,
                {
                    'params': pretrained_params,
                    'lr': self.backbone_lr,
                    'name': 'backbone',
                },
            )
        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        max_epochs = max(1, self.trainer.max_epochs or 1)
        if self.warmup_epochs is not None:
            warmup_steps = max(1, round(self.warmup_epochs * total_steps / max_epochs))
        else:
            warmup_steps = max(1, int(self.warmup_frac * total_steps))
        unfreeze_step = round(self.unfreeze_at_epoch * total_steps / max_epochs)
        decay_steps = max(1, round(self.decay_frac * total_steps))
        decay_start = total_steps - decay_steps
        if self.schedule == 'wsd' and warmup_steps > decay_start:
            raise ValueError(
                f'wsd: the warmup ({warmup_steps} steps) ends after the decay '
                f'starts (step {decay_start} of {total_steps})'
            )

        def ramp(step: int) -> float:
            return 0.01 + 0.99 * step / warmup_steps

        def head_factor(step: int) -> float:
            if step < warmup_steps:
                return ramp(step)
            if self.schedule == 'wsd':
                # stable at the base lr, then linear to 0 at the last step
                if step < decay_start:
                    return 1.0
                return max(0.0, (total_steps - step) / decay_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        def backbone_factor(step: int) -> float:
            # Frozen (0 LR) until the unfreeze epoch.
            if step < unfreeze_step:
                return 0.0
            if self.schedule == 'wsd':
                # its own warmup after the unfreeze, then the head's curve
                since = step - unfreeze_step
                own = ramp(since) if since < warmup_steps else 1.0
                return min(own, head_factor(step))
            return head_factor(step)  # cosine: joins the head curve at once

        factors = [backbone_factor, head_factor] if pretrained_params else [head_factor]
        scheduler = LambdaLR(optimizer, lr_lambda=factors)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }
