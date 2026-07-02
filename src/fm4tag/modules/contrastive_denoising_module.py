"""Contrastive + denoising pretraining module with multi-view augmentations."""

from __future__ import annotations

from collections import defaultdict

import lightning as L
import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..augmentations import Compose
from ..metrics import compute_metric
from ..models import embed_data
from ..models.aggregator import TransformerAggregator
from ..models.backbones import Encoder, GlobalEncoder
from ..utils.ddp import gather_embeddings_sized
from .losses import PretrainLoss, loss_wants
from .view_encoding import (
    encode_constituent_view,
    encode_global_view,
    scatter_valid,
)


class ContrastiveDenoisingModule(L.LightningModule):
    """Multi-view contrastive + denoising pretraining Lightning module.

    For each object (global + constituents) and each batch:

    1. **Contrastive** — all ``V`` views are encoded and projected (POINT A,
       ``encoder.projector``); the contrastive term treats all views of the
       same sample as positives and all views of other samples as negatives.

    2. **Denoising** — the *first* view's encoded representation reconstructs
       the original (pre-augmentation) features via ``sep``-style MLP heads;
       supervised with cross-entropy (categorical) + MSE (continuous).

    3. **Jet contrastive (optional)** — if the loss contains a term consuming
       ``z_jet_list``, the per-view projections of *all* objects are aggregated
       into one ``z_jet`` per view (POINT B, via :class:`TransformerAggregator`) and a
       jet-level contrastive term is applied once per batch.

    The loss is fully described by the injected :class:`PretrainLoss`: each term
    declares the inputs it needs and is dispatched only when those inputs are
    available (per-object scope vs jet-level scope), so a single loss object
    drives both scopes without double-counting.

    Beyond the loss, this module owns the training loop, logging, epoch-end
    summaries, embedding metric accumulation (uniformity, effective rank), and
    the optimizer.

    Valid-mask consistency
    ----------------------
    All views share the **original** ``valid`` mask when flattening
    constituent tensors, so the constituent count ``N`` is identical across
    views — a requirement for stacking the per-view embedding tensors.
    Pre-flatten augmentations that modify the valid mask (e.g.
    :class:`~fm4tag.augmentations.TrackDropout`) therefore have no effect on
    the training loss; they are visible only in :meth:`predict_step` output.

    Args:
        encoders:   :class:`~torch.nn.ModuleDict` mapping object name → encoder.
        aggregator: :class:`TransformerAggregator` mapping per-object projections to a
                    single jet embedding ``z_jet`` (POINT B).  Shared (same
                    weights) with fine-tuning.  Receives no gradient unless the
                    loss contains a jet-level term.
        views:      List of :class:`~fm4tag.augmentations.Compose` pipelines,
                    one per view.  The first view is used as the denoising
                    input; the original batch data is always the reconstruction
                    target.
        loss:       :class:`PretrainLoss` — composable, weighted sum of loss
                    terms.
        cfg:        Full Hydra config (used for eval / optimiser settings).
    """

    def __init__(
        self,
        encoders: torch.nn.ModuleDict,
        aggregator: TransformerAggregator,
        views: list[Compose],
        loss: PretrainLoss,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        # Ignore nn.Module constructor args (already saved via state_dict) so
        # they are not duplicated into the checkpoint's hyper_parameters.
        self.save_hyperparameters(ignore=['encoders', 'aggregator', 'views', 'loss'])

        if len(views) < 2:
            raise ValueError('ContrastiveDenoisingModule requires at least 2 views.')

        self.encoders = encoders
        self.aggregator = aggregator
        self.views = nn.ModuleList(views)
        self.loss = loss

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
    # Per-object loss computation
    # ------------------------------------------------------------------

    def _compute_loss_for_global(
        self,
        batch: dict,
        encoder: GlobalEncoder,
        needs_denoise: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]:
        """Per-object loss for the global object (POINT A).

        Returns ``(obj_loss, obj_logs, z_views)`` where ``z_views`` is the list
        of per-view ``(B, d_global)`` projections, reused for jet aggregation.
        """
        x_orig = batch['global']  # (B, F_g)

        zs: list[torch.Tensor] = []
        X_first: torch.Tensor | None = None
        for i, view in enumerate(self.views):
            z, X = encode_global_view(encoder, view, x_orig)
            zs.append(z)
            if i == 0:
                X_first = X

        kwargs: dict = {'z_list': zs}
        if needs_denoise:
            assert X_first is not None
            con_outs = encoder.reconstructor(X_first)  # list of (B, 1)
            # The global object has no categorical features.
            x_categ_empty = x_orig.new_zeros((x_orig.shape[0], 0), dtype=torch.long)
            kwargs.update(
                cat_outs=[], x_categ=x_categ_empty, con_outs=con_outs, x_cont=x_orig
            )

        obj_loss, obj_logs = self.loss(**kwargs)
        return obj_loss, obj_logs, zs

    def _compute_loss_for_constituent(
        self,
        batch: dict,
        obj_name: str,
        encoder: Encoder,
        needs_denoise: bool,
        needs_jet: bool,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        list[torch.Tensor] | None,
        torch.Tensor,
    ]:
        """Per-object loss for one constituent type (POINT A).

        Returns ``(obj_loss, obj_logs, z_views, valids)`` where ``z_views`` —
        present only when ``needs_jet`` — is the list of per-view
        ``(B, C, d_i)`` projections scattered back onto the padded grid (for
        jet aggregation), and ``valids`` is the ``(B, C)`` mask.
        """
        const = batch['constituents'][obj_name]
        valids = const['valid']  # (B, C)

        # Original valid mask — shared across all views.
        valids_flat = rearrange(valids, 'b c -> (b c)')
        x_categ_orig = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_flat]
        x_cont_orig = rearrange(const['continuous'], 'b c f -> (b c) f')[valids_flat]

        zs: list[torch.Tensor] = []
        X_first: torch.Tensor | None = None
        for i, view in enumerate(self.views):
            z, X = encode_constituent_view(encoder, view, const, valids_flat)
            zs.append(z)
            if i == 0:
                X_first = X

        kwargs: dict = {'z_list': zs}
        if needs_denoise:
            assert X_first is not None
            cat_outs = encoder.cat_reconstructor(
                X_first[:, : encoder.num_categories, :]
            )
            con_outs = encoder.con_reconstructor(
                X_first[:, encoder.num_categories :, :]
            )
            kwargs.update(
                cat_outs=cat_outs,
                x_categ=x_categ_orig,
                con_outs=con_outs,
                x_cont=x_cont_orig,
            )

        obj_loss, obj_logs = self.loss(**kwargs)

        # Scatter per-view projections back onto the padded grid for the
        # aggregator (POINT B) — only when a jet-level term needs them.
        z_views: list[torch.Tensor] | None = None
        if needs_jet:
            B, C = valids.shape
            z_views = [scatter_valid(z, valids_flat, B, C) for z in zs]

        return obj_loss, obj_logs, z_views, valids

    # ------------------------------------------------------------------
    # Total loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        needs_denoise = loss_wants(self.loss, 'con_outs')
        needs_jet = loss_wants(self.loss, 'z_jet_list')

        total_loss = batch['global'].new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        z_global_views: list[torch.Tensor] | None = None
        z_consts_per_obj: list[list[torch.Tensor]] = []  # [obj][view] → (B,C,d_i)
        valids_per_obj: list[torch.Tensor] = []

        for obj_name, encoder in self.encoders.items():
            if obj_name == self.global_object:
                obj_loss, obj_logs, z_global_views = self._compute_loss_for_global(
                    batch, encoder, needs_denoise
                )
            else:
                obj_loss, obj_logs, z_views, valids = (
                    self._compute_loss_for_constituent(
                        batch, obj_name, encoder, needs_denoise, needs_jet
                    )
                )
                if needs_jet:
                    assert z_views is not None
                    z_consts_per_obj.append(z_views)
                    valids_per_obj.append(valids)

            total_loss = total_loss + obj_loss
            for k, v in obj_logs.items():
                log_dict[f'{obj_name}/{k}'] = v

        # Jet-level (POINT B) loss — aggregated across all objects, once.
        if needs_jet:
            assert z_global_views is not None
            z_jet_list: list[torch.Tensor] = []
            for v in range(len(self.views)):
                z_consts_v = [z_views[v] for z_views in z_consts_per_obj]
                z_jet = self.aggregator(z_global_views[v], z_consts_v, valids_per_obj)
                z_jet_list.append(z_jet)

            jet_loss, jet_logs = self.loss(z_jet_list=z_jet_list)
            total_loss = total_loss + jet_loss
            for k, v in jet_logs.items():
                if k == 'loss':
                    continue
                log_dict[k] = v  # top-level, e.g. 'jet_embedding/loss_contrastive'

        log_dict['loss'] = total_loss
        return total_loss, log_dict

    # ------------------------------------------------------------------
    # Eval projection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _project_for_eval(self, batch: dict, obj_name: str) -> torch.Tensor | None:
        """Project clean (no augmentation) embeddings for eval metrics."""
        encoder = self.encoders[obj_name]

        if obj_name == self.global_object:
            X = encoder(batch['global'])  # (B, F_g, dim)
            return encoder.projector(X.flatten(1))  # (B, proj_dim)

        const = batch['constituents'][obj_name]
        valids_flat = rearrange(const['valid'], 'b c -> (b c)')
        if valids_flat.sum() == 0:
            return None

        x_categ = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_flat]
        x_cont = rearrange(const['continuous'], 'b c f -> (b c) f')[valids_flat]
        x_cat_enc, x_con_enc = embed_data(x_categ, x_cont, encoder)
        X = encoder(x_cat_enc, x_con_enc)  # (N, F, dim)
        return encoder.projector(X.flatten(1, 2))  # (N, proj_dim)

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

    def _log_metrics(self, log_dict: dict[str, torch.Tensor], split: str) -> None:
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
                on_step=on_step,
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
        all_subkeys = sorted(
            {
                k.split('/', 1)[1]
                for k in avgs
                if '/' in k and k.split('/', 1)[1] != 'loss'
            }
        )

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

    # ------------------------------------------------------------------
    # Predict step — returns per-view augmented data for visualisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        """Return per-view augmented data at the pre-flatten and raw stages.

        Useful for plotting feature distributions after each augmentation
        pipeline.  Unlike the training loss, here each view uses its **own**
        valid mask (so :class:`~fm4tag.augmentations.TrackDropout` effects
        are visible in the output).

        Returns a dict structured as::

            {
                '<global_obj>': {
                    'original': Tensor (B, F_g),
                    'views': [{'raw': Tensor (B, F_g)}, ...],
                },
                'constituents': {
                    '<obj>': {
                        'original': {
                            'categorical': Tensor (N, F_cat),
                            'continuous':  Tensor (N, F_con),
                        },
                        'views': [
                            {
                                'pre_flatten': {
                                    'categorical': Tensor (N_v, F_cat),
                                    'continuous':  Tensor (N_v, F_con),
                                    'valid':        Tensor (B, C),
                                },
                                'raw': {
                                    'categorical': Tensor (N_v, F_cat),
                                    'continuous':  Tensor (N_v, F_con),
                                },
                            },
                            ...
                        ],
                    },
                },
            }
        """
        result: dict = {}

        # ── Global object ────────────────────────────────────────────────────
        global_name = self.global_object
        x_global = batch['global']
        view_results = []
        for view in self.views:
            raw_out = view.apply_raw({'continuous': x_global.clone()})
            view_results.append({'raw': raw_out['continuous'].cpu()})
        result[global_name] = {
            'original': x_global.cpu(),
            'views': view_results,
        }

        # ── Constituent objects ───────────────────────────────────────────────
        result['constituents'] = {}
        for obj_name in self.constituent_objects:
            const = batch['constituents'][obj_name]

            # Original flattened using the unaugmented valid mask.
            valids_orig = rearrange(const['valid'], 'b c -> (b c)')
            x_categ_orig = rearrange(const['categorical'], 'b c f -> (b c) f')[
                valids_orig
            ]
            x_cont_orig = rearrange(const['continuous'], 'b c f -> (b c) f')[
                valids_orig
            ]

            view_results = []
            for view in self.views:
                # PRE_FLATTEN — use the view's own valid mask here.
                data_pre = view.apply_pre_flatten(
                    {
                        'categorical': const['categorical'].clone(),
                        'continuous': const['continuous'].clone(),
                        'valid': const['valid'].clone(),
                    }
                )
                valids_v = rearrange(data_pre['valid'], 'b c -> (b c)')
                x_cat_v = rearrange(data_pre['categorical'], 'b c f -> (b c) f')[
                    valids_v
                ]
                x_con_v = rearrange(data_pre['continuous'], 'b c f -> (b c) f')[
                    valids_v
                ]

                # RAW stage
                data_raw = view.apply_raw(
                    {'categorical': x_cat_v, 'continuous': x_con_v}
                )

                view_results.append(
                    {
                        'pre_flatten': {
                            'categorical': x_cat_v.cpu(),
                            'continuous': x_con_v.cpu(),
                            'valid': data_pre['valid'].cpu(),
                        },
                        'raw': {
                            'categorical': data_raw['categorical'].cpu(),
                            'continuous': data_raw['continuous'].cpu(),
                        },
                    }
                )

            result['constituents'][obj_name] = {
                'original': {
                    'categorical': x_categ_orig.cpu(),
                    'continuous': x_cont_orig.cpu(),
                },
                'views': view_results,
            }

        return result
