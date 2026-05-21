"""Contrastive + denoising pretraining module with multi-view augmentations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import nn

from ..augmentations import Compose
from ..losses import DenoisingLoss, MultiViewSupConLoss
from ..models import embed_data
from ..models.backbones import Encoder, GlobalEncoder
from .base_pretrain_module import BasePretrainModule


class ContrastiveDenoisingModule(BasePretrainModule):
    """Multi-view contrastive + denoising pretraining module.

    For each object (global + constituents) and each batch:

    1. **Contrastive** — all ``V`` views are encoded and projected; a
       :class:`~fm4tag.losses.MultiViewSupConLoss` treats all views of the
       same sample as positives and all views of other samples as negatives.

    2. **Denoising** — the *first* view's encoded representation reconstructs
       the original (pre-augmentation) features via ``sep``-style MLP heads;
       supervised with cross-entropy (categorical) + MSE (continuous).

    Valid-mask consistency
    ----------------------
    All views share the **original** ``valid`` mask when flattening
    constituent tensors, so the constituent count ``N`` is identical across
    views — a requirement for stacking the per-view embedding tensors.
    Pre-flatten augmentations that modify the valid mask (e.g.
    :class:`~fm4tag.augmentations.TrackDropout`) therefore have no effect on
    the training loss; they are visible only in :meth:`predict_step` output.

    Args:
        encoders: :class:`~torch.nn.ModuleDict` mapping object name → encoder.
        views:    List of :class:`~fm4tag.augmentations.Compose` pipelines, one
                  per view.  Instantiated externally (e.g. via
                  ``hydra.utils.instantiate``) and passed in.  The first view
                  is used as the denoising input; the original batch data is
                  always the reconstruction target.
        cfg:      Full Hydra config.  Relevant sub-keys:

                  ``cfg.pretrain.nce_temp``           — contrastive temperature
                  ``cfg.pretrain.loss_type``           — ``'out'`` or ``'in'``
                  ``cfg.pretrain.include_pos_in_denom``— bool
                  ``cfg.pretrain.lam_contrastive``     — contrastive loss weight
                  ``cfg.pretrain.lam_denoising_cat``   — categorical denoising weight
                  ``cfg.pretrain.lam_denoising_con``   — continuous denoising weight
    """

    def __init__(
        self,
        encoders: torch.nn.ModuleDict,
        views: list[Compose],
        cfg: DictConfig,
    ) -> None:
        super().__init__(encoders, cfg)

        if len(views) < 2:
            raise ValueError(
                "ContrastiveDenoisingModule requires at least 2 views."
            )
        self.views = nn.ModuleList(views)

        cfg_pt = cfg.pretrain
        self.contrastive_loss = MultiViewSupConLoss(
            temperature=cfg_pt.nce_temp,
            loss_type=cfg_pt.get('loss_type', 'out'),
            include_pos_in_denom=cfg_pt.get('include_pos_in_denom', True),
        )
        self.denoising_loss = DenoisingLoss()

    # ------------------------------------------------------------------
    # Per-view encoding helpers
    # ------------------------------------------------------------------

    def _encode_global_view(
        self,
        x_orig: torch.Tensor,
        view: Compose,
        encoder: GlobalEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one view's augmentations and encode global features.

        Returns:
            z: ``(B, proj_dim)`` projected embedding (for contrastive loss).
            X: ``(B, F_g, dim)`` transformer output (for denoising loss).
        """
        # RAW stage
        data = view.apply_raw({'continuous': x_orig})
        x = data['continuous']

        # Encode
        X = encoder(x)   # (B, F_g, dim)

        # EMBEDDING stage
        data_emb = view.apply_embedding({'continuous': X})
        X_emb = data_emb['continuous']

        z = encoder.projector(X_emb.flatten(1))   # (B, proj_dim)
        return z, X_emb

    def _encode_constituent_view(
        self,
        const: dict,
        valids_flat: torch.Tensor,
        view: Compose,
        encoder: Encoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one view's augmentations and encode constituent features.

        ``valids_flat`` is always the **original** valid mask (flattened to
        ``(B*C,)``), shared across all views to guarantee consistent N.
        Pre-flatten augmentations in ``view`` may modify feature values but
        their valid-mask output is ignored here.

        Returns:
            z: ``(N, proj_dim)`` projected embedding (for contrastive loss).
            X: ``(N, F, dim)`` transformer output (for denoising loss).
        """
        # PRE_FLATTEN stage — apply to get possibly modified features.
        # Valid mask from the aug output is intentionally discarded; we always
        # flatten with the original mask so N is consistent across views.
        data_pre = view.apply_pre_flatten({
            'categorical': const['categorical'],
            'continuous':  const['continuous'],
            'valid':        const['valid'],
        })

        x_categ = rearrange(data_pre['categorical'], 'b c f -> (b c) f')[valids_flat]
        x_cont  = rearrange(data_pre['continuous'],  'b c f -> (b c) f')[valids_flat]

        # RAW stage
        data_raw = view.apply_raw({'categorical': x_categ, 'continuous': x_cont})
        x_categ = data_raw['categorical']
        x_cont  = data_raw['continuous']

        # Embed
        x_cat_enc, x_con_enc = embed_data(x_categ, x_cont, encoder)

        # EMBEDDING stage
        data_emb = view.apply_embedding({
            'categorical': x_cat_enc, 'continuous': x_con_enc
        })
        x_cat_enc = data_emb['categorical']
        x_con_enc = data_emb['continuous']

        # Encode
        X = encoder(x_cat_enc, x_con_enc)   # (N, F, dim)
        z = encoder.projector(X.flatten(1, 2))   # (N, proj_dim)
        return z, X

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss_for_global(
        self,
        batch: dict,
        encoder: GlobalEncoder,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg_pt = self.cfg.pretrain
        x_orig = batch['global']   # (B, F_g)

        zs: list[torch.Tensor] = []
        X_first: torch.Tensor | None = None

        for i, view in enumerate(self.views):
            z, X = self._encode_global_view(x_orig, view, encoder)
            zs.append(z)
            if i == 0:
                X_first = X

        total_loss = x_orig.new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        # Contrastive
        l_cont = self.contrastive_loss(zs)
        log_dict['loss_contrastive'] = l_cont
        total_loss = total_loss + cfg_pt.lam_contrastive * l_cont

        # Denoising (first view → reconstruct original)
        lam_con: float = cfg_pt.get('lam_denoising_con', 0.0)
        if lam_con > 0.0:
            assert X_first is not None
            con_pred = torch.cat(encoder.reconstructor(X_first), dim=1)   # (B, F_g)
            l_den_con = F.mse_loss(con_pred, x_orig)
            log_dict['loss_denoising_con'] = l_den_con
            total_loss = total_loss + lam_con * l_den_con

        log_dict['loss'] = total_loss
        return total_loss, log_dict

    def _compute_loss_for_constituent(
        self,
        batch: dict,
        obj_name: str,
        encoder: Encoder,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg_pt = self.cfg.pretrain
        const = batch['constituents'][obj_name]

        # Original valid mask — shared across all views.
        valids_flat = rearrange(const['valid'], 'b c -> (b c)')
        x_categ_orig = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_flat]
        x_cont_orig  = rearrange(const['continuous'],  'b c f -> (b c) f')[valids_flat]

        zs: list[torch.Tensor] = []
        X_first: torch.Tensor | None = None

        for i, view in enumerate(self.views):
            z, X = self._encode_constituent_view(const, valids_flat, view, encoder)
            zs.append(z)
            if i == 0:
                X_first = X

        total_loss = x_categ_orig.new_zeros(())
        log_dict: dict[str, torch.Tensor] = {}

        # Contrastive
        l_cont = self.contrastive_loss(zs)
        log_dict['loss_contrastive'] = l_cont
        total_loss = total_loss + cfg_pt.lam_contrastive * l_cont

        # Denoising (first view → reconstruct original)
        assert X_first is not None
        lam_cat: float = cfg_pt.get('lam_denoising_cat', 0.0)
        lam_con: float = cfg_pt.get('lam_denoising_con', 0.0)

        if lam_cat > 0.0 or lam_con > 0.0:
            cat_outs = encoder.cat_reconstructor(X_first[:, :encoder.num_categories, :])
            con_outs = encoder.con_reconstructor(X_first[:, encoder.num_categories:, :])
            l_cat, l_con = self.denoising_loss(cat_outs, x_categ_orig, con_outs, x_cont_orig)

            if lam_cat > 0.0:
                log_dict['loss_denoising_cat'] = l_cat
                total_loss = total_loss + lam_cat * l_cat
            if lam_con > 0.0:
                log_dict['loss_denoising_con'] = l_con
                total_loss = total_loss + lam_con * l_con

        log_dict['loss'] = total_loss
        return total_loss, log_dict

    def _compute_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    # Eval projection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _project_for_eval(
        self, batch: dict, obj_name: str
    ) -> torch.Tensor | None:
        """Project clean (no augmentation) embeddings for eval metrics."""
        encoder = self.encoders[obj_name]

        if obj_name == self.global_object:
            X = encoder(batch['global'])            # (B, F_g, dim)
            return encoder.projector(X.flatten(1))  # (B, proj_dim)

        const = batch['constituents'][obj_name]
        valids_flat = rearrange(const['valid'], 'b c -> (b c)')
        if valids_flat.sum() == 0:
            return None

        x_categ = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_flat]
        x_cont  = rearrange(const['continuous'],  'b c f -> (b c) f')[valids_flat]
        x_cat_enc, x_con_enc = embed_data(x_categ, x_cont, encoder)
        X = encoder(x_cat_enc, x_con_enc)           # (N, F, dim)
        return encoder.projector(X.flatten(1, 2))   # (N, proj_dim)

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
            x_categ_orig = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_orig]
            x_cont_orig  = rearrange(const['continuous'],  'b c f -> (b c) f')[valids_orig]

            view_results = []
            for view in self.views:
                # PRE_FLATTEN — use the view's own valid mask here.
                data_pre = view.apply_pre_flatten({
                    'categorical': const['categorical'].clone(),
                    'continuous':  const['continuous'].clone(),
                    'valid':        const['valid'].clone(),
                })
                valids_v = rearrange(data_pre['valid'], 'b c -> (b c)')
                x_cat_v  = rearrange(data_pre['categorical'], 'b c f -> (b c) f')[valids_v]
                x_con_v  = rearrange(data_pre['continuous'],  'b c f -> (b c) f')[valids_v]

                # RAW stage
                data_raw = view.apply_raw({'categorical': x_cat_v, 'continuous': x_con_v})

                view_results.append({
                    'pre_flatten': {
                        'categorical': x_cat_v.cpu(),
                        'continuous':  x_con_v.cpu(),
                        'valid':        data_pre['valid'].cpu(),
                    },
                    'raw': {
                        'categorical': data_raw['categorical'].cpu(),
                        'continuous':  data_raw['continuous'].cpu(),
                    },
                })

            result['constituents'][obj_name] = {
                'original': {
                    'categorical': x_categ_orig.cpu(),
                    'continuous':  x_cont_orig.cpu(),
                },
                'views': view_results,
            }

        return result
