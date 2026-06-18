"""Shared per-view encoding helpers used by pretraining and fine-tuning.

Both the contrastive pretraining module and the fine-tuning module apply an
augmentation :class:`~fm4tag.augmentations.Compose` pipeline at the RAW /
EMBEDDING stages (and, for constituents, the PRE_FLATTEN stage), then encode
and project to POINT A (``encoder.projector``).  The logic lives here so the two
modules share exactly one implementation — finetune views therefore behave
identically to pretrain views.
"""

from __future__ import annotations

import torch
from einops import rearrange

from ..augmentations import Compose
from ..models import embed_data
from ..models.backbones import Encoder, GlobalEncoder


def encode_global_view(
    encoder: GlobalEncoder,
    view: Compose,
    g: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one view's augmentations and encode global features.

    Mirrors :func:`encode_constituent_view`: features are embedded first
    (:meth:`GlobalEncoder.embed`), EMBEDDING-stage augmentations (e.g. Mixup)
    are applied to the embedded tokens **before** the transformer, then the
    attention stack runs.  The global object has no valid mask (all jets valid),
    so there is no PRE_FLATTEN stage.

    Args:
        g: Global batch dict ``{'categorical': (B, F_gcat), 'continuous': (B, F_gcon)}``.

    Returns:
        z: ``(B, proj_dim)`` projected embedding (POINT A).
        X: ``(B, F_g, dim)`` encoder output (for denoising), token order ``[cat; con]``.
    """
    # RAW stage
    data_raw = view.apply_raw(
        {'categorical': g['categorical'], 'continuous': g['continuous']}
    )

    # Embed
    x_cat_enc, x_con_enc = encoder.embed(
        data_raw['categorical'], data_raw['continuous']
    )

    # EMBEDDING stage
    data_emb = view.apply_embedding(
        {'categorical': x_cat_enc, 'continuous': x_con_enc}
    )

    # Encode (attention)
    X = encoder(data_emb['categorical'], data_emb['continuous'])  # (B, F_g, dim)
    z = encoder.projector(X.flatten(1, 2))  # (B, proj_dim)
    return z, X


def encode_constituent_view(
    encoder: Encoder,
    view: Compose,
    const: dict,
    valids_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one view's augmentations and encode constituent features.

    ``valids_flat`` is always the **original** valid mask (flattened to
    ``(B*C,)``), shared across all views to guarantee a consistent ``N``.
    Pre-flatten augmentations in ``view`` may modify feature values but their
    valid-mask output is intentionally discarded.

    Returns:
        z: ``(N, proj_dim)`` projected embedding (POINT A).
        X: ``(N, F, dim)`` encoder output (for denoising).
    """
    # PRE_FLATTEN stage — apply to get possibly modified features.
    data_pre = view.apply_pre_flatten(
        {
            'categorical': const['categorical'],
            'continuous': const['continuous'],
            'valid': const['valid'],
        }
    )

    x_categ = rearrange(data_pre['categorical'], 'b c f -> (b c) f')[valids_flat]
    x_cont = rearrange(data_pre['continuous'], 'b c f -> (b c) f')[valids_flat]

    # RAW stage
    data_raw = view.apply_raw({'categorical': x_categ, 'continuous': x_cont})
    x_categ = data_raw['categorical']
    x_cont = data_raw['continuous']

    # Embed
    x_cat_enc, x_con_enc = embed_data(x_categ, x_cont, encoder)

    # EMBEDDING stage
    data_emb = view.apply_embedding({'categorical': x_cat_enc, 'continuous': x_con_enc})
    x_cat_enc = data_emb['categorical']
    x_con_enc = data_emb['continuous']

    # Encode
    X = encoder(x_cat_enc, x_con_enc)  # (N, F, dim)
    z = encoder.projector(X.flatten(1, 2))  # (N, proj_dim)
    return z, X


def scatter_valid(
    z: torch.Tensor,
    valids_flat: torch.Tensor,
    B: int,
    C: int,
) -> torch.Tensor:
    """Scatter flat valid projections ``(N_valid, d)`` back onto ``(B, C, d)``.

    Invalid (padding) slots are filled with zeros.  ``z`` must be in the same
    order as ``valids_flat``'s ``True`` entries (i.e. produced by indexing the
    flattened grid with ``valids_flat``).
    """
    d = z.shape[1]
    z_all = z.new_zeros(B * C, d)
    z_all[valids_flat] = z
    return z_all.reshape(B, C, d)
