"""Shared encoding helpers used by pretraining and fine-tuning.

Both the pretraining and the fine-tuning module apply an augmentation
:class:`~fm4tag.augmentations.Compose` pipeline at the RAW / EMBEDDING stages
(and, for constituents, the PRE_FLATTEN stage), then encode and project via
``encoder.projector``.  The logic lives here so the two modules share exactly
one implementation — finetune views therefore behave identically to pretrain
views.  :func:`encode_clean` is the shared augmentation-free forward used by
both modules' ``forward`` (the contract the
:class:`~fm4tag.callbacks.EmbeddingMetrics` callback consumes).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from einops import rearrange

from ..augmentations import Compose
from ..models import embed_data
from ..models.backbones import Encoder, GlobalEncoder


def encode_global_view(
    encoder: GlobalEncoder,
    view: Compose,
    x_orig: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one view's augmentations and encode global features.

    Returns:
        z: ``(B, proj_dim)`` projected embedding (POINT A).
        X: ``(B, F_g, dim)`` encoder output (for denoising).
    """
    # RAW stage
    data = view.apply_raw({'continuous': x_orig})
    x = data['continuous']

    # Encode
    X = encoder(x)  # (B, F_g, dim)

    # EMBEDDING stage
    data_emb = view.apply_embedding({'continuous': X})
    X_emb = data_emb['continuous']

    z = encoder.projector(X_emb.flatten(1))  # (B, proj_dim)
    return z, X_emb


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


def encode_clean(
    encoders: torch.nn.ModuleDict,
    aggregator: torch.nn.Module,
    batch: dict,
    global_object: str,
    constituent_objects: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Clean (no augmentation) embeddings for every object plus the aggregator.

    Shared ``forward`` implementation of the pretraining and fine-tuning
    modules — and thereby the contract consumed by the
    :class:`~fm4tag.callbacks.EmbeddingMetrics` callback.

    Returns:
        ``{'<obj>_embedding': (B or N_valid, d), ..., 'aggregator': (B, d)}``
    """
    encoder = encoders[global_object]
    X = encoder(batch['global'])  # (B, F_g, dim)
    z_global = encoder.projector(X.flatten(1))  # (B, proj_dim)
    out = {f'{global_object}_embedding': z_global}

    z_consts: list[torch.Tensor] = []
    valids_list: list[torch.Tensor] = []
    for obj_name in constituent_objects:
        encoder = encoders[obj_name]
        const = batch['constituents'][obj_name]
        valids = const['valid']
        valids_flat = rearrange(valids, 'b c -> (b c)')
        x_categ = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_flat]
        x_cont = rearrange(const['continuous'], 'b c f -> (b c) f')[valids_flat]
        x_cat_enc, x_con_enc = embed_data(x_categ, x_cont, encoder)
        X = encoder(x_cat_enc, x_con_enc)  # (N_valid, F, dim)
        z = encoder.projector(X.flatten(1, 2))  # (N_valid, proj_dim)
        out[f'{obj_name}_embedding'] = z

        B, C = valids.shape
        z_consts.append(scatter_valid(z, valids_flat, B, C))
        valids_list.append(valids)

    out['aggregator'] = aggregator(z_global, z_consts, valids_list)
    return out


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
