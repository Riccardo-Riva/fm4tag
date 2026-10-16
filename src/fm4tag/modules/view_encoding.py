"""Shared encoding helpers used by pretraining and fine-tuning.

Both the pretraining and the fine-tuning module apply an augmentation
:class:`~fm4tag.augmentations.Compose` pipeline at the RAW / EMBEDDING stages
(and, for constituents, the PRE_FLATTEN stage), then encode and project via
``encoder.projector``.  The logic lives here so the two modules share exactly
one implementation — finetune views therefore behave identically to pretrain
views.  :func:`encode_clean` is the shared augmentation-free forward used by
both modules' ``forward`` (the contract the
:class:`~fm4tag.callbacks.EmbeddingMetrics` callback consumes).

Jet chunks: with ``chunk_size = C`` the batch is cut into consecutive chunks
of ``C`` jets (the last may be shorter) and the intersample attention runs
within a chunk only — among the chunk's jets for the global encoder, among
all valid constituents of the chunk's jets for a constituent encoder.  The
packing below is jet-major, so a jet chunk is a contiguous run of packed
constituents; :func:`batch_row_groups` builds those boundaries once per batch.
"""

from __future__ import annotations

from collections.abc import Sequence

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from einops import rearrange

from ..augmentations import Compose
from ..models import embed_data
from ..models.attention import RowGroups
from ..models.backbones import Encoder, GlobalEncoder


def _jet_boundaries(n_jets: int, chunk_size: int, device) -> torch.Tensor:
    cu = torch.arange(0, n_jets, chunk_size, device=device)
    return torch.cat([cu, cu.new_tensor([n_jets])])


def jet_row_groups(n_jets: int, chunk_size: int | None, device) -> RowGroups | None:
    """Chunks of ``chunk_size`` consecutive jets; ``None`` = no chunking."""
    if chunk_size is None:
        return None
    cu = _jet_boundaries(n_jets, chunk_size, device).to(torch.int32)
    return RowGroups(cu, min(chunk_size, n_jets))


def constituent_row_groups(
    valids: torch.Tensor, chunk_size: int | None
) -> RowGroups | None:
    """The packed valid constituents of each jet chunk; ``None`` = no chunking.

    ``valids`` is the ``(B, C)`` mask.  Packing is jet-major, so chunk ``k``
    covers packed rows ``[offset[k*chunk], offset[(k+1)*chunk])``, where
    ``offset`` is the exclusive cumsum of the per-jet valid counts.
    """
    if chunk_size is None:
        return None
    offset = F.pad(valids.sum(1).cumsum(0), (1, 0))  # (B+1,)
    jet_cu = _jet_boundaries(valids.shape[0], chunk_size, valids.device)
    return RowGroups.from_boundaries(offset[jet_cu])


def batch_row_groups(
    batch: dict,
    global_object: str,
    constituent_objects: Sequence[str],
    chunk_size: int | None,
) -> dict[str, RowGroups | None]:
    """Row groups of every object of ``batch`` (one host sync per constituent)."""
    groups = {
        global_object: jet_row_groups(
            batch['global'].shape[0], chunk_size, batch['global'].device
        )
    }
    for obj_name in constituent_objects:
        valids = batch['constituents'][obj_name]['valid']
        groups[obj_name] = constituent_row_groups(valids, chunk_size)
    return groups


def encode_global_view(
    encoder: GlobalEncoder,
    view: Compose,
    x_orig: torch.Tensor,
    groups: RowGroups | None = None,
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
    X = encoder(x, groups=groups)  # (B, F_g, dim)

    # EMBEDDING stage — note: applied to the encoder OUTPUT here, whereas for
    # constituents it is applied to the embedded tokens BEFORE the transformer.
    data_emb = view.apply_embedding({'continuous': X})
    X_emb = data_emb['continuous']

    z = encoder.projector(X_emb.flatten(1))  # (B, proj_dim)
    return z, X_emb


def encode_constituent_view(
    encoder: Encoder,
    view: Compose,
    const: dict,
    valids_flat: torch.Tensor,
    groups: RowGroups | None = None,
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
    X = encoder(x_cat_enc, x_con_enc, groups=groups)  # (N, F, dim)
    z = encoder.projector(X.flatten(1, 2))  # (N, proj_dim)
    return z, X


def encode_clean(
    encoders: torch.nn.ModuleDict,
    aggregator: torch.nn.Module,
    batch: dict,
    global_object: str,
    constituent_objects: Sequence[str],
    groups: Mapping[str, RowGroups | None] | None = None,
) -> dict[str, torch.Tensor]:
    """Clean (no augmentation) embeddings for every object plus the aggregator.

    Shared ``forward`` implementation of the pretraining and fine-tuning
    modules — and thereby the contract consumed by the
    :class:`~fm4tag.callbacks.EmbeddingMetrics` callback.  ``groups`` (from
    :func:`batch_row_groups`) chunks the intersample attention; ``None`` = none.

    Returns:
        ``{'<obj>_embedding': (B or N_valid, d), ..., 'aggregator': (B, d)}``
    """
    groups = groups or {}
    encoder = encoders[global_object]
    X = encoder(batch['global'], groups=groups.get(global_object))  # (B, F_g, dim)
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
        X = encoder(
            x_cat_enc, x_con_enc, groups=groups.get(obj_name)
        )  # (N_valid, F, dim)
        z = encoder.projector(X.flatten(1, 2))  # (N_valid, proj_dim)
        out[f'{obj_name}_embedding'] = z

        z_consts.append(z)  # packed; the aggregator runs on valid slots only
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
