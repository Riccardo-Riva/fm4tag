"""Jet-level aggregator: cross-constituent attention + pooling into one vector.

The aggregator turns the per-object ``projector`` projections (POINT A) into a
single jet embedding ``z_jet`` (POINT B in the pipeline):

    z_global  (B, d_global)                  ─┐
    z_consts  [(N_valid, d_i) or (B, C, d_i)] ─┤→ TransformerAggregator → z_jet (B, out_dim)
    valids    [(B, C), ...]                   ─┘

For each constituent type the projected embeddings are refined by a
:class:`~fm4tag.models.heads.transformers.Classifier_Transformer` (cross-
constituent attention within each jet), mean-pooled over the jet's valid
constituents, then concatenated with the global projection.  It runs on the
valid constituents only (packed, one varlen attention group per jet), never
on the padded ``(B, C)`` grid, which is ~80% padding.

This module is shared (same weights) between pretraining and fine-tuning.
The logic was previously inlined in ``MultiStreamClassifierHead.forward``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .attention import RowGroups
from .heads.transformers import Classifier_Transformer


class TransformerAggregator(nn.Module):
    """Aggregate per-object projections into a single jet embedding.

    For each constituent type ``i``:

    1. the jet's valid projected embeddings → cross-constituent transformer
       (attention within the jet only)
    2. mean pool over the jet's valid constituents → ``(B, d_i)``; a jet with
       none pools to a zero vector

    The pooled constituent vectors are concatenated with the global projection
    ``z_global`` to form ``z_jet`` of dimension
    ``global_dim + sum(const_dims)`` (exposed as :attr:`out_dim`).

    Args:
        global_dim:   Output dimension of the global encoder's ``projector``.
        const_dims:   Output dimension of each constituent encoder's
                      ``projector``, one per constituent type and in the same
                      order as ``cfg.constituent_objects``.
        depth:        Cross-constituent transformer depth per stream.
        heads:        Number of attention heads per stream.
        dim_head:     Dimension per attention head.
        ff_mult:      Feed-forward expansion factor in the transformer.
        ff_dropout:   Dropout inside transformer feed-forward sub-layers.
        attn_dropout: Dropout inside transformer attention.
    """

    def __init__(
        self,
        global_dim: int,
        const_dims: list[int],
        depth: int = 3,
        heads: int = 8,
        dim_head: int = 16,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Cross-constituent transformer per stream, sized to that stream's dim.
        self.const_transformer = nn.ModuleList(
            [
                Classifier_Transformer(
                    out_dim,
                    depth=depth,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                )
                for out_dim in const_dims
            ]
        )

        self.out_dim = global_dim + sum(const_dims)

    def forward(
        self,
        z_global: torch.Tensor,
        z_consts: list[torch.Tensor],
        valids: list[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate projections into a single jet embedding.

        Args:
            z_global: ``(B, d_global)`` — output of ``global_encoder.projector``.
            z_consts: Per constituent type, the ``(N_valid, d_i)`` projections
                      of the valid constituents in jet-major packed order (as
                      produced by indexing the flattened grid with the valid
                      mask), or the padded ``(B, C, d_i)`` grid.
            valids:   List of ``(B, C)`` bool masks.

        Returns:
            ``(B, self.out_dim)`` jet embedding (POINT B).
        """
        reprs = [z_global]
        n_jets = z_global.shape[0]

        for i, (z, valid) in enumerate(zip(z_consts, valids)):
            if z.dim() == 3:  # padded grid → packed valid constituents
                z = z[valid]
            counts = valid.sum(1)
            # One attention group per jet; the slot count C bounds the group
            # length, so no host sync is needed for max_seqlen.
            offsets = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
            z = self.const_transformer[i](z, groups=RowGroups(offsets, valid.shape[1]))

            # Mean over each jet's valid constituents, accumulated in fp32.
            jet_ids = torch.repeat_interleave(
                torch.arange(n_jets, device=z.device), counts, output_size=z.shape[0]
            )
            pooled = z.new_zeros(n_jets, z.shape[1], dtype=torch.float32)
            pooled.index_add_(0, jet_ids, z.float())
            reprs.append(pooled / counts.clamp(min=1).unsqueeze(-1))

        return torch.cat(reprs, dim=-1)
