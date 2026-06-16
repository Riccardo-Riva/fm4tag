"""Jet-level aggregator: cross-constituent attention + pooling into one vector.

The aggregator turns the per-object ``projector`` projections (POINT A) into a
single jet embedding ``z_jet`` (POINT B in the pipeline):

    z_global  (B, d_global)                ─┐
    z_consts  [(B, C, d_i), ...]           ─┤→ JetAggregator → z_jet (B, out_dim)
    valids    [(B, C), ...]                 ─┘

For each constituent type the projected embeddings are refined by a
:class:`~fm4tag.models.heads.transformers.Classifier_Transformer` (cross-
constituent attention), masked-mean-pooled over valid constituents, then
concatenated with the global projection.

This module is shared (same weights) between pretraining and fine-tuning.
The logic was previously inlined in ``MultiStreamClassifierHead.forward``.
"""

from __future__ import annotations

import torch
from torch import nn

from .heads.transformers import Classifier_Transformer


class JetAggregator(nn.Module):
    """Aggregate per-object projections into a single jet embedding.

    For each constituent type ``i``:

    1. ``(B, C, d_i)`` projected embeddings → cross-constituent transformer
       → ``(B, C, d_i)``
    2. masked mean pool over valid constituents → ``(B, d_i)``

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
            z_consts: List of ``(B, C, d_i)`` tensors — output of
                      ``encoder.projector`` scattered back over the padded
                      constituent grid, zeros at invalid slots.
            valids:   List of ``(B, C)`` bool masks.

        Returns:
            ``(B, self.out_dim)`` jet embedding (POINT B).
        """
        reprs = [z_global]

        for i, (z, valid) in enumerate(zip(z_consts, valids)):
            # Zero out padding slots before attention.
            z = torch.where(valid.unsqueeze(-1), z, 0.0)

            # Cross-constituent transformer.
            z = self.const_transformer[i](z, valid)

            # Masked mean pool over valid constituents → (B, d_i)
            n_valids = valid.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            z = z.sum(dim=1) / n_valids

            reprs.append(z)

        return torch.cat(reprs, dim=-1)
