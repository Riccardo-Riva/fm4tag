"""Composable transformer layer types used by :class:`~fm4tag.models.Encoder`.

Each class implements one block in the transformer stack and is selected by the
``type`` key in the ``backbone.constituents.<name>.layers`` list — no
``_target_`` needed.  :class:`~fm4tag.models.Encoder` instantiates the correct
class and injects ``dim`` and ``nfeats`` automatically.

The ``depth`` parameter stacks that many attention+FFN sub-steps inside a
single class instance.  Different types can be freely mixed in the ``layers``
list, each with its own ``depth``.

All classes share the forward signature::

    forward(x: Tensor, mask: Tensor | None = None) -> Tensor
    x:   (B, N, dim)
    out: (B, N, dim)

Example ``backbone.constituents.tracks.layers`` config::

    layers:
      - type: rowcol
        depth: 3
        col_heads: 2
        row_heads: 8
        dim_head: 32
        dim_row_head: 32
        ff_mult: 1
        attn_dropout: 0.0
        ff_dropout: 0.0
        num_inds: 16
      - type: col
        depth: 1
        heads: 4
        dim_head: 32
"""

from __future__ import annotations

import torch
from torch import nn

from ..attention import Attention, InducedRowAttention, RowAttention
from ..blocks import FeedForward, PreNormResidual


def _build_row_attention(
    dim: int,
    nfeats: int,
    *,
    heads: int,
    dim_row_head: int,
    dropout: float,
    num_inds: int | None,
) -> nn.Module:
    """Select the intersample-attention variant at construction time.

    ``num_inds=None`` → all-pairs :class:`RowAttention` (O(B²) in the batch
    size); otherwise :class:`InducedRowAttention` with that many inducing points
    (O(B·num_inds)).  Keeping this decision here means neither module needs a
    structural branch in its ``forward``.
    """
    if num_inds is None:
        return RowAttention(
            dim, heads=heads, dim_row_head=dim_row_head, dropout=dropout
        )
    return InducedRowAttention(
        dim,
        nfeats,
        heads=heads,
        dim_row_head=dim_row_head,
        dropout=dropout,
        num_inds=num_inds,
    )


class RowMixer(nn.Module):
    """One depth step of intersample (row) attention + feed-forward.

    Operates on the token grid ``(B, N, dim)`` directly: row attention runs over
    the batch axis independently for each of the ``N`` tokens, at width ``dim``,
    with the projections shared across tokens.  This is the tabicl
    ``ColEmbedding`` factorisation — each feature builds its own summary across
    samples, and cross-feature mixing is left to the column attention.

    Args:
        dim:      Token embedding width — also the width of the row step.
        nfeats:   Number of tokens ``N`` (sizes the per-token inducing points).
        num_inds: Inducing points for :class:`InducedRowAttention`; ``None``
                  selects all-pairs :class:`RowAttention` instead.
    """

    def __init__(
        self,
        dim: int,
        nfeats: int,
        *,
        heads: int,
        dim_row_head: int,
        ff_mult: int,
        attn_dropout: float,
        ff_dropout: float,
        num_inds: int | None,
    ) -> None:
        super().__init__()
        self.attn = PreNormResidual(
            dim,
            _build_row_attention(
                dim,
                nfeats,
                heads=heads,
                dim_row_head=dim_row_head,
                dropout=attn_dropout,
                num_inds=num_inds,
            ),
        )
        self.ff = PreNormResidual(
            dim, FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.attn(x, mask=mask)
        return self.ff(x)


class ColTransformer(nn.Module):
    """Column-attention (within-sample) transformer with configurable depth.

    Each depth step is ``PreNormResidual(Attention) + PreNormResidual(FF)``.

    Args:
        dim:          Embedding dimension.
        depth:        Number of attention+FFN steps to stack.
        heads:        Number of attention heads.
        dim_head:     Dimension per head.
        ff_mult:      Feed-forward hidden-size multiplier.
        attn_dropout: Dropout rate inside attention.
        ff_dropout:   Dropout rate inside feed-forward.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 16,
        ff_mult: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNormResidual(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNormResidual(
                            dim, FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # The incoming mask is per-sample ``(b,)``; column Attention expects a
        # per-token ``(b, n)`` key-padding mask, so broadcast it across tokens.
        col_mask = mask[:, None].expand(-1, x.size(1)) if mask is not None else None
        for attn, ff in self.blocks:
            x = attn(x, mask=col_mask)
            x = ff(x)
        return x


class RowTransformer(nn.Module):
    """Row-attention (intersample) transformer with configurable depth.

    Each depth step is one row-attn+FFN pair, run over the batch axis at token
    width ``dim`` (see :class:`RowMixer`).

    Args:
        dim:          Embedding dimension per token.
        nfeats:       Number of tokens per sample (``N_cat + N_con``).  Injected
                      automatically by :class:`~fm4tag.models.Encoder`.
        depth:        Number of row-attn+FFN steps to stack.
        row_heads:    Number of attention heads for row attention.
        dim_row_head: Head dimension for row attention.
        ff_mult:      Feed-forward hidden-size multiplier.
        attn_dropout: Dropout rate inside attention.
        ff_dropout:   Dropout rate inside feed-forward.
        num_inds:     Number of inducing points
                      (:class:`~fm4tag.models.attention.InducedRowAttention`,
                      O(B·num_inds)); ``None`` uses all-pairs
                      :class:`~fm4tag.models.attention.RowAttention` (O(B²)).
    """

    def __init__(
        self,
        dim: int,
        nfeats: int,
        depth: int = 1,
        row_heads: int = 8,
        dim_row_head: int = 64,
        ff_mult: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        num_inds: int | None = 16,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RowMixer(
                    dim,
                    nfeats,
                    heads=row_heads,
                    dim_row_head=dim_row_head,
                    ff_mult=ff_mult,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    num_inds=num_inds,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for mixer in self.blocks:
            x = mixer(x, mask=mask)
        return x


class RowColTransformer(nn.Module):
    """Alternating col-then-row transformer with configurable depth.

    Each depth step applies column attention+FFN (within-sample) then row
    attention+FFN (intersample).

    Args:
        dim:          Embedding dimension.
        nfeats:       Number of tokens per sample.  Injected automatically by
                      :class:`~fm4tag.models.Encoder`.
        depth:        Number of col+row steps to stack.
        col_heads:    Number of attention heads for column attention.
        row_heads:    Number of attention heads for row attention.
        dim_head:     Head dimension for column attention.
        dim_row_head: Head dimension for row attention.
        ff_mult:      Feed-forward hidden-size multiplier (shared).
        attn_dropout: Dropout inside both attention sub-layers.
        ff_dropout:   Dropout inside both feed-forward sub-layers.
        num_inds:     Inducing points for the row step — see
                      :class:`RowTransformer`.
    """

    def __init__(
        self,
        dim: int,
        nfeats: int,
        depth: int = 1,
        col_heads: int = 8,
        row_heads: int = 8,
        dim_head: int = 16,
        dim_row_head: int = 64,
        ff_mult: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        num_inds: int | None = 16,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNormResidual(
                            dim,
                            Attention(
                                dim,
                                heads=col_heads,
                                dim_head=dim_head,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNormResidual(
                            dim, FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
                        ),
                        RowMixer(
                            dim,
                            nfeats,
                            heads=row_heads,
                            dim_row_head=dim_row_head,
                            ff_mult=ff_mult,
                            attn_dropout=attn_dropout,
                            ff_dropout=ff_dropout,
                            num_inds=num_inds,
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Column Attention expects a per-token ``(b, n)`` mask; row attention
        # keeps the per-sample ``(b,)`` mask.
        col_mask = mask[:, None].expand(-1, x.size(1)) if mask is not None else None
        for col_attn, col_ff, row_mixer in self.blocks:
            x = col_attn(x, mask=col_mask)
            x = col_ff(x)
            x = row_mixer(x, mask=mask)
        return x
