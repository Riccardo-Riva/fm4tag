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
        chunk_size: null
      - type: col
        depth: 1
        heads: 4
        dim_head: 32
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from ..attention import Attention, ChunkedRowAttention, RowAttention
from ..blocks import FeedForward, PreNorm, Residual


def _build_row_attention(
    row_dim: int,
    *,
    heads: int,
    dim_row_head: int,
    dropout: float,
    chunk_size: int | None,
) -> nn.Module:
    """Select the intersample-attention variant at construction time.

    ``chunk_size=None`` → whole-batch :class:`RowAttention`; otherwise chunked
    :class:`ChunkedRowAttention`.  Keeping this decision here means neither
    module needs a structural branch in its ``forward``.
    """
    if chunk_size is None:
        return RowAttention(
            row_dim, heads=heads, dim_row_head=dim_row_head, dropout=dropout
        )
    return ChunkedRowAttention(
        row_dim,
        heads=heads,
        dim_row_head=dim_row_head,
        dropout=dropout,
        chunk_size=chunk_size,
    )


class ColTransformer(nn.Module):
    """Column-attention (within-sample) transformer with configurable depth.

    Each depth step is ``PreNorm(Residual(Attention)) + PreNorm(Residual(FF))``.

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
                        PreNorm(
                            dim,
                            Residual(
                                Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=attn_dropout,
                                )
                            ),
                        ),
                        PreNorm(
                            dim,
                            Residual(
                                FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
                            ),
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

    Flattens the token dimension to ``(B, N*dim)`` for intersample attention
    then unflattens.  Each depth step is one row-attn+FFN pair.

    Args:
        dim:          Embedding dimension per token.
        nfeats:       Number of tokens per sample (``N_cat + N_con``).
                      Determines ``row_dim = dim * nfeats``.  Injected
                      automatically by :class:`~fm4tag.models.Encoder`.
        depth:        Number of row-attn+FFN steps to stack.
        row_heads:    Number of attention heads for row attention.
        dim_row_head: Head dimension for row attention.
        ff_mult:      Feed-forward hidden-size multiplier.
        attn_dropout: Dropout rate inside attention.
        ff_dropout:   Dropout rate inside feed-forward.
        chunk_size:   If set, splits the batch into disjoint groups for
                      attention (:class:`~fm4tag.models.attention.ChunkedRowAttention`);
                      ``None`` uses whole-batch
                      :class:`~fm4tag.models.attention.RowAttention`.
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
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        row_dim = dim * nfeats
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            row_dim,
                            Residual(
                                _build_row_attention(
                                    row_dim,
                                    heads=row_heads,
                                    dim_row_head=dim_row_head,
                                    dropout=attn_dropout,
                                    chunk_size=chunk_size,
                                )
                            ),
                        ),
                        PreNorm(
                            row_dim,
                            Residual(
                                FeedForward(row_dim, mult=ff_mult, dropout=ff_dropout)
                            ),
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, n, _ = x.shape
        x = rearrange(x, 'b n d -> b (n d)')
        for attn, ff in self.blocks:
            x = attn(x, mask=mask)
            x = ff(x)
        return rearrange(x, 'b (n d) -> b n d', n=n)


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
        chunk_size:   Chunked row attention — see :class:`RowTransformer`.
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
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        row_dim = dim * nfeats
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Residual(
                                Attention(
                                    dim,
                                    heads=col_heads,
                                    dim_head=dim_head,
                                    dropout=attn_dropout,
                                )
                            ),
                        ),
                        PreNorm(
                            dim,
                            Residual(
                                FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
                            ),
                        ),
                        PreNorm(
                            row_dim,
                            Residual(
                                _build_row_attention(
                                    row_dim,
                                    heads=row_heads,
                                    dim_row_head=dim_row_head,
                                    dropout=attn_dropout,
                                    chunk_size=chunk_size,
                                )
                            ),
                        ),
                        PreNorm(
                            row_dim,
                            Residual(
                                FeedForward(row_dim, mult=ff_mult, dropout=ff_dropout)
                            ),
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, n, _ = x.shape
        # Column Attention expects a per-token ``(b, n)`` mask; row attention
        # keeps the per-sample ``(b,)`` mask.
        col_mask = mask[:, None].expand(-1, n) if mask is not None else None
        for col_attn, col_ff, row_attn, row_ff in self.blocks:
            x = col_attn(x, mask=col_mask)
            x = col_ff(x)
            x = rearrange(x, 'b n d -> b (n d)')
            x = row_attn(x, mask=mask)
            x = row_ff(x)
            x = rearrange(x, 'b (n d) -> b n d', n=n)
        return x
