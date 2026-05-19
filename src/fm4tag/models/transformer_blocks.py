"""Composable transformer block types for fm4tag.

Each class is one depth-step in a transformer stack — the smallest swappable
unit, configured via ``_target_`` in YAML, mirroring the augmentation system.

All blocks share the same forward signature::

    forward(x: Tensor, mask: Tensor | None = None) -> Tensor
    x:   (B, N, dim)
    out: (B, N, dim)

Example YAML config::

    transformer_layers:
      - _target_: fm4tag.models.ColBlock
        dim: 64
        heads: 2
        dim_head: 32
        ff_mult: 1
        attn_dropout: 0.0
        ff_dropout: 0.0
      - _target_: fm4tag.models.RowColBlock
        dim: 64
        nfeats: 19          # len(cat_features) + len(con_features)
        col_heads: 2
        row_heads: 8
        dim_head: 32
        dim_row_head: 32
        ff_mult: 1
        attn_dropout: 0.0
        ff_dropout: 0.0
        chunk_size: null
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from .attention import Attention, RowAttention
from .blocks import FeedForward, PreNorm, Residual


class ColBlock(nn.Module):
    """One column-attention (within-sample) transformer layer.

    ``PreNorm(Residual(Attention))`` followed by ``PreNorm(Residual(FeedForward))``.

    Args:
        dim:          Embedding dimension.
        heads:        Number of attention heads.
        dim_head:     Dimension per head.
        ff_mult:      Feed-forward hidden-size multiplier.
        attn_dropout: Dropout rate inside attention.
        ff_dropout:   Dropout rate inside feed-forward.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 16,
        ff_mult: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = PreNorm(
            dim,
            Residual(
                Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)
            ),
        )
        self.ff = PreNorm(
            dim, Residual(FeedForward(dim, mult=ff_mult, dropout=ff_dropout))
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.attn(x, mask=mask)
        x = self.ff(x)
        return x


class RowBlock(nn.Module):
    """One row-attention (intersample) transformer layer.

    Flattens the token dimension to ``(B, N*dim)`` for intersample attention,
    then unflattens.

    Args:
        dim:          Embedding dimension per token.
        nfeats:       Number of tokens per sample (``N_cat + N_con``).
                      Determines ``row_dim = dim * nfeats``.
        row_heads:    Number of attention heads for row attention.
        dim_row_head: Head dimension for row attention.
        ff_mult:      Feed-forward hidden-size multiplier.
        attn_dropout: Dropout rate inside attention.
        ff_dropout:   Dropout rate inside feed-forward.
        chunk_size:   If set and ``< B`` in train mode, splits the batch into
                      disjoint groups of this size for attention.  See
                      :class:`~fm4tag.models.attention.RowAttention`.
    """

    def __init__(
        self,
        dim: int,
        nfeats: int,
        row_heads: int = 8,
        dim_row_head: int = 64,
        ff_mult: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        row_dim = dim * nfeats
        self.attn = PreNorm(
            row_dim,
            Residual(
                RowAttention(
                    row_dim,
                    heads=row_heads,
                    dim_row_head=dim_row_head,
                    dropout=attn_dropout,
                    chunk_size=chunk_size,
                )
            ),
        )
        self.ff = PreNorm(
            row_dim, Residual(FeedForward(row_dim, mult=ff_mult, dropout=ff_dropout))
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, n, _ = x.shape
        x = rearrange(x, 'b n d -> b (n d)')
        x = self.attn(x, mask=mask)
        x = self.ff(x)
        return rearrange(x, 'b (n d) -> b n d', n=n)


class RowColBlock(nn.Module):
    """One alternating col-then-row transformer layer.

    Column attention + FFN (within-sample), then row attention + FFN
    (intersample), in a single depth step.

    Args:
        dim:          Embedding dimension.
        nfeats:       Number of tokens per sample (``N_cat + N_con``).
        col_heads:    Number of attention heads for column attention.
        row_heads:    Number of attention heads for row attention.
        dim_head:     Head dimension for column attention.
        dim_row_head: Head dimension for row attention.
        ff_mult:      Feed-forward hidden-size multiplier (shared).
        attn_dropout: Dropout rate inside both attention sub-layers.
        ff_dropout:   Dropout rate inside both feed-forward sub-layers.
        chunk_size:   Chunked row attention — see :class:`RowBlock`.
    """

    def __init__(
        self,
        dim: int,
        nfeats: int,
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
        self.col_attn = PreNorm(
            dim,
            Residual(
                Attention(dim, heads=col_heads, dim_head=dim_head, dropout=attn_dropout)
            ),
        )
        self.col_ff = PreNorm(
            dim, Residual(FeedForward(dim, mult=ff_mult, dropout=ff_dropout))
        )
        self.row_attn = PreNorm(
            row_dim,
            Residual(
                RowAttention(
                    row_dim,
                    heads=row_heads,
                    dim_row_head=dim_row_head,
                    dropout=attn_dropout,
                    chunk_size=chunk_size,
                )
            ),
        )
        self.row_ff = PreNorm(
            row_dim, Residual(FeedForward(row_dim, mult=ff_mult, dropout=ff_dropout))
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, n, _ = x.shape
        x = self.col_attn(x, mask=mask)
        x = self.col_ff(x)
        x = rearrange(x, 'b n d -> b (n d)')
        x = self.row_attn(x, mask=mask)
        x = self.row_ff(x)
        return rearrange(x, 'b (n d) -> b n d', n=n)
