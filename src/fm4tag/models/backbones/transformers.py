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
from ..blocks import FeedForward, PreNormResidual


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


class RowMixer(nn.Module):
    """One depth step of intersample (row) attention + feed-forward.

    Operates on the flattened ``(B, row_dim)`` representation, where
    ``row_dim = dim * nfeats``.  When ``out_row_dim`` is set and smaller than
    ``row_dim`` the attention and FFN run inside a down-projected
    ``out_row_dim`` bottleneck — so their (dominant) cost scales with
    ``out_row_dim`` instead of the full ``row_dim`` — wrapped by a single
    residual whose up-projection is zero-initialised, so the block starts as
    the identity and learns intersample mixing from there (adapter-style).
    ``out_row_dim=None`` (or ``>= row_dim``) keeps the plain full-width path.

    Args:
        row_dim:     Flattened feature dimension ``dim * nfeats``.
        out_row_dim: Bottleneck width; ``None`` or ``>= row_dim`` disables it.
    """

    def __init__(
        self,
        row_dim: int,
        out_row_dim: int | None,
        *,
        heads: int,
        dim_row_head: int,
        ff_mult: int,
        attn_dropout: float,
        ff_dropout: float,
        chunk_size: int | None,
    ) -> None:
        super().__init__()
        self.bottleneck = out_row_dim is not None and out_row_dim < row_dim
        d = out_row_dim if self.bottleneck else row_dim
        if self.bottleneck:
            self.down = nn.Linear(row_dim, d)
            self.up = nn.Linear(d, row_dim)
            # Zero-init the up-projection so the mixer starts as the identity.
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        self.attn = PreNormResidual(
            d,
            _build_row_attention(
                d,
                heads=heads,
                dim_row_head=dim_row_head,
                dropout=attn_dropout,
                chunk_size=chunk_size,
            ),
        )
        self.ff = PreNormResidual(d, FeedForward(d, mult=ff_mult, dropout=ff_dropout))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.bottleneck:
            z = self.down(x)
            z = self.attn(z, mask=mask)
            z = self.ff(z)
            return x + self.up(z)
        x = self.attn(x, mask=mask)
        x = self.ff(x)
        return x


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
        out_row_dim:  If set (and ``< row_dim``), run the row attention+FFN in
                      a down-projected bottleneck of this width (see
                      :class:`RowMixer`); ``None`` keeps the full ``row_dim``.
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
        out_row_dim: int | None = None,
    ) -> None:
        super().__init__()
        row_dim = dim * nfeats
        self.blocks = nn.ModuleList(
            [
                RowMixer(
                    row_dim,
                    out_row_dim,
                    heads=row_heads,
                    dim_row_head=dim_row_head,
                    ff_mult=ff_mult,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    chunk_size=chunk_size,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, n, _ = x.shape
        x = rearrange(x, 'b n d -> b (n d)')
        for mixer in self.blocks:
            x = mixer(x, mask=mask)
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
        out_row_dim:  If set (and ``< row_dim``), run the row attention+FFN in a
                      down-projected bottleneck of this width (see
                      :class:`RowMixer`); ``None`` keeps the full ``row_dim``.
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
        out_row_dim: int | None = None,
    ) -> None:
        super().__init__()
        row_dim = dim * nfeats
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
                            row_dim,
                            out_row_dim,
                            heads=row_heads,
                            dim_row_head=dim_row_head,
                            ff_mult=ff_mult,
                            attn_dropout=attn_dropout,
                            ff_dropout=ff_dropout,
                            chunk_size=chunk_size,
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
        for col_attn, col_ff, row_mixer in self.blocks:
            x = col_attn(x, mask=col_mask)
            x = col_ff(x)
            x = rearrange(x, 'b n d -> b (n d)')
            x = row_mixer(x, mask=mask)
            x = rearrange(x, 'b (n d) -> b n d', n=n)
        return x
