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

    ``row_mode`` selects how the ``(B, N, dim)`` token grid is presented to the
    row step; both variants attend over the batch axis ``B`` and return
    ``(B, N, dim)``:

    * ``'per_token'`` (default) — one row attention per token, run independently
      for each of the ``N`` tokens at width ``dim`` with the projections shared
      across tokens.  This is the tabicl ``ColEmbedding`` factorisation: each
      feature builds its own summary across samples, and cross-feature mixing is
      left to the column attention.  ``N`` narrow attentions per step; the cost
      of ``num_inds`` is paid ``N`` times.

    * ``'concat'`` — the token grid is flattened to a single wide row vector
      ``(B, N*dim)`` and the whole row step (norm + attention + feed-forward)
      runs at width ``row_dim = N*dim``.  This is the pre-ISAB "row attention on
      the concatenated token embeddings" shape, but with ISAB
      (:class:`InducedRowAttention`) as the O(B*num_inds) summary in place of the
      old O(B²) / chunked attention.  One wide attention per step instead of
      ``N`` narrow ones: it uses the GPU far better and ``num_inds`` is paid once,
      so it can grow cheaply — the whole point of recovering this shape.  The
      trade is parameter count: the projections and feed-forward now scale with
      ``row_dim`` rather than ``dim``.

    Args:
        dim:      Token embedding width.
        nfeats:   Number of tokens ``N``.  Sizes the per-token inducing points in
                  ``per_token`` mode, and the flattened ``row_dim = N*dim`` in
                  ``concat`` mode.
        num_inds: Inducing points for :class:`InducedRowAttention`; ``None``
                  selects all-pairs :class:`RowAttention` instead.  Orthogonal to
                  ``row_mode`` — every (mode, num_inds) pair is valid.
        row_mode: ``'per_token'`` | ``'concat'`` (see above).
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
        row_mode: str = 'per_token',
    ) -> None:
        super().__init__()
        if row_mode not in ('per_token', 'concat'):
            raise ValueError(
                f"row_mode must be 'per_token' or 'concat', got {row_mode!r}"
            )
        self.row_mode = row_mode

        # In concat mode the whole feature vector is a single row token of width
        # row_dim; the attention then has nfeats == 1 (one inducing-point set).
        width = dim * nfeats if row_mode == 'concat' else dim
        row_nfeats = 1 if row_mode == 'concat' else nfeats

        self.attn = PreNormResidual(
            width,
            _build_row_attention(
                width,
                row_nfeats,
                heads=heads,
                dim_row_head=dim_row_head,
                dropout=attn_dropout,
                num_inds=num_inds,
            ),
        )
        self.ff = PreNormResidual(
            width, FeedForward(width, mult=ff_mult, dropout=ff_dropout)
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.row_mode == 'concat':
            b, n, d = x.shape
            # Flatten the token grid into one wide row vector; the row step runs
            # at width row_dim = n*d, then we restore the grid.
            x = x.reshape(b, 1, n * d)
            x = self.attn(x, mask=mask)
            x = self.ff(x)
            return x.reshape(b, n, d)
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
        row_mode:     ``'per_token'`` | ``'concat'`` — how the row step sees the
                      token grid; see :class:`RowMixer`.
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
        row_mode: str = 'per_token',
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
                    row_mode=row_mode,
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
        row_mode:     ``'per_token'`` | ``'concat'`` — how the row step sees the
                      token grid; see :class:`RowMixer`.
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
        row_mode: str = 'per_token',
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
                            row_mode=row_mode,
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
