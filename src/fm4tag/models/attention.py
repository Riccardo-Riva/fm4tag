import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False

_FLASH_ATTN_DTYPES = (torch.float16, torch.bfloat16)


def sdpa(q, k, v, attn_mask=None, dropout_p=0.0):
    """Scaled dot-product attention, routed through FlashAttention-2 when possible.

    ``q``/``k``/``v``: ``(..., h, seq, d)`` — any number of leading batch-like
    dims (none for :class:`RowAttention`, one for :class:`Attention`, one
    "group" dim for :class:`ChunkedRowAttention`).  Falls back to
    ``F.scaled_dot_product_attention`` whenever an explicit ``attn_mask`` is
    given (flash-attn doesn't support arbitrary attention masks), when
    ``flash-attn`` isn't installed, or off-CUDA/unsupported dtype.
    """
    if (
        HAS_FLASH_ATTN
        and attn_mask is None
        and q.is_cuda
        and q.dtype in _FLASH_ATTN_DTYPES
    ):
        orig_shape = q.shape
        qf = q.reshape(-1, *q.shape[-3:]).transpose(1, 2).contiguous()
        kf = k.reshape(-1, *k.shape[-3:]).transpose(1, 2).contiguous()
        vf = v.reshape(-1, *v.shape[-3:]).transpose(1, 2).contiguous()
        out = flash_attn_func(qf, kf, vf, dropout_p=dropout_p)
        return out.transpose(1, 2).reshape(orig_shape)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )


class Attention(nn.Module):
    """Scaled dot-product self-attention over the token dimension of a sample.

    Attends across the ``n`` tokens of each sample independently.  The optional
    mask is a per-token key-padding mask, so padded positions can be excluded
    individually (used e.g. for cross-constituent attention within a jet).

    Args:
        x:    ``(b, n, dim)`` — batch of token sequences.
        mask: ``(b, n)`` bool — ``True`` = valid token, ``False`` = padding.
    """

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x, mask=None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :]
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))

        out = sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class RowAttention(nn.Module):
    """All-pairs intersample (row) attention over the batch axis.

    Every sample attends to every other sample in the batch, independently for
    each of the ``N`` token positions and at width ``dim`` — the token axis is
    a batch dimension, so one attention operator is shared across tokens.  Cost
    is O(B²) in the batch size; see :class:`InducedRowAttention` for the
    O(B·num_inds) inducing-point variant.

    Consequence: outputs (also at inference) depend on batch composition.

    Args:
        x:    ``(B, N, dim)`` — token grid.
        mask: ``(B,)`` bool — ``True`` = valid sample, ``False`` = padding.
    """

    def __init__(self, dim, heads=8, dim_row_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_row_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x, mask=None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # (B, N, h*d) → (N, h, B, d): tokens become a batch dim, the sample axis
        # B becomes the attention sequence.
        q = rearrange(q, 'b n (h d) -> n h b d', h=h)
        k = rearrange(k, 'b n (h d) -> n h b d', h=h)
        v = rearrange(v, 'b n (h d) -> n h b d', h=h)

        attn_mask = None
        if mask is not None:
            # Key-padding mask over the sample axis: (B,) → (1, 1, 1, B).
            attn_mask = torch.where(mask, 0.0, float('-inf'))[None, None, None, :]

        out = sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, 'n h b d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention with the sample axis as the sequence.

    Both operands are ``(N, S, dim)``: the token axis ``N`` is a batch
    dimension, so the projections are shared across token positions, and ``S``
    (samples, or inducing points) is the attention sequence.  Used to build the
    two stages of :class:`InducedRowAttention`.
    """

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x_q, x_kv, attn_mask=None):
        h = self.heads

        q = rearrange(self.to_q(x_q), 'n s (h d) -> n h s d', h=h)
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)
        k = rearrange(k, 'n s (h d) -> n h s d', h=h)
        v = rearrange(v, 'n s (h d) -> n h s d', h=h)

        out = sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, 'n h s d -> n s (h d)')
        return self.to_out(out)


class InducedRowAttention(nn.Module):
    """Induced intersample (row) attention — ISAB over the batch axis.

    Two cross-attentions through ``num_inds`` learned inducing points replace
    the all-pairs row attention of :class:`RowAttention`:

    1. the inducing points read the whole batch → summary ``(num_inds, dim)``
    2. every sample reads that summary          → ``(B, dim)``

    Cost is O(B · num_inds) rather than O(B²), and the batch size only ever
    appears as a key/query length against a *fixed* number of inducing points.
    A variable ``B`` — the packed valid-constituent count changes every step —
    therefore needs no chunking, no random permutation and no tail padding, and
    no attention mask is required, which keeps the whole row path eligible for
    FlashAttention.

    Like :class:`RowAttention` this runs per token at width ``dim`` with the
    projections shared across the ``N`` token positions: the tabicl
    ``ColEmbedding`` factorisation, in which each feature builds its own
    distribution summary across samples and cross-feature mixing is left to the
    column attention.

    Two deliberate choices:

    * The inducing points are **per token** (``(N, num_inds, dim)``).  tabicl
      shares one set across columns because its column count varies per table;
      here ``N`` is fixed per encoder and each token is a distinct feature with
      its own embedding subspace, so each gets its own probes.  The cost is
      ``N · num_inds · dim`` parameters (~19k for tracks).
    * ``to_out`` of the second cross-attention is **zero-initialised**, so under
      the surrounding residual the block starts as the identity and learns
      intersample mixing from there.  This replaces the zero-initialised
      up-projection of the old ``out_row_dim`` bottleneck as the stabiliser for
      the row path.

    Args:
        x:    ``(B, N, dim)`` — token grid.
        mask: ``(B,)`` bool — ``True`` = valid sample, ``False`` = padding.
    """

    def __init__(self, dim, nfeats, heads=8, dim_row_head=16, dropout=0.0, num_inds=16):
        super().__init__()
        self.inds = nn.Parameter(torch.empty(nfeats, num_inds, dim))
        nn.init.trunc_normal_(self.inds, std=0.02)

        self.norm_inds = nn.LayerNorm(dim)
        self.attn_inds = CrossAttention(
            dim, heads=heads, dim_head=dim_row_head, dropout=dropout
        )
        self.norm_summary = nn.LayerNorm(dim)
        self.attn_out = CrossAttention(
            dim, heads=heads, dim_head=dim_row_head, dropout=dropout
        )

        nn.init.zeros_(self.attn_out.to_out.weight)
        nn.init.zeros_(self.attn_out.to_out.bias)

    def forward(self, x, mask=None):
        # (B, N, dim) → (N, B, dim): tokens become a batch dim, samples become
        # the attention sequence.
        xt = rearrange(x, 'b n d -> n b d')

        attn_mask = None
        if mask is not None:
            # Samples are keys only in the first cross-attention; the inducing
            # points (keys of the second) are never padding.
            attn_mask = torch.where(mask, 0.0, float('-inf'))[None, None, None, :]

        inds = self.norm_inds(self.inds)  # (N, num_inds, dim)
        summary = inds + self.attn_inds(inds, xt, attn_mask)  # (N, num_inds, dim)
        out = self.attn_out(xt, self.norm_summary(summary))  # (N, B, dim)
        return rearrange(out, 'n b d -> b n d')
