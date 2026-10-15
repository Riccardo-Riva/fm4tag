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
    dims (one for :class:`Attention`, ``n`` tokens for :class:`RowAttention`).
    Falls back to ``F.scaled_dot_product_attention`` whenever an explicit
    ``attn_mask`` is given (flash-attn doesn't support arbitrary attention
    masks), when ``flash-attn`` isn't installed, or off-CUDA/unsupported dtype.

    Do not route the small column attentions (``seq=19``, ``head_dim=8``) onto
    the MATH backend to dodge the padding the fused kernels apply: measured
    2026-08-17, it does cut their backward ~20%, but triples their forward
    (unaligned GEMMs + fp32 scores) for a net 2-3% loss and +0.4 GiB.
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
    is O(B²) in the batch size.

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
