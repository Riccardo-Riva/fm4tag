import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


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

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class RowAttention(nn.Module):
    """Intersample (row) attention across the whole batch dimension.

    Every sample attends to every other sample in the batch.  For chunked
    (grouped) intersample attention see :class:`ChunkedRowAttention`.

    Args:
        x:    ``(B, dim)`` — one flattened feature vector per sample.
        mask: ``(B,)`` bool — ``True`` = valid, ``False`` = padding.
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
        q = rearrange(q, 'b (h d) -> h b d', h=h)
        k = rearrange(k, 'b (h d) -> h b d', h=h)
        v = rearrange(v, 'b (h d) -> h b d', h=h)

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None]
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = rearrange(out, 'h b d -> b (h d)')
        return self.to_out(out)


class ChunkedRowAttention(nn.Module):
    """Intersample (row) attention within disjoint chunks of the batch.

    The batch is split into disjoint groups of ``chunk_size`` samples and
    attention is computed independently within each group.  In **train** mode a
    random permutation is applied before chunking and its inverse afterwards, so
    the caller sees outputs in the original batch order.  In **eval** mode
    contiguous (identity) chunks are used, keeping evaluation deterministic.

    When ``chunk_size`` does not divide ``B`` (including ``chunk_size >= B``,
    which yields a single group) the batch is zero-padded up to a multiple of
    ``chunk_size``; the padded rows are discarded after attention.

    Args:
        x:          ``(B, dim)`` — one flattened feature vector per sample.
        chunk_size: Group size along B.
    """

    def __init__(self, dim, heads=8, dim_row_head=16, dropout=0.0, chunk_size=None):
        super().__init__()
        if chunk_size is None:
            raise ValueError('ChunkedRowAttention requires chunk_size to be set')
        inner_dim = dim_row_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout
        self.chunk_size = chunk_size

    def forward(self, x, mask=None):
        B = x.size(0)
        h = self.heads

        if self.training:
            perm = torch.randperm(B, device=x.device)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(B, device=x.device)
            xp = x[perm]
        else:
            xp = x

        pad = (-B) % self.chunk_size
        if pad:
            # pad the B axis with zeros; padded rows are discarded after
            xp = F.pad(xp, (0, 0, 0, pad))
        xp = rearrange(xp, '(g c) d -> g c d', c=self.chunk_size)

        # Mask the padded rows as keys so they do not dilute the softmax of the
        # valid samples sharing the (last) incomplete group.  Only key columns
        # are masked, so every query row keeps at least one valid key
        # (pad < chunk_size) and no row becomes fully -inf (hence no NaN); the
        # padded query rows are discarded by the ``[:B]`` slice below.
        attn_mask = None
        if pad:
            g = xp.size(0)
            key_valid = torch.arange(g * self.chunk_size, device=x.device) < B
            key_valid = rearrange(key_valid, '(g c) -> g c', c=self.chunk_size)
            attn_mask = torch.where(key_valid[:, None, None, :], 0.0, float('-inf'))

        q, k, v = self.to_qkv(xp).chunk(3, dim=-1)
        q = rearrange(q, 'g c (h d) -> g h c d', h=h)
        k = rearrange(k, 'g c (h d) -> g h c d', h=h)
        v = rearrange(v, 'g c (h d) -> g h c d', h=h)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = rearrange(out, 'g h c d -> g c (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'g c d -> (g c) d')[:B]
        if self.training:
            return out[inv]
        return out
