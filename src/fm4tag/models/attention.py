import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class Attention(nn.Module):
    """Standard scaled dot-product attention within tokens of a single sample.

    Operates along the feature/token dimension (column attention).

    Args:
        x:    ``(b, n, dim)`` — batch of sequences.
        mask: ``(b,)`` bool — ``True`` = valid sample, ``False`` = padding.
    """

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x, mask=None):
        _, n, _ = x.shape
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, None].repeat(1, 1, 1, n)
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
    """Intersample (row) attention across the batch dimension.

    When ``chunk_size`` is set and ``chunk_size < B``, the batch is split into
    disjoint groups of ``chunk_size`` samples.  In **train** mode a random
    permutation is applied before chunking and its inverse is applied after,
    so the caller sees outputs in the original batch order.  In **eval** mode
    contiguous (identity) chunks are used, keeping evaluation deterministic.

    ``chunk_size=None`` or ``chunk_size >= B`` always uses whole-batch attention.

    Args:
        x:          ``(B, dim)`` — one flattened feature vector per sample.
        mask:       ``(B,)`` bool — ``True`` = valid, ``False`` = padding.
                    Applied only in the whole-batch path.
        chunk_size: Group size along B.
    """

    def __init__(self, dim, heads=8, dim_row_head=16, dropout=0.0, chunk_size=None):
        super().__init__()
        inner_dim = dim_row_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout
        self.chunk_size = chunk_size

    def forward(self, x, mask=None):
        B = x.size(0)
        h = self.heads

        if self.chunk_size is not None and self.chunk_size < B:
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

            q, k, v = self.to_qkv(xp).chunk(3, dim=-1)
            q = rearrange(q, 'g c (h d) -> g h c d', h=h)
            k = rearrange(k, 'g c (h d) -> g h c d', h=h)
            v = rearrange(v, 'g c (h d) -> g h c d', h=h)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            out = rearrange(out, 'g h c d -> g c (h d)')
            out = self.to_out(out)
            out = rearrange(out, 'g c d -> (g c) d')[:B]
            if self.training:
                return out[inv]
            return out

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


class Classifier_Attention(nn.Module):
    """Scaled dot-product attention across constituents within a jet.

    Args:
        x:    ``(b, c, dim)`` — batch of jets, each with ``c`` constituents.
        mask: ``(b, c)`` bool — ``True`` = valid constituent, ``False`` = padding.
    """

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x, mask=None):
        _, c, _ = x.shape
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b c (h d) -> b h c d', h=h)
        k = rearrange(k, 'b c (h d) -> b h c d', h=h)
        v = rearrange(v, 'b c (h d) -> b h c d', h=h)

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
        out = rearrange(out, 'b h c d -> b c (h d)')
        return self.to_out(out)
