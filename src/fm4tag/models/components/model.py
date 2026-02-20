import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from einops import rearrange

# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ff_encodings(x, B):
    x_proj = (2.0 * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# classes
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    """
    Standard scaled dot-product attention within tokens of a sample.
    """

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x, mask=None):
        """
        x: (b, n, dim)
        mask: (b) boolean, True = keep sample, False = pad
        """

        _, n, _ = x.shape
        h = self.heads

        # Compute Q,K,V
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        # Build attention mask for SDPA
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, None].repeat(1, 1, 1, n)
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))

        # Fast scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # broadcasting works
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # NO autoregressive attention
        )

        # Merge heads
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class RowAttention(nn.Module):
    """
    Row attention (aka Intersample attention). The scaled dot-product attention is computed
    among the batch samples, instead of within the tokens of a sample.
    """

    def __init__(self, dim, heads=8, dim_row_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_row_head * heads
        self.heads = heads
        self.dim_head = dim_row_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x, mask=None):
        """
        x: (b, dim)
        mask: (b) boolean, True = keep token, False = pad
        """

        h = self.heads

        # Compute Q,K,V
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b (h d) -> h b d", h=h)
        k = rearrange(k, "b (h d) -> h b d", h=h)
        v = rearrange(v, "b (h d) -> h b d", h=h)

        # Build attention mask for SDPA
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None]
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))

        # Fast scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # broadcasting works
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # NO autoregressive attention
        )

        # Merge heads
        out = rearrange(out, "h b d -> b (h d)")
        return self.to_out(out)


# transformer
class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, ff_mult=1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
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
            )

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        b, n, _ = x.shape

        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)

        return x


class RowTransformer(nn.Module):
    def __init__(
        self,
        dim,
        nfeats,
        depth,
        heads,
        dim_row_head,
        attn_dropout,
        ff_dropout,
        ff_mult=1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim * nfeats,
                            Residual(
                                RowAttention(
                                    dim * nfeats,
                                    heads=heads,
                                    dim_row_head=dim_row_head,
                                    dropout=attn_dropout,
                                )
                            ),
                        ),
                        PreNorm(
                            dim * nfeats,
                            Residual(
                                FeedForward(
                                    dim * nfeats, mult=ff_mult, dropout=ff_dropout
                                )
                            ),
                        ),
                    ]
                )
            )

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        b, n, d = x.shape

        for attn_row, ff_row in self.layers:
            x = rearrange(x, "b n d -> b (n d)")
            x = attn_row(x, mask=mask)
            x = ff_row(x)
            x = rearrange(x, "b (n d) -> b n d", n=n)

        return x


class RowColTransformer(nn.Module):
    def __init__(
        self,
        dim,
        nfeats,
        depth,
        heads,
        dim_head,
        dim_row_head,
        attn_dropout,
        ff_dropout,
        ff_mult=1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
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
                        PreNorm(
                            dim * nfeats,
                            Residual(
                                RowAttention(
                                    dim * nfeats,
                                    heads=heads,
                                    dim_row_head=dim_row_head,
                                    dropout=attn_dropout,
                                )
                            ),
                        ),
                        PreNorm(
                            dim * nfeats,
                            Residual(
                                FeedForward(
                                    dim * nfeats, mult=ff_mult, dropout=ff_dropout
                                )
                            ),
                        ),
                    ]
                )
            )

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        b, n, d = x.shape

        for attn, ff, attn_row, ff_row in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
            x = rearrange(x, "b n d -> b (n d)")
            x = attn_row(x, mask=mask)
            x = ff_row(x)
            x = rearrange(x, "b (n d) -> b n d", n=n)

        return x


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=-2)
        return x


class Classifier_Attention(nn.Module):
    """
    Standard scaled dot-product attention within tokens of a sample.
    """

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x, mask=None):
        """
        x: (b, c, dim)
        mask: (b, c) boolean, True = keep sample, False = pad
        """

        _, c, _ = x.shape
        h = self.heads

        # Compute Q,K,V
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b c (h d) -> b h c d", h=h)
        k = rearrange(k, "b c (h d) -> b h c d", h=h)
        v = rearrange(v, "b c (h d) -> b h c d", h=h)

        # Build attention mask for SDPA
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :]
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))

        # Fast scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # broadcasting works
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # NO autoregressive attention
        )

        # Merge heads
        out = rearrange(out, "b h c d -> b c (h d)")
        return self.to_out(out)


class Classifier_Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, ff_mult=1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Residual(
                                Classifier_Attention(
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
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)

        return x

# mlp
class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_dropout(nn.Module):
    def __init__(self, dims, act=None, dropout=0.0):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if not is_last:
                if act is not None:
                    layers.append(act)
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred
