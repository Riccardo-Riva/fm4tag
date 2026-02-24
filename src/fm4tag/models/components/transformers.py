import torch
from torch import nn
from einops import rearrange

from .attention import Attention, Classifier_Attention, RowAttention
from .blocks import FeedForward, PreNorm, Residual


class Transformer(nn.Module):
    """Column-wise (within-sample) transformer encoder."""

    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, ff_mult=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, mult=ff_mult, dropout=ff_dropout))),
            ])
            for _ in range(depth)
        ])

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class RowTransformer(nn.Module):
    """Row (intersample) transformer encoder."""

    def __init__(self, dim, nfeats, depth, heads, dim_row_head, attn_dropout, ff_dropout, ff_mult=1):
        super().__init__()
        row_dim = dim * nfeats
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(row_dim, Residual(RowAttention(row_dim, heads=heads, dim_row_head=dim_row_head, dropout=attn_dropout))),
                PreNorm(row_dim, Residual(FeedForward(row_dim, mult=ff_mult, dropout=ff_dropout))),
            ])
            for _ in range(depth)
        ])

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        for attn_row, ff_row in self.layers:
            x = rearrange(x, 'b n d -> b (n d)')
            x = attn_row(x, mask=mask)
            x = ff_row(x)
            x = rearrange(x, 'b (n d) -> b n d', n=n)
        return x


class RowColTransformer(nn.Module):
    """Alternating column-then-row transformer encoder."""

    def __init__(self, dim, nfeats, depth, heads, dim_head, dim_row_head, attn_dropout, ff_dropout, ff_mult=1):
        super().__init__()
        row_dim = dim * nfeats
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, mult=ff_mult, dropout=ff_dropout))),
                PreNorm(row_dim, Residual(RowAttention(row_dim, heads=heads, dim_row_head=dim_row_head, dropout=attn_dropout))),
                PreNorm(row_dim, Residual(FeedForward(row_dim, mult=ff_mult, dropout=ff_dropout))),
            ])
            for _ in range(depth)
        ])

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        for attn, ff, attn_row, ff_row in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
            x = rearrange(x, 'b n d -> b (n d)')
            x = attn_row(x, mask=mask)
            x = ff_row(x)
            x = rearrange(x, 'b (n d) -> b n d', n=n)
        return x


class Concat(nn.Module):
    """Trivial 'transformer' that only concatenates categorical and continuous tokens."""

    def __init__(self):
        super().__init__()

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=-2)
        return x


class Classifier_Transformer(nn.Module):
    """Cross-constituent transformer used in the classification head."""

    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, ff_mult=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Residual(Classifier_Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, mult=ff_mult, dropout=ff_dropout))),
            ])
            for _ in range(depth)
        ])

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
