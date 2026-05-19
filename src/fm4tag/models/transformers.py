from torch import nn

from .attention import Classifier_Attention
from .blocks import FeedForward, PreNorm, Residual


class Classifier_Transformer(nn.Module):
    """Cross-constituent transformer used in the classification head."""

    def __init__(
        self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, ff_mult=1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
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
                for _ in range(depth)
            ]
        )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
