from torch import nn

from ..attention import Attention
from ..blocks import FeedForward, PreNormResidual


class Classifier_Transformer(nn.Module):
    """Cross-constituent transformer used in the classification head.

    Uses the shared :class:`~fm4tag.models.attention.Attention` either with a
    per-token ``(B, C)`` key-padding mask over a padded grid, or with
    ``groups`` over packed valid constituents (one group per jet).
    """

    def __init__(
        self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, ff_mult=1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
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

    def forward(self, x, mask=None, groups=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, groups=groups)
            x = ff(x)
        return x
