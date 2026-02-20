import torch
from torch import nn

from .model import Classifier_Transformer, MLP_dropout


class classifier(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        y_dim=2,
        mlp_dropout=0.0,
        ff_dropout=0.0,
        attn_dropout=0.0,
        ff_mult=4,
        heads=8,
        dim_head=16,
        depth=3,
    ):
        super().__init__()

        self.encoder = encoder
        self.dim = encoder.dim
        self.norm = nn.LayerNorm(self.dim)

        # total_dim = (encoder.num_categories+encoder.num_continuous)*dim

        self.mlpphi = MLP_dropout([self.dim, 512, self.dim], dropout=mlp_dropout)
        self.transformer = Classifier_Transformer(
            self.dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
        )
        self.mlpfory = MLP_dropout([self.dim, 512, 256, y_dim], dropout=mlp_dropout)

    def forward(self, x, valids):
        # x -> (b, c, 1(CLS)+f_cat+f_con, d)
        # valids -> (b, c)
        x = x[:, :, 0, :]  # (b, c, d)
        x = self.mlpphi(x)  # (b, c, d)

        # zero out invalid entries
        x = torch.where(valids.unsqueeze(-1), x, 0.0)  # (b, c, d)
        x = self.transformer(x, valids)  # (b, c, d)

        n_valids = valids.unsqueeze(-1).sum(dim=1)  # (b,)

        x = x.sum(dim=1) / n_valids  # (b, d)
        x = self.mlpfory(x)  # (b,)
        return x
