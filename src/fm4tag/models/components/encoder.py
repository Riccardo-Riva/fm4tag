from .model import *


class saint_encoder(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_row_head=64,
        num_special_tokens=0,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=1,
        cont_embeddings="MLP",
        attentiontype="col",
        final_mlp_style="sep",
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), (
            "number of each category must be positive"
        )

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(
            torch.tensor(list(categories)), (1, 0), value=num_special_tokens
        )
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer("categories_offset", categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        # TODO(check for a better continuous feature embedding strategy)
        if num_continuous > 0 and self.cont_embeddings == "MLP":
            nfeats = self.num_categories + num_continuous
            H = 2 * dim  # hidden size in each MLP
            self.cont_fc1 = nn.Conv1d(
                num_continuous, num_continuous * H, kernel_size=1, groups=num_continuous
            )
            self.cont_fc2 = nn.Conv1d(
                num_continuous * H,
                num_continuous * dim,
                kernel_size=1,
                groups=num_continuous,
            )
        elif self.cont_embeddings == "pos_singleMLP":
            self.cont_MLP = nn.ModuleList(
                [simple_MLP([1, 2 * self.dim, self.dim]) for _ in range(1)]
            )
            nfeats = self.num_categories + num_continuous
        else:
            print("Continous features are not passed through attention")
            nfeats = self.num_categories

        # embedding layer for categorical features
        self.embeds = nn.Embedding(self.total_tokens, self.dim)

        # transformer
        if attentiontype == "col":
            self.transformer = Transformer(
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                ff_mult=ff_mult,
            )
        elif attentiontype == "colrow":
            self.transformer = RowColTransformer(
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                dim_row_head=dim_row_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                ff_mult=ff_mult,
            )
        elif attentiontype == "row":
            self.transformer = RowTransformer(
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_row_head=dim_row_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                ff_mult=ff_mult,
            )
        elif attentiontype == "concat":
            # the self.transofrmer simply concatenates the categorical and continuous features
            self.transformer = Concat()
        else:
            raise NotImplementedError(f"Attention type {attentiontype} not implemented")

        # self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)

        if self.final_mlp_style == "common":
            self.mlp1 = simple_MLP([dim, (self.total_tokens) * 2, self.total_tokens])
            self.mlp2 = simple_MLP([dim, (self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(
                dim, self.num_continuous, torch.ones(self.num_continuous).int()
            )

        self.pt_mlp1 = simple_MLP(
            [
                dim * (self.num_continuous + self.num_categories),
                6 * dim * (self.num_continuous + self.num_categories) // 5,
                dim * (self.num_continuous + self.num_categories) // 2,
            ]
        )
        self.pt_mlp2 = simple_MLP(
            [
                dim * (self.num_continuous + self.num_categories),
                6 * dim * (self.num_continuous + self.num_categories) // 5,
                dim * (self.num_continuous + self.num_categories) // 2,
            ]
        )

    def forward(self, x_categ, x_cont, mask=None):
        return self.transformer(x_categ, x_cont, mask)
