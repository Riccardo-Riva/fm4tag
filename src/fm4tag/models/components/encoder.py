import warnings

import torch
import torch.nn.functional as F
from torch import nn

from .mlp import sep_MLP, simple_MLP
from .transformers import Concat, RowColTransformer, RowTransformer, Transformer


class GlobalEncoder(nn.Module):
    """Per-feature MLP encoder for global flat continuous features.

    Each of the ``num_features`` continuous features is independently projected
    to a ``dim``-dimensional embedding via a 2-layer grouped Conv1d MLP.  No
    attention is applied across features — the output is a plain set of token
    embeddings, one per global feature.

    This encoder is used to pretrain global-object representations alongside the
    constituent :class:`Encoder`.  Its output shape ``(N, F_g, dim)`` mirrors the
    per-constituent-token output of :class:`Encoder`, so downstream heads can
    treat both uniformly.

    Args:
        num_features: Number of global continuous features ``F_g``.
        dim:          Embedding dimension (should match the constituent
                      :class:`Encoder` ``dim``).
    """

    def __init__(self, num_features: int, dim: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.dim = dim

        H = 2 * dim  # hidden size inside each per-feature MLP
        self.fc1 = nn.Conv1d(
            num_features, num_features * H, kernel_size=1, groups=num_features
        )
        self.fc2 = nn.Conv1d(
            num_features * H, num_features * dim, kernel_size=1, groups=num_features
        )

        # Denoising reconstruction head: one scalar per feature → (N, 1) each.
        self.mlp_recon = sep_MLP(dim, num_features, [1] * num_features)

        # Contrastive projection heads.
        proj_in = num_features * dim
        proj_hidden = 6 * proj_in // 5
        proj_out = proj_in // 2
        self.pt_mlp1 = simple_MLP([proj_in, proj_hidden, proj_out])
        self.pt_mlp2 = simple_MLP([proj_in, proj_hidden, proj_out])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed global features.

        Args:
            x: ``(N, F_g)`` continuous feature values (already normalised).

        Returns:
            ``(N, F_g, dim)`` — one embedding token per feature.
        """
        h = F.relu(self.fc1(x.unsqueeze(-1)))  # (N, F_g*H, 1)
        out = self.fc2(h).squeeze(-1)           # (N, F_g*dim)
        return out.view(x.size(0), self.num_features, self.dim)  # (N, F_g, dim)


class Encoder(nn.Module):
    """SAINT-style transformer encoder for mixed categorical/continuous tabular data.

    Each constituent (e.g. track) is treated as a row of tabular features.
    Categorical features are embedded via a learned lookup table; continuous
    features are projected per-feature through a grouped Conv1d MLP.  All
    embedded tokens are then processed by a standard (column-wise), row-wise,
    or alternating transformer.

    The **first categorical feature** (index 0) plays the role of a CLS token:
    the :class:`~fm4tag.models.components.heads.ClassifierHead` extracts it as
    the per-constituent summary, and the denoising loss skips reconstructing it.

    Args:
        categories:       Cardinality of each categorical feature (positive ints).
        num_continuous:   Number of continuous features.
        dim:              Embedding dimension shared by all tokens.
        depth:            Number of transformer layers.
        heads:            Number of attention heads.
        dim_head:         Dimension per head (column attention).
        dim_row_head:     Dimension per head (row attention).
        num_special_tokens: Extra tokens prepended to the embedding table.
        attn_dropout:     Dropout inside attention.
        ff_dropout:       Dropout inside feed-forward sub-layers.
        ff_mult:          Feed-forward expansion factor.
        cont_embeddings:  Continuous embedding strategy:
                          ``'MLP'``            – per-feature grouped Conv1d MLP;
                          ``'pos_singleMLP'``  – single shared MLP;
                          ``None``             – continuous features ignored.
        attentiontype:    Transformer variant: ``'col'``, ``'colrow'``,
                          ``'row'``, or ``'concat'``.
        final_mlp_style:  Reconstruction head style: ``'sep'`` (per-feature) or
                          ``'common'`` (shared).
    """

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
        cont_embeddings='MLP',
        attentiontype='col',
        final_mlp_style='sep',
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), (
            'number of each category must be positive'
        )

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # Offset buffer: maps raw category indices to positions in embeds.
        categories_offset = F.pad(
            torch.tensor(list(categories)), (1, 0), value=num_special_tokens
        )
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if num_continuous > 0 and cont_embeddings == 'MLP':
            nfeats = self.num_categories + num_continuous
            H = 2 * dim
            self.cont_fc1 = nn.Conv1d(
                num_continuous, num_continuous * H, kernel_size=1, groups=num_continuous
            )
            self.cont_fc2 = nn.Conv1d(
                num_continuous * H, num_continuous * dim, kernel_size=1, groups=num_continuous
            )
        elif cont_embeddings == 'pos_singleMLP':
            self.cont_MLP = nn.ModuleList(
                [simple_MLP([1, 2 * dim, dim]) for _ in range(1)]
            )
            nfeats = self.num_categories + num_continuous
        else:
            warnings.warn(
                'cont_embeddings not set to MLP or pos_singleMLP — '
                'continuous features will not be passed through attention.',
                stacklevel=2,
            )
            nfeats = self.num_categories

        self.embeds = nn.Embedding(self.total_tokens, dim)

        # Transformer backbone.
        if attentiontype == 'col':
            self.transformer = Transformer(
                dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                attn_dropout=attn_dropout, ff_dropout=ff_dropout, ff_mult=ff_mult,
            )
        elif attentiontype == 'colrow':
            self.transformer = RowColTransformer(
                dim=dim, nfeats=nfeats, depth=depth, heads=heads,
                dim_head=dim_head, dim_row_head=dim_row_head,
                attn_dropout=attn_dropout, ff_dropout=ff_dropout, ff_mult=ff_mult,
            )
        elif attentiontype == 'row':
            self.transformer = RowTransformer(
                dim=dim, nfeats=nfeats, depth=depth, heads=heads,
                dim_row_head=dim_row_head,
                attn_dropout=attn_dropout, ff_dropout=ff_dropout, ff_mult=ff_mult,
            )
        elif attentiontype == 'concat':
            self.transformer = Concat()
        else:
            raise NotImplementedError(f'Attention type {attentiontype!r} not implemented')

        # Reconstruction heads (used by DenoisingLoss during pretraining).
        if final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim, self.total_tokens * 2, self.total_tokens])
            self.mlp2 = simple_MLP([dim, num_continuous, 1])
        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(dim, num_continuous, torch.ones(num_continuous).int())

        # Projection heads for the contrastive (InfoNCE) loss.
        proj_in = dim * (num_continuous + self.num_categories)
        proj_hidden = proj_in // 2
        proj_out = proj_in // 4
        self.pt_mlp1 = simple_MLP([proj_in, proj_hidden, proj_out])
        self.pt_mlp2 = simple_MLP([proj_in, proj_hidden, proj_out])

    def forward(self, x_categ, x_cont, mask=None):
        return self.transformer(x_categ, x_cont, mask)
