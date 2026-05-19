import warnings

import torch
import torch.nn.functional as F
from torch import nn

from .mlp import sep_MLP, simple_MLP


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

    The contrastive projection heads always use ``proj_in = num_features * dim``
    as input, ``2 * proj_in`` as the hidden dimension, and ``proj_in`` as the
    output dimension.

    Args:
        num_features: Number of global continuous features ``F_g``.
        dim:          Embedding dimension (should match the constituent
                      :class:`Encoder` ``dim``).
    """

    def __init__(
        self,
        num_features: int,
        dim: int,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.dim = dim

        H = 2 * dim
        self.fc1 = nn.Conv1d(
            num_features, num_features * H, kernel_size=1, groups=num_features
        )
        self.fc2 = nn.Conv1d(
            num_features * H, num_features * dim, kernel_size=1, groups=num_features
        )

        self.mlp_recon = sep_MLP(dim, num_features, [1] * num_features)

        proj_in = num_features * dim
        self.pt_mlp1 = simple_MLP([proj_in, 2 * proj_in, proj_in])
        self.pt_mlp2 = simple_MLP([proj_in, 2 * proj_in, proj_in])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed global features.

        Args:
            x: ``(N, F_g)`` continuous feature values (already normalised).

        Returns:
            ``(N, F_g, dim)`` — one embedding token per feature.
        """
        h = F.relu(self.fc1(x.unsqueeze(-1)))  # (N, F_g*H, 1)
        out = self.fc2(h).squeeze(-1)  # (N, F_g*dim)
        return out.view(x.size(0), self.num_features, self.dim)  # (N, F_g, dim)


class Encoder(nn.Module):
    """SAINT-style transformer encoder with a composable list of transformer blocks.

    The transformer backbone is a flat list of block modules (e.g.
    :class:`~fm4tag.models.transformer_blocks.ColBlock`,
    :class:`~fm4tag.models.transformer_blocks.RowColBlock`) configured via
    ``_target_`` entries in the YAML config, mirroring the augmentation system.

    Each categorical feature is embedded via a learned lookup table; each
    continuous feature is projected per-feature through a grouped Conv1d MLP.
    The embedded tokens are then passed through the block sequence.

    The **first categorical feature** (index 0) plays the role of a CLS token.

    Args:
        categories:         Cardinality of each categorical feature (positive ints).
        num_continuous:     Number of continuous features.
        dim:                Embedding dimension shared by all tokens.
        transformer_layers: Pre-built block modules.  With Hydra ``instantiate``
                            these are constructed from ``_target_`` list entries
                            before being passed here.
        num_special_tokens: Extra tokens prepended to the embedding table.
        cont_embeddings:    Continuous embedding strategy:
                            ``'MLP'``            – per-feature grouped Conv1d MLP;
                            ``'pos_singleMLP'``  – single shared MLP;
                            ``None``             – continuous features ignored.
        final_mlp_style:    Reconstruction head style: ``'sep'`` or ``'common'``.
        proj_hidden:        Hidden dim of the contrastive projection heads.
                            ``None`` → ``3 * proj_in // 4`` (auto).
        proj_out:           Output dim of the contrastive projection heads.
                            ``None`` → ``proj_in // 2`` (auto).
    """

    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        transformer_layers: list,
        num_special_tokens: int = 0,
        cont_embeddings: str = 'MLP',
        final_mlp_style: str = 'sep',
        proj_hidden=None,
        proj_out=None,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), (
            'number of each category must be positive'
        )

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        categories_offset = F.pad(
            torch.tensor(list(categories)), (1, 0), value=num_special_tokens
        )
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.final_mlp_style = final_mlp_style

        if num_continuous > 0 and cont_embeddings == 'MLP':
            H = 2 * dim
            self.cont_fc1 = nn.Conv1d(
                num_continuous, num_continuous * H, kernel_size=1, groups=num_continuous
            )
            self.cont_fc2 = nn.Conv1d(
                num_continuous * H,
                num_continuous * dim,
                kernel_size=1,
                groups=num_continuous,
            )
        elif cont_embeddings == 'pos_singleMLP':
            self.cont_MLP = nn.ModuleList(
                [simple_MLP([1, 2 * dim, dim]) for _ in range(1)]
            )
        else:
            warnings.warn(
                'cont_embeddings not set to MLP or pos_singleMLP — '
                'continuous features will not be passed through attention.',
                stacklevel=2,
            )

        self.embeds = nn.Embedding(self.total_tokens, dim)

        self.transformer_layers = nn.ModuleList(transformer_layers)

        if final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim, self.total_tokens * 2, self.total_tokens])
            self.mlp2 = simple_MLP([dim, num_continuous, 1])
        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(dim, num_continuous, torch.ones(num_continuous).int())

        proj_in = dim * (num_continuous + self.num_categories)
        _proj_hidden = proj_hidden if proj_hidden is not None else 3 * proj_in // 4
        _proj_out = proj_out if proj_out is not None else proj_in // 2
        self.pt_mlp1 = simple_MLP([proj_in, _proj_hidden, _proj_out])
        self.pt_mlp2 = simple_MLP([proj_in, _proj_hidden, _proj_out])

    def forward(self, x_categ, x_cont=None, mask=None):
        x = x_categ
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for block in self.transformer_layers:
            x = block(x, mask=mask)
        return x


def embed_data(
    x_categ: torch.Tensor,
    x_cont: torch.Tensor,
    encoder: 'Encoder',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Embed raw categorical indices and continuous values via the encoder's embedding layers.

    Called before the transformer forward pass; separate so augmentations can be
    applied to the embedded tokens before passing them to ``encoder.forward``.
    """
    x_categ = x_categ + encoder.categories_offset.type_as(x_categ)
    x_categ_enc = encoder.embeds(x_categ)  # (N, F_cat, dim)

    x_cont_enc: torch.Tensor | None = None
    if encoder.num_continuous > 0:
        if encoder.cont_embeddings == 'MLP':
            x = x_cont.unsqueeze(-1)
            h = F.relu(encoder.cont_fc1(x))
            out = encoder.cont_fc2(h)
            x_cont_enc = out.view(x_cont.size(0), encoder.num_continuous, encoder.dim)
        elif encoder.cont_embeddings == 'pos_singleMLP':
            x_cont_enc = encoder.cont_MLP[0](x_cont.unsqueeze(-1))

    return x_categ_enc, x_cont_enc
