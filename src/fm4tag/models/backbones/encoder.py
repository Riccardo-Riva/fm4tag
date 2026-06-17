from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from ..mlp import sep_MLP, simple_MLP
from .transformers import ColTransformer, RowColTransformer, RowTransformer

_LAYER_TYPES: dict[str, type[nn.Module]] = {
    'col': ColTransformer,
    'row': RowTransformer,
    'rowcol': RowColTransformer,
}


def _build_layers(layers: list, *, dim: int, nfeats: int) -> nn.ModuleList:
    """Build transformer blocks from a declarative layer-config list.

    Each entry is a dict with a ``type`` key (``col`` | ``row`` | ``rowcol``)
    selecting the block class plus its keyword arguments.  ``dim`` and (for
    row-aware blocks) ``nfeats`` are injected here and must not appear in the
    layer dicts.
    """
    built: list[nn.Module] = []
    for lcfg in layers:
        kwargs = {k: v for k, v in lcfg.items()}
        layer_type = kwargs.pop('type')
        cls = _LAYER_TYPES.get(layer_type)
        if cls is None:
            raise ValueError(
                f'Unknown layer type {layer_type!r}. Choose from: {list(_LAYER_TYPES)}'
            )
        if layer_type in ('row', 'rowcol'):
            kwargs['nfeats'] = nfeats
        built.append(cls(dim=dim, **kwargs))
    return nn.ModuleList(built)


class GlobalEncoder(nn.Module):
    """Per-feature MLP encoder for global flat continuous features.

    Each of the ``num_features`` continuous features is independently projected
    to a ``dim``-dimensional embedding via a 2-layer grouped Conv1d MLP.  No
    attention is applied — the output is a plain set of token embeddings, one
    per global feature.

    Heads
    -----
    * ``projector``     — MLP mapping ``(num_features * dim)`` → projection space,
                          shared across all views for the contrastive loss.
    * ``reconstructor`` — per-feature sep_MLP for denoising reconstruction (MSE).

    Args:
        num_features: Number of global continuous features ``F_g``.
        dim:          Embedding dimension.
    """

    def __init__(
        self,
        num_features: int,
        feature_dim: int,  # injected at runtime from variables.jets.inputs
        dim: int,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        self.feature_dim = feature_dim

        H = 2 * self.feature_dim
        self.fc1 = nn.Conv1d(
            num_features, num_features * H, kernel_size=1, groups=num_features
        )
        self.fc2 = nn.Conv1d(
            num_features * H,
            num_features * self.feature_dim,
            kernel_size=1,
            groups=num_features,
        )

        proj_in = num_features * self.feature_dim
        self.projector = simple_MLP([proj_in, 2 * proj_in, self.dim])
        self.reconstructor = sep_MLP(self.feature_dim, num_features, [1] * num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed global features.

        Args:
            x: ``(N, F_g)`` continuous feature values (already normalised).

        Returns:
            ``(N, F_g, feature_dim)`` — one embedding token per feature.
        """
        h = F.relu(self.fc1(x.unsqueeze(-1)))  # (N, F_g*H, 1)
        out = self.fc2(h).squeeze(-1)  # (N, F_g*feature_dim)
        return out.view(x.size(0), self.num_features, self.feature_dim)


class GlobalTransformerEncoder(GlobalEncoder):
    """Transformer encoder for global flat continuous features.

    Drop-in alternative to :class:`GlobalEncoder` (same constructor arguments
    plus ``layers``, same heads, same forward signature).  After the
    per-feature MLP embedding, the ``(N, F_g, feature_dim)`` tokens are
    refined by a declarative stack of attention blocks using the same
    ``layers`` syntax as :class:`Encoder`:

    * ``type: col``    → :class:`ColTransformer`
    * ``type: row``    → :class:`RowTransformer`
    * ``type: rowcol`` → :class:`RowColTransformer`

    The attention ``dim`` is ``feature_dim`` and ``nfeats`` (for row-aware
    blocks) is ``num_features`` — both injected automatically.

    Args:
        num_features: Number of global continuous features ``F_g``.
        feature_dim:  Per-feature token embedding dimension.
        dim:          Output dimension of the contrastive projector.
        layers:       List of layer-config dicts (see :class:`Encoder`).
    """

    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        dim: int,
        layers: list,
    ) -> None:
        super().__init__(num_features=num_features, feature_dim=feature_dim, dim=dim)
        self.layers = _build_layers(layers, dim=feature_dim, nfeats=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed global features and refine them with attention.

        Args:
            x: ``(N, F_g)`` continuous feature values (already normalised).

        Returns:
            ``(N, F_g, feature_dim)`` — one embedding token per feature.
        """
        tokens = super().forward(x)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


class Encoder(nn.Module):
    """SAINT-style transformer encoder with a declarative list of backbone layers.

    The transformer backbone is configured via the ``layers`` list.  Each entry
    is a plain dict (or OmegaConf DictConfig) with a ``type`` key selecting the
    block class and any additional keyword arguments for that class:

    * ``type: col``    → :class:`ColTransformer`
    * ``type: row``    → :class:`RowTransformer`
    * ``type: rowcol`` → :class:`RowColTransformer`

    ``dim`` and ``nfeats`` (for row-aware blocks) are injected automatically —
    they must **not** appear in the layer dicts.

    Each categorical feature is embedded via a learned lookup table; each
    continuous feature is projected per-feature through a grouped Conv1d MLP.

    Heads
    -----
    * ``projector``          — shared MLP for all views' contrastive projection.
    * ``cat_reconstructor``  — sep_MLP for categorical denoising (cross-entropy).
    * ``con_reconstructor``  — sep_MLP for continuous denoising (MSE).

    Args:
        categories:         Cardinality of each categorical feature.
        num_continuous:     Number of continuous features.
        dim:                Embedding dimension.
        layers:             List of layer-config dicts.
        num_special_tokens: Extra tokens prepended to the embedding table.
        cont_embeddings:    ``'MLP'`` (grouped Conv1d) or ``'pos_singleMLP'``.
        proj_hidden:        Hidden dim of the projector. ``None`` → auto (3/4 of proj_in).
        proj_out:           Output dim of the projector. ``None`` → auto (1/2 of proj_in).
    """

    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        layers: list,
        num_special_tokens: int = 0,
        cont_embeddings: str = 'MLP',
        proj_hidden=None,
        proj_out=None,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), (
            'number of each category must be positive'
        )

        self.categories = tuple(int(c) for c in categories)
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

        nfeats = self.num_categories + num_continuous
        self.layers = _build_layers(layers, dim=dim, nfeats=nfeats)

        # Denoising reconstruction heads (always sep style).
        self.cat_reconstructor = sep_MLP(dim, self.num_categories, categories)
        self.con_reconstructor = sep_MLP(
            dim, num_continuous, torch.ones(num_continuous, dtype=torch.int)
        )

        # Contrastive projection head — shared across all views.
        proj_in = dim * (num_continuous + self.num_categories)
        _proj_hidden = proj_hidden if proj_hidden is not None else 3 * proj_in // 4
        _proj_out = proj_out if proj_out is not None else proj_in // 2
        self.projector = simple_MLP([proj_in, _proj_hidden, _proj_out])

    def forward(self, x_categ, x_cont=None, mask=None):
        x = x_categ
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for layer in self.layers:
            x = layer(x, mask=mask)
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
