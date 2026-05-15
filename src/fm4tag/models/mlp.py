from torch import nn


class MLP(nn.Module):
    """Generic MLP with optional activation between layers (no activation on the last layer).

    Args:
        dims: List of layer sizes, e.g. ``[input, hidden, output]``.
        act:  Activation module inserted between layers. ``None`` = linear stack.
    """

    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= len(dims_pairs) - 1
            layers.append(nn.Linear(dim_in, dim_out))
            if not is_last and act is not None:
                layers.append(act)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_dropout(nn.Module):
    """MLP with dropout between hidden layers (not applied after the last layer).

    Args:
        dims:    List of layer sizes.
        act:     Activation module. ``None`` = no activation.
        dropout: Dropout probability applied after each hidden activation.
    """

    def __init__(self, dims, act=None, dropout=0.0):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= len(dims_pairs) - 1
            layers.append(nn.Linear(dim_in, dim_out))
            if not is_last:
                if act is not None:
                    layers.append(act)
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    """Three-layer MLP: ``dims[0] → dims[1] → dims[2]`` with ReLU in between."""

    def __init__(self, dims):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
        )

    def forward(self, x):
        return self.layers(x)


class sep_MLP(nn.Module):
    """Per-feature separate MLPs, used for reconstruction of categorical tokens.

    Each of the ``len_feats`` features gets its own ``simple_MLP`` that maps
    ``dim → 2*dim → categories[i]``.

    Args:
        dim:       Input embedding dimension.
        len_feats: Number of independent feature heads.
        categories: List of output class counts per feature.
    """

    def __init__(self, dim, len_feats, categories):
        super().__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList(
            [simple_MLP([dim, 2 * dim, categories[i]]) for i in range(len_feats)]
        )

    def forward(self, x):
        return [self.layers[i](x[:, i, :]) for i in range(self.len_feats)]
