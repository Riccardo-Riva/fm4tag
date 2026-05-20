"""Tests for model instantiation via dict-based layer config."""

from __future__ import annotations

import torch

from fm4tag.models import Encoder, ColTransformer, RowColTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CATEGORIES = [3, 5]  # 2 categorical features with cardinality 3 and 5
_NUM_CONTINUOUS = 4
_NFEATS = len(_CATEGORIES) + _NUM_CONTINUOUS  # = 6
_DIM = 32
_B = 8


def _make_tokens():
    """Return dummy pre-embedded token tensors ``(B, N_cat, dim)`` / ``(B, N_con, dim)``."""
    x_categ = torch.randn(_B, len(_CATEGORIES), _DIM)
    x_cont = torch.randn(_B, _NUM_CONTINUOUS, _DIM)
    return x_categ, x_cont


# ---------------------------------------------------------------------------
# Test 1: default config builds and runs a forward pass
# ---------------------------------------------------------------------------


def test_default_config_builds_and_runs():
    """Encoder builds from dict-based layer config and runs forward correctly."""
    model = Encoder(
        categories=_CATEGORIES,
        num_continuous=_NUM_CONTINUOUS,
        dim=_DIM,
        proj_hidden=64,
        proj_out=32,
        layers=[
            {'type': 'col', 'depth': 2, 'heads': 2, 'dim_head': 8},
        ],
    )

    x_categ, x_cont = _make_tokens()
    out = model(x_categ, x_cont)

    assert out.shape == (_B, _NFEATS, _DIM)
    assert not out.isnan().any()


# ---------------------------------------------------------------------------
# Test 2: swapping a block type produces the expected class and runs forward
# ---------------------------------------------------------------------------


def test_layer_swap_produces_correct_block_type_and_runs():
    """Changing 'type' in the layers list produces the expected block class."""
    model_col = Encoder(
        categories=_CATEGORIES,
        num_continuous=_NUM_CONTINUOUS,
        dim=_DIM,
        proj_hidden=64,
        proj_out=32,
        layers=[{'type': 'col', 'heads': 2, 'dim_head': 8}],
    )
    model_rowcol = Encoder(
        categories=_CATEGORIES,
        num_continuous=_NUM_CONTINUOUS,
        dim=_DIM,
        proj_hidden=64,
        proj_out=32,
        layers=[
            {
                'type': 'rowcol',
                'col_heads': 2,
                'row_heads': 2,
                'dim_head': 8,
                'dim_row_head': 8,
            }
        ],
    )

    assert isinstance(model_col.layers[0], ColTransformer)
    assert isinstance(model_rowcol.layers[0], RowColTransformer)

    x_categ, x_cont = _make_tokens()
    out_col = model_col(x_categ, x_cont)
    out_rowcol = model_rowcol(x_categ, x_cont)

    assert out_col.shape == (_B, _NFEATS, _DIM)
    assert out_rowcol.shape == (_B, _NFEATS, _DIM)
    assert not out_col.isnan().any()
    assert not out_rowcol.isnan().any()
