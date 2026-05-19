"""Tests for model instantiation via Hydra _target_ — Task B."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from fm4tag.models import ColBlock, RowColBlock


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


def _col_encoder_cfg(n_layers: int = 2):
    return OmegaConf.create(
        {
            '_target_': 'fm4tag.models.Encoder',
            'dim': _DIM,
            'cont_embeddings': 'MLP',
            'final_mlp_style': 'sep',
            'proj_hidden': 64,
            'proj_out': 32,
            'transformer_layers': [
                {
                    '_target_': 'fm4tag.models.ColBlock',
                    'dim': _DIM,
                    'heads': 2,
                    'dim_head': 8,
                    'ff_mult': 1,
                    'attn_dropout': 0.0,
                    'ff_dropout': 0.0,
                }
            ]
            * n_layers,
        }
    )


def _rowcol_encoder_cfg():
    return OmegaConf.create(
        {
            '_target_': 'fm4tag.models.Encoder',
            'dim': _DIM,
            'cont_embeddings': 'MLP',
            'final_mlp_style': 'sep',
            'proj_hidden': 64,
            'proj_out': 32,
            'transformer_layers': [
                {
                    '_target_': 'fm4tag.models.RowColBlock',
                    'dim': _DIM,
                    'nfeats': _NFEATS,
                    'col_heads': 2,
                    'row_heads': 2,
                    'dim_head': 8,
                    'dim_row_head': 8,
                    'ff_mult': 1,
                    'attn_dropout': 0.0,
                    'ff_dropout': 0.0,
                    'chunk_size': None,
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: default config builds and runs a forward pass
# ---------------------------------------------------------------------------


def test_default_config_builds_and_runs():
    """Default encoder config builds via instantiate and runs forward correctly."""
    cfg = _col_encoder_cfg(n_layers=2)
    model = instantiate(cfg, categories=_CATEGORIES, num_continuous=_NUM_CONTINUOUS)

    x_categ, x_cont = _make_tokens()
    out = model(x_categ, x_cont)

    assert out.shape == (_B, len(_CATEGORIES) + _NUM_CONTINUOUS, _DIM)
    assert not out.isnan().any()


# ---------------------------------------------------------------------------
# Test 2: swapping a block type via config override
# ---------------------------------------------------------------------------


def test_layer_swap_produces_correct_block_type_and_runs():
    """Swapping _target_ in transformer_layers produces the expected block and runs."""
    model_col = instantiate(
        _col_encoder_cfg(n_layers=1),
        categories=_CATEGORIES,
        num_continuous=_NUM_CONTINUOUS,
    )
    model_rowcol = instantiate(
        _rowcol_encoder_cfg(),
        categories=_CATEGORIES,
        num_continuous=_NUM_CONTINUOUS,
    )

    # Verify the swapped block type.
    assert isinstance(model_col.transformer_layers[0], ColBlock)
    assert isinstance(model_rowcol.transformer_layers[0], RowColBlock)

    x_categ, x_cont = _make_tokens()
    out_col = model_col(x_categ, x_cont)
    out_rowcol = model_rowcol(x_categ, x_cont)

    expected_shape = (_B, len(_CATEGORIES) + _NUM_CONTINUOUS, _DIM)
    assert out_col.shape == expected_shape
    assert out_rowcol.shape == expected_shape
    assert not out_col.isnan().any()
    assert not out_rowcol.isnan().any()
