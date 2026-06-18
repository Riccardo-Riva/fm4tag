"""Tests for model instantiation via dict-based layer config."""

from __future__ import annotations

import torch

from fm4tag.models import (
    ColTransformer,
    Encoder,
    GlobalEncoder,
    GlobalTransformerEncoder,
    RowColTransformer,
)


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


# ---------------------------------------------------------------------------
# Test 3: GlobalTransformerEncoder matches the GlobalEncoder interface
# ---------------------------------------------------------------------------

_N_GLOBAL = 2
_FEATURE_DIM = 16


def test_global_transformer_encoder_builds_and_runs():
    """GlobalTransformerEncoder is a drop-in replacement for GlobalEncoder."""
    model = GlobalTransformerEncoder(
        num_features=_N_GLOBAL,
        feature_dim=_FEATURE_DIM,
        dim=_DIM,
        layers=[{'type': 'col', 'depth': 2, 'heads': 2, 'dim_head': 8}],
    )
    assert isinstance(model.layers[0], ColTransformer)

    # Continuous-only: embed splits embedding from the attention forward.
    x_cont = torch.randn(_B, _N_GLOBAL)
    x_cat_enc, x_con_enc = model.embed(torch.zeros(_B, 0, dtype=torch.long), x_cont)
    assert x_cat_enc is None
    out = model(x_cat_enc, x_con_enc)
    assert out.shape == (_B, _N_GLOBAL, _FEATURE_DIM)
    assert not out.isnan().any()

    # Same heads as GlobalEncoder, used identically by the pretrain module.
    z = model.projector(out.flatten(1, 2))
    assert z.shape == (_B, _DIM)
    rec = torch.cat(model.reconstructor(out), dim=1)
    assert rec.shape == (_B, _N_GLOBAL)


def test_global_encoder_with_categorical_features():
    """GlobalEncoder embeds + reconstructs categorical jet features like Encoder."""
    categories = [3, 5]
    model = GlobalTransformerEncoder(
        num_features=_N_GLOBAL,
        feature_dim=_FEATURE_DIM,
        dim=_DIM,
        layers=[{'type': 'col', 'depth': 1, 'heads': 2, 'dim_head': 8}],
        categories=categories,
    )

    x_categ = torch.stack([torch.randint(0, n, (_B,)) for n in categories], dim=1)
    x_cont = torch.randn(_B, _N_GLOBAL)
    x_cat_enc, x_con_enc = model.embed(x_categ, x_cont)
    assert x_cat_enc.shape == (_B, len(categories), _FEATURE_DIM)
    assert x_con_enc.shape == (_B, _N_GLOBAL, _FEATURE_DIM)

    # Tokens are ordered [categorical; continuous].
    out = model(x_cat_enc, x_con_enc)
    n_tokens = len(categories) + _N_GLOBAL
    assert out.shape == (_B, n_tokens, _FEATURE_DIM)

    z = model.projector(out.flatten(1, 2))
    assert z.shape == (_B, _DIM)

    cat_outs = model.cat_reconstructor(out[:, : len(categories), :])
    assert len(cat_outs) == len(categories)
    for logits, n in zip(cat_outs, categories):
        assert logits.shape == (_B, n)
    con_outs = model.reconstructor(out[:, len(categories):, :])
    assert torch.cat(con_outs, dim=1).shape == (_B, _N_GLOBAL)


def test_global_encoder_selected_via_hydra_target():
    """Both global encoder classes instantiate via _target_, as in the engine."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    ff_cfg = OmegaConf.create(
        {
            '_target_': 'fm4tag.models.GlobalEncoder',
            'feature_dim': _FEATURE_DIM,
            'dim': _DIM,
        }
    )
    tr_cfg = OmegaConf.create(
        {
            '_target_': 'fm4tag.models.GlobalTransformerEncoder',
            'feature_dim': _FEATURE_DIM,
            'dim': _DIM,
            'layers': [{'type': 'col', 'depth': 1, 'heads': 2, 'dim_head': 8}],
        }
    )

    ff_enc = instantiate(ff_cfg, num_features=_N_GLOBAL)
    tr_enc = instantiate(tr_cfg, num_features=_N_GLOBAL)
    assert type(ff_enc) is GlobalEncoder
    assert type(tr_enc) is GlobalTransformerEncoder

    x_cont = torch.randn(_B, _N_GLOBAL)
    empty_cat = torch.zeros(_B, 0, dtype=torch.long)
    ff_cat, ff_con = ff_enc.embed(empty_cat, x_cont)
    tr_cat, tr_con = tr_enc.embed(empty_cat, x_cont)
    assert (
        ff_enc(ff_cat, ff_con).shape
        == tr_enc(tr_cat, tr_con).shape
        == (_B, _N_GLOBAL, _FEATURE_DIM)
    )
