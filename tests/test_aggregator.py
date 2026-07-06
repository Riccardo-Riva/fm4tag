"""Tests for fm4tag.models.aggregator.TransformerAggregator.

The key contract: the aggregated jet embedding is a function of the *valid*
constituents only.  Neither the amount of padding nor the values stored in
padded slots may influence the output.
"""

from __future__ import annotations

import torch

from fm4tag.models import TransformerAggregator

_GLOBAL_DIM = 5
_CONST_DIM = 8


def _make_aggregator() -> TransformerAggregator:
    torch.manual_seed(0)
    return TransformerAggregator(
        global_dim=_GLOBAL_DIM,
        const_dims=[_CONST_DIM],
        depth=2,
        heads=2,
        dim_head=4,
        ff_mult=1,
    ).eval()


def test_output_shape():
    agg = _make_aggregator()
    B, C = 4, 6
    out = agg(
        torch.randn(B, _GLOBAL_DIM),
        [torch.randn(B, C, _CONST_DIM)],
        [torch.ones(B, C, dtype=torch.bool)],
    )
    assert out.shape == (B, agg.out_dim)
    assert agg.out_dim == _GLOBAL_DIM + _CONST_DIM


def test_padding_amount_invariance():
    """Adding extra all-invalid slots must not change the jet embedding."""
    agg = _make_aggregator()
    B, C, pad = 4, 6, 10
    z_global = torch.randn(B, _GLOBAL_DIM)
    z_const = torch.randn(B, C, _CONST_DIM)
    valid = torch.zeros(B, C, dtype=torch.bool)
    valid[:, :3] = True

    with torch.no_grad():
        out = agg(z_global, [z_const], [valid])
        out_padded = agg(
            z_global,
            [torch.cat([z_const, torch.randn(B, pad, _CONST_DIM)], dim=1)],
            [torch.cat([valid, torch.zeros(B, pad, dtype=torch.bool)], dim=1)],
        )

    assert torch.allclose(out, out_padded, atol=1e-6)


def test_padding_value_invariance():
    """Garbage values in invalid slots must not change the jet embedding."""
    agg = _make_aggregator()
    B, C = 4, 6
    z_global = torch.randn(B, _GLOBAL_DIM)
    z_const = torch.randn(B, C, _CONST_DIM)
    valid = torch.zeros(B, C, dtype=torch.bool)
    valid[:, :2] = True

    z_garbage = z_const.clone()
    z_garbage[~valid] = 1e6

    with torch.no_grad():
        out = agg(z_global, [z_const], [valid])
        out_garbage = agg(z_global, [z_garbage], [valid])

    assert torch.allclose(out, out_garbage, atol=1e-6)


def test_all_invalid_jet_pools_to_zero():
    """A jet with zero valid constituents contributes a zero pooled vector
    (and stays finite), while other jets in the batch are unaffected."""
    agg = _make_aggregator()
    B, C = 4, 6
    z_global = torch.randn(B, _GLOBAL_DIM)
    z_const = torch.randn(B, C, _CONST_DIM)
    valid = torch.ones(B, C, dtype=torch.bool)
    valid[0] = False  # first jet: no valid constituents

    with torch.no_grad():
        out = agg(z_global, [z_const], [valid])

    assert torch.isfinite(out).all()
    # Constituent part of jet 0 is exactly zero; global part passes through.
    assert torch.equal(out[0, :_GLOBAL_DIM], z_global[0])
    assert torch.equal(out[0, _GLOBAL_DIM:], torch.zeros(_CONST_DIM))
