"""Tests for fm4tag.models.aggregator.TransformerAggregator.

The key contract: the aggregated jet embedding is a function of the *valid*
constituents only.  Neither the amount of padding nor the values stored in
padded slots may influence the output.
"""

from __future__ import annotations

import pytest
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


def _padded_reference(agg, z_global, z, valid):
    """The pre-packing implementation: padded grid + key mask + masked mean."""
    z = torch.where(valid.unsqueeze(-1), z, 0.0)
    z = agg.const_transformer[0](z, valid)
    z = torch.where(valid.unsqueeze(-1), z, 0.0)
    pooled = z.sum(dim=1) / valid.sum(dim=1, keepdim=True).float().clamp(min=1.0)
    return torch.cat([z_global, pooled], dim=-1)


@pytest.mark.parametrize('packed_input', [True, False])
def test_packed_aggregator_matches_padded_reference(packed_input):
    agg = _make_aggregator().eval()
    torch.manual_seed(3)
    B, C = 9, 7
    valid = torch.rand(B, C) < 0.5
    valid[2] = False  # a jet without constituents
    z_global = torch.randn(B, _GLOBAL_DIM)
    z = torch.randn(B, C, _CONST_DIM)
    with torch.no_grad():
        ref = _padded_reference(agg, z_global, z, valid)
        out = agg(z_global, [z[valid] if packed_input else z], [valid])
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.count_nonzero(out[2, _GLOBAL_DIM:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
def test_packed_aggregator_bf16_gpu_matches_fp32_reference():
    agg = _make_aggregator().cuda()
    torch.manual_seed(4)
    B, C = 64, 40
    valid = torch.rand(B, C, device='cuda') < 0.18
    valid[0] = False
    z_global = torch.randn(B, _GLOBAL_DIM, device='cuda')
    z = torch.randn(B, C, _CONST_DIM, device='cuda')
    with torch.no_grad():
        ref = _padded_reference(agg, z_global, z, valid)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = agg(z_global, [z[valid]], [valid])
    assert torch.allclose(out.float(), ref, atol=5e-2, rtol=5e-2)


def test_attention_and_ff_dropout_act_only_in_train_mode():
    torch.manual_seed(0)
    agg = TransformerAggregator(
        global_dim=_GLOBAL_DIM,
        const_dims=[_CONST_DIM],
        depth=2,
        heads=2,
        dim_head=4,
        ff_mult=1,
        ff_dropout=0.3,
        attn_dropout=0.3,
    )
    B, C = 6, 12
    valid = torch.rand(B, C) > 0.4
    valid[0] = False  # a jet without constituents
    valid[1, 0] = True
    args = (torch.randn(B, _GLOBAL_DIM), [torch.randn(B, C, _CONST_DIM)], [valid])

    agg.eval()
    with torch.no_grad():
        eval_out = agg(*args)
        assert torch.equal(eval_out, agg(*args))

    agg.train()
    torch.manual_seed(1)
    out = agg(*args)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, eval_out, atol=1e-3)
    assert torch.count_nonzero(out[0, _GLOBAL_DIM:]) == 0  # empty jet pools to zero
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in agg.parameters())
