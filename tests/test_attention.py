"""Tests for fm4tag.models.attention.RowAttention masking."""

from __future__ import annotations

import torch

from fm4tag.models.attention import RowAttention


def _make_attn(heads: int = 4) -> RowAttention:
    torch.manual_seed(0)
    return RowAttention(dim=16, heads=heads, dim_row_head=8).eval()


def test_mask_shape_broadcasts():
    """Regression: (B,) mask must work when B != heads."""
    attn = _make_attn(heads=4)
    x = torch.randn(10, 16)
    mask = torch.ones(10, dtype=torch.bool)
    mask[7:] = False
    out = attn(x, mask=mask)
    assert out.shape == (10, 16)
    assert torch.isfinite(out).all()


def test_masked_rows_are_ignored_as_keys():
    """Garbage in masked rows must not change valid rows' outputs."""
    attn = _make_attn()
    x = torch.randn(10, 16)
    mask = torch.ones(10, dtype=torch.bool)
    mask[7:] = False

    x_garbage = x.clone()
    x_garbage[7:] = 1e4

    with torch.no_grad():
        out = attn(x, mask=mask)
        out_garbage = attn(x_garbage, mask=mask)

    assert torch.allclose(out[:7], out_garbage[:7], atol=1e-5)


def test_all_valid_mask_matches_no_mask():
    attn = _make_attn()
    x = torch.randn(10, 16)
    with torch.no_grad():
        out_none = attn(x)
        out_all = attn(x, mask=torch.ones(10, dtype=torch.bool))
    assert torch.allclose(out_none, out_all, atol=1e-6)
