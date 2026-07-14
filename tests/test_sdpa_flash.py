"""Tests for the shared `sdpa` helper (FlashAttention-2 fast path + SDPA fallback)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

import fm4tag.models.attention as attention_mod
from fm4tag.models.attention import Attention, sdpa


def test_cpu_no_mask_matches_direct_sdpa():
    """Off-CUDA must always fall back, regardless of flash-attn availability."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    out = sdpa(q, k, v)
    assert torch.equal(out, expected)


def test_cpu_with_mask_matches_direct_sdpa():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)
    mask = torch.zeros(2, 1, 1, 6)
    mask[:, :, :, :3] = float('-inf')
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    out = sdpa(q, k, v, attn_mask=mask)
    assert torch.equal(out, expected)


def test_masked_call_falls_back_even_if_flash_attn_available():
    """A non-None attn_mask must always take the SDPA path, even with flash-attn 'available'."""
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)
    mask = torch.zeros(2, 1, 1, 6)

    flash_fn = MagicMock(
        side_effect=AssertionError('flash_attn_func should not be called')
    )
    with (
        patch.object(attention_mod, 'HAS_FLASH_ATTN', True),
        patch.object(attention_mod, 'flash_attn_func', flash_fn),
    ):
        out = sdpa(q, k, v, attn_mask=mask)
    flash_fn.assert_not_called()
    assert out.shape == (2, 4, 6, 8)


def test_cpu_tensor_falls_back_even_if_flash_attn_available():
    """CPU tensors must always take the SDPA path, even with flash-attn 'available'."""
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)

    flash_fn = MagicMock(
        side_effect=AssertionError('flash_attn_func should not be called')
    )
    with (
        patch.object(attention_mod, 'HAS_FLASH_ATTN', True),
        patch.object(attention_mod, 'flash_attn_func', flash_fn),
    ):
        out = sdpa(q, k, v)
    flash_fn.assert_not_called()
    assert out.shape == (2, 4, 6, 8)


# (q_shape, kv_shape) — equal for self-attention, different sequence lengths for
# the cross-attention inside InducedRowAttention.
_SHAPES = [
    # Attention (column), self: (b, h, n, d)
    ((2, 4, 6, 8), (2, 4, 6, 8)),
    # RowAttention, self over the sample axis: (n, h, B, d)
    ((5, 4, 10, 8), (5, 4, 10, 8)),
    # CrossAttention stage 1 — inducing points read the batch: q seq = num_inds,
    # kv seq = B.
    ((5, 4, 6, 8), (5, 4, 10, 8)),
    # CrossAttention stage 2 — samples read the summary: q seq = B,
    # kv seq = num_inds.
    ((5, 4, 10, 8), (5, 4, 6, 8)),
]


@pytest.mark.parametrize('q_shape,kv_shape', _SHAPES)
def test_fallback_handles_cross_attention_shapes(q_shape, kv_shape):
    """The SDPA fallback (what runs without flash-attn) must accept seq_q != seq_kv."""
    torch.manual_seed(0)
    q = torch.randn(*q_shape)
    k = torch.randn(*kv_shape)
    v = torch.randn(*kv_shape)
    out = sdpa(q, k, v)
    assert out.shape == q_shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
@pytest.mark.skipif(
    not attention_mod.HAS_FLASH_ATTN, reason='requires flash-attn installed'
)
@pytest.mark.parametrize('q_shape,kv_shape', _SHAPES)
def test_flash_path_matches_sdpa_on_gpu(q_shape, kv_shape):
    torch.manual_seed(0)
    device = torch.device('cuda')
    dtype = torch.float16
    q = torch.randn(*q_shape, device=device, dtype=dtype)
    k = torch.randn(*kv_shape, device=device, dtype=dtype)
    v = torch.randn(*kv_shape, device=device, dtype=dtype)

    out_flash = sdpa(q, k, v)
    out_ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    assert out_flash.shape == q_shape
    assert torch.allclose(out_flash, out_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
@pytest.mark.skipif(
    not attention_mod.HAS_FLASH_ATTN, reason='requires flash-attn installed'
)
def test_attention_module_matches_between_flash_and_sdpa_backends():
    """End-to-end: Attention module output should agree whether or not flash-attn is used."""
    torch.manual_seed(0)
    device = torch.device('cuda')
    attn = (
        Attention(dim=32, heads=4, dim_head=8)
        .to(device=device, dtype=torch.float16)
        .eval()
    )
    x = torch.randn(4, 10, 32, device=device, dtype=torch.float16)

    with torch.no_grad():
        with patch.object(attention_mod, 'HAS_FLASH_ATTN', True):
            out_flash = attn(x)
        with patch.object(attention_mod, 'HAS_FLASH_ATTN', False):
            out_sdpa = attn(x)

    assert torch.allclose(out_flash, out_sdpa, atol=2e-2, rtol=2e-2)
