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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
@pytest.mark.skipif(
    not attention_mod.HAS_FLASH_ATTN, reason='requires flash-attn installed'
)
@pytest.mark.parametrize(
    'shape',
    [
        (2, 4, 6, 8),  # Attention-style: (b, h, n, d)
        (4, 6, 8),  # RowAttention-style: (h, seq, d), no batch dim
        (3, 4, 5, 8),  # ChunkedRowAttention-style: (g, h, c, d)
    ],
)
def test_flash_path_matches_sdpa_on_gpu(shape):
    torch.manual_seed(0)
    device = torch.device('cuda')
    dtype = torch.float16
    q = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(*shape, device=device, dtype=dtype)
    v = torch.randn(*shape, device=device, dtype=dtype)

    out_flash = sdpa(q, k, v)
    out_ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
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
