"""Tests for intersample (row) attention — plain and chunked variants."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from fm4tag.models.attention import ChunkedRowAttention, RowAttention


# ---------------------------------------------------------------------------
# Test 1: chunk_size == B in eval mode matches whole-batch RowAttention
# ---------------------------------------------------------------------------


def test_chunk_eq_B_matches_no_chunk_eval():
    """ChunkedRowAttention(chunk_size=B) in eval ⇒ single group ⇒ matches RowAttention."""
    torch.manual_seed(0)
    B, dim = 8, 32

    attn_ref = RowAttention(dim, heads=4, dim_row_head=8, dropout=0.0).eval()
    attn_chunk = ChunkedRowAttention(
        dim, heads=4, dim_row_head=8, dropout=0.0, chunk_size=B
    ).eval()
    attn_chunk.load_state_dict(attn_ref.state_dict())

    x = torch.randn(B, dim)
    with torch.no_grad():
        y_ref = attn_ref(x)
        y_chunk = attn_chunk(x)

    assert torch.allclose(y_ref, y_chunk, atol=1e-6), (
        f'max diff: {(y_ref - y_chunk).abs().max().item():.2e}'
    )


# ---------------------------------------------------------------------------
# Test 2: train-mode inv-permutation correctness
# ---------------------------------------------------------------------------


def test_train_mode_inv_perm_restores_order():
    """With a fixed internal perm P, train forward(x) equals eval forward(x[P])[inv_P]."""
    torch.manual_seed(0)
    B, dim, chunk_size = 8, 32, 4

    attn = ChunkedRowAttention(
        dim, heads=4, dim_row_head=8, dropout=0.0, chunk_size=chunk_size
    )
    attn.train()

    x = torch.randn(B, dim)

    # Pre-generate the permutation we will inject via monkeypatch.
    fixed_perm = torch.randperm(B)
    inv_perm = torch.empty_like(fixed_perm)
    inv_perm[fixed_perm] = torch.arange(B)

    # Train-mode forward with the fixed internal perm.
    with torch.no_grad():
        with patch('torch.randperm', return_value=fixed_perm):
            y_train = attn(x)

    # Reference: eval mode on x[fixed_perm] (contiguous chunks = same grouping),
    # then un-permute.
    attn.eval()
    with torch.no_grad():
        y_eval_on_perm = attn(x[fixed_perm])
    expected = y_eval_on_perm[inv_perm]

    assert torch.allclose(y_train, expected, atol=1e-5), (
        f'max diff: {(y_train - expected).abs().max().item():.2e}'
    )


# ---------------------------------------------------------------------------
# Test 3: shapes and no NaNs for various B / chunk_size combos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'B,chunk_size',
    [
        (12, 4),  # B divisible by chunk_size
        (10, 4),  # B % chunk_size != 0  → padding
        (4, 8),  # chunk_size >= B      → single padded group
    ],
)
def test_shapes_and_no_nans(B: int, chunk_size: int):
    dim = 32
    attn = ChunkedRowAttention(
        dim, heads=4, dim_row_head=8, dropout=0.0, chunk_size=chunk_size
    )
    attn.train()
    x = torch.randn(B, dim)
    out = attn(x)
    assert out.shape == (B, dim), (
        f'B={B}, chunk={chunk_size}: expected ({B},{dim}), got {out.shape}'
    )
    assert not out.isnan().any(), f'B={B}, chunk={chunk_size}: NaN in output'


def test_chunked_requires_chunk_size():
    """ChunkedRowAttention must be given a chunk_size."""
    with pytest.raises(ValueError):
        ChunkedRowAttention(32, heads=4, dim_row_head=8, dropout=0.0)


# ---------------------------------------------------------------------------
# Test 4: the internal padding mask makes padded rows invisible
# ---------------------------------------------------------------------------


def test_padding_masked_matches_per_group_whole_batch():
    """A padded (incomplete) group must equal whole-batch attention over only
    its valid members — i.e. the zero-padding does not leak into the softmax."""
    torch.manual_seed(0)
    B, dim, chunk_size = 6, 32, 4  # eval groups: [0:4] full, [4:6] padded

    ref = RowAttention(dim, heads=4, dim_row_head=8, dropout=0.0).eval()
    chunk = ChunkedRowAttention(
        dim, heads=4, dim_row_head=8, dropout=0.0, chunk_size=chunk_size
    ).eval()
    chunk.load_state_dict(ref.state_dict())

    x = torch.randn(B, dim)
    with torch.no_grad():
        y_chunk = chunk(x)
        # eval mode uses contiguous groups; run each group's valid rows alone.
        y_ref = torch.cat([ref(x[0:4]), ref(x[4:6])], dim=0)

    assert torch.allclose(y_chunk, y_ref, atol=1e-6), (
        f'max diff: {(y_chunk - y_ref).abs().max().item():.2e}'
    )


def test_padding_mask_single_group_matches_whole_batch():
    """chunk_size >= B (single padded group) must equal whole-batch attention."""
    torch.manual_seed(0)
    B, dim, chunk_size = 6, 32, 8

    ref = RowAttention(dim, heads=4, dim_row_head=8, dropout=0.0).eval()
    chunk = ChunkedRowAttention(
        dim, heads=4, dim_row_head=8, dropout=0.0, chunk_size=chunk_size
    ).eval()
    chunk.load_state_dict(ref.state_dict())

    x = torch.randn(B, dim)
    with torch.no_grad():
        y_ref = ref(x)
        y_chunk = chunk(x)

    assert torch.allclose(y_ref, y_chunk, atol=1e-6), (
        f'max diff: {(y_ref - y_chunk).abs().max().item():.2e}'
    )
