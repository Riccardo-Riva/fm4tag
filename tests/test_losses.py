"""Tests for fm4tag.losses.MultiViewSupConLoss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fm4tag.losses import MultiViewSupConLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_views(N: int, D: int, V: int, seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    return [torch.randn(N, D) for _ in range(V)]


# ---------------------------------------------------------------------------
# Basic shape / value tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('V', [2, 3, 4])
def test_output_is_scalar(V):
    loss_fn = MultiViewSupConLoss(temperature=0.1)
    zs = _make_views(8, 32, V)
    loss = loss_fn(zs)
    assert loss.shape == ()


def test_requires_at_least_two_views():
    loss_fn = MultiViewSupConLoss()
    with pytest.raises(ValueError, match='at least 2'):
        loss_fn([torch.randn(4, 16)])


def test_loss_is_finite():
    loss_fn = MultiViewSupConLoss(temperature=0.07)
    zs = _make_views(16, 64, 3)
    assert torch.isfinite(loss_fn(zs))


def test_loss_is_non_negative():
    # SupCon loss = -log(p) where p ∈ (0,1] → loss ≥ 0.
    loss_fn = MultiViewSupConLoss(temperature=0.1)
    for seed in range(5):
        zs = _make_views(8, 32, 2, seed=seed)
        assert loss_fn(zs).item() >= -1e-6


# ---------------------------------------------------------------------------
# Loss type: L_out vs L_in
# ---------------------------------------------------------------------------

def test_loss_out_and_in_differ_on_random_input():
    zs = _make_views(8, 32, 3, seed=42)
    l_out = MultiViewSupConLoss(loss_type='out')(zs)
    l_in  = MultiViewSupConLoss(loss_type='in')(zs)
    assert not torch.isclose(l_out, l_in)


def test_invalid_loss_type():
    with pytest.raises(ValueError, match="loss_type must be"):
        MultiViewSupConLoss(loss_type='bad')


# ---------------------------------------------------------------------------
# include_pos_in_denom flag
# ---------------------------------------------------------------------------

def test_include_pos_in_denom_changes_loss():
    zs = _make_views(8, 32, 3, seed=0)
    l_with  = MultiViewSupConLoss(include_pos_in_denom=True)(zs)
    l_without = MultiViewSupConLoss(include_pos_in_denom=False)(zs)
    assert not torch.isclose(l_with, l_without)


# ---------------------------------------------------------------------------
# Collapse sensitivity: identical views should give near-zero gradients
# (perfectly aligned views still produce a valid but low loss)
# ---------------------------------------------------------------------------

def test_perfectly_aligned_views_low_loss():
    # All views are identical → all positives are perfectly aligned.
    # The loss should be lower than with random misaligned views.
    z = F.normalize(torch.randn(16, 64), dim=-1)
    zs_aligned = [z] * 3
    zs_random  = _make_views(16, 64, 3, seed=7)

    loss_aligned = MultiViewSupConLoss(temperature=0.1)(zs_aligned).item()
    loss_random  = MultiViewSupConLoss(temperature=0.1)(zs_random).item()
    assert loss_aligned < loss_random


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow_to_all_views():
    zs = [torch.randn(8, 32, requires_grad=True) for _ in range(3)]
    loss = MultiViewSupConLoss()(zs)
    loss.backward()
    for i, z in enumerate(zs):
        assert z.grad is not None, f"No gradient for view {i}"
        assert z.grad.abs().sum() > 0, f"Zero gradient for view {i}"


# ---------------------------------------------------------------------------
# DDP-fallback path (variable N across ranks)
# ---------------------------------------------------------------------------

def test_no_gather_when_single_process():
    # Without DDP, _all_gather_with_grad should return (z, 0, N).
    z = torch.randn(12, 32)
    z_out, start, end = MultiViewSupConLoss._all_gather_with_grad(z)
    assert torch.equal(z_out, z)
    assert start == 0
    assert end == 12


# ---------------------------------------------------------------------------
# Symmetry: loss should not depend on view ordering for L_out
# ---------------------------------------------------------------------------

def test_loss_out_invariant_to_view_order():
    torch.manual_seed(1)
    z0 = torch.randn(8, 32)
    z1 = torch.randn(8, 32)
    loss_fn = MultiViewSupConLoss(loss_type='out', temperature=0.1)
    l_01 = loss_fn([z0, z1])
    l_10 = loss_fn([z1, z0])
    assert torch.isclose(l_01, l_10, atol=1e-5)
