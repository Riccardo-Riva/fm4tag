"""Tests for fm4tag.losses.MultiViewSupConLoss and loss-weight handling."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fm4tag.losses import MultiViewSupConLoss, normalize_loss_weights


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
    l_in = MultiViewSupConLoss(loss_type='in')(zs)
    assert not torch.isclose(l_out, l_in)


def test_invalid_loss_type():
    with pytest.raises(ValueError, match='loss_type must be'):
        MultiViewSupConLoss(loss_type='bad')


# ---------------------------------------------------------------------------
# include_pos_in_denom flag
# ---------------------------------------------------------------------------


def test_include_pos_in_denom_changes_loss():
    zs = _make_views(8, 32, 3, seed=0)
    l_with = MultiViewSupConLoss(include_pos_in_denom=True)(zs)
    l_without = MultiViewSupConLoss(include_pos_in_denom=False)(zs)
    assert not torch.isclose(l_with, l_without)


# ---------------------------------------------------------------------------
# include_same_view_negatives flag
# ---------------------------------------------------------------------------


def test_same_view_negatives_flag_changes_loss():
    zs = _make_views(8, 32, 3, seed=0)
    l_with = MultiViewSupConLoss(include_same_view_negatives=True)(zs)
    l_without = MultiViewSupConLoss(include_same_view_negatives=False)(zs)
    assert not torch.isclose(l_with, l_without)


def _reference_supcon(
    zs: list[torch.Tensor],
    temperature: float,
    loss_type: str,
    include_pos_in_denom: bool,
    include_same_view_negatives: bool,
) -> torch.Tensor:
    """Brute-force per-anchor loops mirroring the MultiViewSupConLoss docs."""
    V = len(zs)
    N = zs[0].shape[0]
    z = F.normalize(torch.stack(zs, dim=1).reshape(N * V, -1), dim=-1)
    sim = (z @ z.T) / temperature
    total = N * V

    losses = []
    for i in range(total):
        sample_i, view_i = divmod(i, V)
        pos = [j for j in range(total) if j != i and j // V == sample_i]
        denom = []
        for j in range(total):
            if j == i:
                continue
            same_sample = j // V == sample_i
            if same_sample and not include_pos_in_denom:
                continue
            if not same_sample and not include_same_view_negatives and j % V == view_i:
                continue
            denom.append(j)
        log_Z = torch.logsumexp(sim[i, denom], dim=0)
        if loss_type == 'out':
            losses.append(torch.stack([log_Z - sim[i, p] for p in pos]).mean())
        else:
            losses.append(log_Z - torch.logsumexp(sim[i, pos], dim=0))
    return torch.stack(losses).mean()


@pytest.mark.parametrize('loss_type', ['out', 'in'])
@pytest.mark.parametrize('include_pos', [True, False])
@pytest.mark.parametrize('include_same_view', [True, False])
def test_matches_bruteforce_reference(loss_type, include_pos, include_same_view):
    zs = _make_views(6, 16, 3, seed=3)
    loss_fn = MultiViewSupConLoss(
        temperature=0.1,
        loss_type=loss_type,
        include_pos_in_denom=include_pos,
        include_same_view_negatives=include_same_view,
    )
    expected = _reference_supcon(
        zs,
        temperature=0.1,
        loss_type=loss_type,
        include_pos_in_denom=include_pos,
        include_same_view_negatives=include_same_view,
    )
    assert torch.isclose(loss_fn(zs), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Collapse sensitivity: identical views should give near-zero gradients
# (perfectly aligned views still produce a valid but low loss)
# ---------------------------------------------------------------------------


def test_perfectly_aligned_views_low_loss():
    # All views are identical → all positives are perfectly aligned.
    # The loss should be lower than with random misaligned views.
    z = F.normalize(torch.randn(16, 64), dim=-1)
    zs_aligned = [z] * 3
    zs_random = _make_views(16, 64, 3, seed=7)

    loss_aligned = MultiViewSupConLoss(temperature=0.1)(zs_aligned).item()
    loss_random = MultiViewSupConLoss(temperature=0.1)(zs_random).item()
    assert loss_aligned < loss_random


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_to_all_views():
    zs = [torch.randn(8, 32, requires_grad=True) for _ in range(3)]
    loss = MultiViewSupConLoss()(zs)
    loss.backward()
    for i, z in enumerate(zs):
        assert z.grad is not None, f'No gradient for view {i}'
        assert z.grad.abs().sum() > 0, f'Zero gradient for view {i}'


# ---------------------------------------------------------------------------
# DDP-fallback path (variable N across ranks)
# ---------------------------------------------------------------------------


def test_no_gather_when_single_process():
    # Without DDP, all_gather_with_grad should return (z, 0, N).
    from fm4tag.utils.ddp import all_gather_with_grad

    z = torch.randn(12, 32)
    z_out, start, end = all_gather_with_grad(z)
    assert torch.equal(z_out, z)
    assert start == 0
    assert end == 12


# ---------------------------------------------------------------------------
# normalize_loss_weights
# ---------------------------------------------------------------------------


def test_weights_normalised_to_unit_sum():
    w = normalize_loss_weights({'a': 2.0, 'b': 1.0, 'c': 1.0})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w['a'] == pytest.approx(0.5)


def test_proportional_weightings_are_equivalent():
    # [2, 1] must be equivalent to [1, 0.5].
    assert normalize_loss_weights({'a': 2.0, 'b': 1.0}) == pytest.approx(
        normalize_loss_weights({'a': 1.0, 'b': 0.5})
    )


def test_zero_weights_stay_zero():
    w = normalize_loss_weights({'a': 1.0, 'b': 0.0})
    assert w == {'a': 1.0, 'b': 0.0}


def test_all_zero_weights_raise():
    with pytest.raises(ValueError, match='At least one'):
        normalize_loss_weights({'a': 0.0, 'b': 0.0})


def test_negative_weights_raise():
    with pytest.raises(ValueError, match='non-negative'):
        normalize_loss_weights({'a': 1.0, 'b': -0.5})


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


# ---------------------------------------------------------------------------
# Empty-input guards (NaN prevention)
# ---------------------------------------------------------------------------


def test_empty_local_anchor_slice_returns_zero(monkeypatch):
    """A rank whose local shard is empty must not produce NaN (DDP edge case)."""
    import fm4tag.losses.contrastive as contrastive_mod

    def _fake_gather(z):
        return z, 0, 0  # gathered pool non-empty, local slice empty

    monkeypatch.setattr(contrastive_mod, 'all_gather_with_grad', _fake_gather)
    loss_fn = MultiViewSupConLoss(temperature=0.1)
    zs = [torch.randn(4, 8, requires_grad=True) for _ in range(2)]
    loss = loss_fn(zs)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()  # graph stays connected


def test_denoising_losses_empty_batch_return_zero():
    from fm4tag.losses import denoising_cat_loss, denoising_con_loss

    cat_outs = [torch.zeros(0, 5, requires_grad=True) for _ in range(2)]
    x_categ = torch.zeros(0, 2, dtype=torch.long)
    l_cat = denoising_cat_loss(cat_outs, x_categ)
    assert torch.isfinite(l_cat) and l_cat.item() == 0.0

    con_outs = [torch.zeros(0, 1, requires_grad=True) for _ in range(3)]
    x_cont = torch.zeros(0, 3)
    l_con = denoising_con_loss(con_outs, x_cont)
    assert torch.isfinite(l_con) and l_con.item() == 0.0
