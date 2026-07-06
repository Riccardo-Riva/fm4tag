"""DDP gradient-correctness tests for the contrastive losses.

These tests verify the property the reference script
(``torch-ddp-testing/scripts/test_ddp_contrastive_loss.py``) checks: a
*contrastive* objective under DDP must produce the **same gradient** as if the
whole global batch had been processed on a single device.  That only holds if
the cross-rank all-gather is *differentiable* (so gradients reach every rank's
rows); a plain ``torch.distributed.all_gather`` silently drops the cross-rank
interaction terms.

All tests use 2 CPU processes with the gloo backend (see ``tests/conftest``).

Test summary
------------
1. value_matches_independent_ref – framework loss == independent SupCon impl
2. ddp_gradient_matches_full_batch – DDP-averaged grad == single-process
   full-batch grad (the core differentiable-gather guarantee)
3. variable_n_gather_layout – unequal per-rank sizes: padding is stripped and
   offsets are correct
4. ddp_variable_n_value_and_grad – unequal per-rank sizes: every anchor still
   sees the full global pool; loss and gradient match the single-process
   mean-of-per-rank-means reference
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import run_ddp_test
from fm4tag.losses import MultiViewSupConLoss
from fm4tag.utils.ddp import all_gather_with_grad


# ---------------------------------------------------------------------------
# Independent reference SupCon (L_out, positives kept in the denominator).
# Mirrors torch-ddp-testing/scripts/test_ddp_contrastive_loss.py::supcon_loss,
# adapted to the list-of-views interface.  Deliberately a *separate*
# implementation so it also cross-checks the framework's formula.
# ---------------------------------------------------------------------------


def _supcon_reference(
    z_list: list[torch.Tensor], temperature: float, eps: float = 1e-12
) -> torch.Tensor:
    V = len(z_list)
    N = z_list[0].size(0)
    # View-major layout + matching labels (SupCon is permutation-invariant, so
    # the ordering need not match the framework's interleaving).
    feats = F.normalize(torch.cat(z_list, dim=0), dim=-1)  # (V*N, D)
    labels = torch.arange(N).repeat(V)  # (V*N,)
    b = feats.size(0)

    sim = (feats @ feats.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(b, dtype=torch.bool)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask

    exp_sim = torch.exp(sim) * (~self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    pos_count = pos_mask.sum(dim=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)
    valid = pos_count > 0
    return -mean_log_prob_pos[valid].mean()


# ---------------------------------------------------------------------------
# Shared problem definition (identical on every rank and in the reference).
# ---------------------------------------------------------------------------

_N_PER_RANK = 4
_F = 6
_D = 5
_V = 3
_TEMP = 0.1
_VIEW_SCALES = [1.0 + 0.3 * v for v in range(_V)]


def _make_encoder() -> torch.nn.Linear:
    """Deterministic encoder; DDP also broadcasts rank-0 params at wrap time."""
    torch.manual_seed(123)
    return torch.nn.Linear(_F, _D, bias=False)


def _embed(encoder: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Produce V deterministically-distinct views in a *single* forward pass.

    Concatenating the views into one forward keeps DDP happy (one forward per
    backward) and avoids any RNG, so the result is bit-reproducible.
    """
    n = x.size(0)
    x_views = torch.cat([x * s for s in _VIEW_SCALES], dim=0)  # (V*n, F)
    h = encoder(x_views)  # (V*n, D)
    return list(h.split(n, dim=0))  # V x (n, D)


# ===========================================================================
# Test 1 – framework loss value matches the independent reference (no DDP)
# ===========================================================================


def test_value_matches_independent_ref():
    torch.manual_seed(0)
    zs = [torch.randn(8, _D) for _ in range(_V)]
    framework = MultiViewSupConLoss(
        temperature=_TEMP, loss_type='out', include_pos_in_denom=True
    )(zs)
    reference = _supcon_reference(zs, temperature=_TEMP)
    assert torch.allclose(framework, reference, atol=1e-6), (
        f'framework {framework.item():.8f} != reference {reference.item():.8f}'
    )


# ===========================================================================
# Test 2 – DDP-averaged gradient equals the single-process full-batch gradient
# ===========================================================================


def _worker_grad_equiv(rank: int, world_size: int) -> None:
    # Fixed global batch, identical on every rank.
    torch.manual_seed(7)
    x_full = torch.randn(world_size * _N_PER_RANK, _F)

    # ---- DDP gradient (the path under test) -------------------------------
    enc = _make_encoder()
    ddp = torch.nn.parallel.DistributedDataParallel(enc)

    x_local = x_full[rank * _N_PER_RANK : (rank + 1) * _N_PER_RANK]
    zs_local = _embed(ddp, x_local)
    loss = MultiViewSupConLoss(
        temperature=_TEMP, loss_type='out', include_pos_in_denom=True
    )(zs_local)
    loss.backward()
    ddp_grad = enc.weight.grad.detach().clone()  # DDP has averaged across ranks

    # ---- Reference: full-batch gradient on a single process --------------
    # Independent encoder with identical init; no dist calls, so no gather.
    enc_ref = _make_encoder()
    zs_full = _embed(enc_ref, x_full)
    ref_loss = _supcon_reference(zs_full, temperature=_TEMP)
    ref_loss.backward()
    ref_grad = enc_ref.weight.grad.detach().clone()

    assert torch.allclose(ddp_grad, ref_grad, atol=1e-5, rtol=1e-4), (
        f'rank {rank}: DDP gradient disagrees with full-batch reference\n'
        f'  max abs diff = {(ddp_grad - ref_grad).abs().max().item():.3e}\n'
        f'  ddp_grad[0]  = {ddp_grad.flatten()[:4].tolist()}\n'
        f'  ref_grad[0]  = {ref_grad.flatten()[:4].tolist()}'
    )


@pytest.mark.ddp
def test_ddp_gradient_matches_full_batch():
    """Differentiable gather ⇒ DDP grad == single-device full-batch grad."""
    run_ddp_test(_worker_grad_equiv)


# ===========================================================================
# Test 3 – variable-N gather: padding stripped, rank order and offsets kept
# ===========================================================================


def _worker_variable_n_gather(rank: int, world_size: int) -> None:
    """Rank 0 has 7 rows, rank 1 has 5 → z_all is (12, D) with no padding."""
    D = 4
    n_local = 7 if rank == 0 else 5
    z = torch.full((n_local, D), float(rank + 1))

    z_all, start, end = all_gather_with_grad(z)

    assert z_all.shape == (12, D), f'unexpected shape {z_all.shape}'
    assert torch.all(z_all[:7] == 1.0), 'rank-0 rows must come first'
    assert torch.all(z_all[7:] == 2.0), 'rank-1 rows must follow'

    expected_start = 0 if rank == 0 else 7
    assert (start, end) == (expected_start, expected_start + n_local)
    assert torch.equal(z_all[start:end], z), 'local slice must be own data'


@pytest.mark.ddp
def test_variable_n_gather_layout():
    """Unequal per-rank sizes: padded gather strips padding, offsets correct."""
    run_ddp_test(_worker_variable_n_gather)


# ===========================================================================
# Test 4 – variable-N loss: full global pool per anchor, value + grad match
# ===========================================================================

_N_SPLIT = (6, 4)  # unequal per-rank sample counts


def _per_anchor_supcon(z_list: list[torch.Tensor], temperature: float) -> torch.Tensor:
    """Per-anchor L_out values over the full pool, shape ``(N, V)``.

    Per-anchor values are ordering-independent (each depends only on the set
    of instances), so this single-process computation is a valid reference for
    any sharding of the same data.
    """
    V = len(z_list)
    N = z_list[0].size(0)
    z = F.normalize(torch.stack(z_list, dim=1).reshape(N * V, -1), dim=-1)
    sim = (z @ z.t()) / temperature
    labels = torch.arange(N).repeat_interleave(V)
    self_mask = torch.eye(N * V, dtype=torch.bool)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    log_Z = torch.logsumexp(sim.masked_fill(self_mask, float('-inf')), dim=-1)
    n_pos = pos_mask.float().sum(1)
    per_anchor = log_Z - (sim * pos_mask.float()).sum(1) / n_pos.clamp(min=1)
    return per_anchor.reshape(N, V)


def _worker_variable_n_value_and_grad(rank: int, world_size: int) -> None:
    assert world_size == 2
    torch.manual_seed(7)
    x_full = torch.randn(sum(_N_SPLIT), _F)
    bounds = (0, _N_SPLIT[0], sum(_N_SPLIT))

    # ---- DDP path with unequal per-rank batch ------------------------------
    enc = _make_encoder()
    ddp = torch.nn.parallel.DistributedDataParallel(enc)
    x_local = x_full[bounds[rank] : bounds[rank + 1]]
    zs_local = _embed(ddp, x_local)
    loss = MultiViewSupConLoss(
        temperature=_TEMP, loss_type='out', include_pos_in_denom=True
    )(zs_local)
    loss.backward()
    ddp_grad = enc.weight.grad.detach().clone()

    # ---- Single-process reference ------------------------------------------
    # Each rank reduces over its own anchors, but every anchor must see the
    # full global pool; DDP then averages ranks equally (mean of means).
    enc_ref = _make_encoder()
    zs_full = _embed(enc_ref, x_full)
    per_anchor = _per_anchor_supcon(zs_full, _TEMP)  # (N_total, V)
    rank_means = [
        per_anchor[bounds[r] : bounds[r + 1]].mean() for r in range(world_size)
    ]

    assert torch.allclose(loss.detach(), rank_means[rank].detach(), atol=1e-5), (
        f'rank {rank}: local loss {loss.item():.8f} != full-pool reference '
        f'{rank_means[rank].item():.8f} — anchors did not see the global pool'
    )

    ref_loss = torch.stack(rank_means).mean()
    ref_loss.backward()
    ref_grad = enc_ref.weight.grad.detach().clone()

    assert torch.allclose(ddp_grad, ref_grad, atol=1e-5, rtol=1e-4), (
        f'rank {rank}: DDP gradient disagrees with mean-of-rank-means reference\n'
        f'  max abs diff = {(ddp_grad - ref_grad).abs().max().item():.3e}'
    )


@pytest.mark.ddp
def test_ddp_variable_n_value_and_grad():
    """Unequal per-rank N: anchors see the global pool; grads match reference."""
    run_ddp_test(_worker_variable_n_value_and_grad)
