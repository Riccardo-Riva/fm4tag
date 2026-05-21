"""DDP correctness tests for fm4tag embedding gather logic.

All multi-process tests use 2 CPU processes with the gloo backend.
Worker functions must be module-level (not closures) to be picklable
by torch.multiprocessing.spawn.

Test summary
------------
1. basic_size_matched_gather     – shape, data order, cross-rank equality
2. zero_rank_skips               – n_min==0 returns None on both ranks, no hang
3. multi_object_asymmetric       – missing-object rank still joins collective
4. pretrain_module_e2e           – full on_validation_epoch_end path
5. sanity_check_skip             – sanity_checking==True clears buffers, no log
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.conftest import run_ddp_test
from fm4tag.utils.ddp import gather_embeddings_sized


# ---------------------------------------------------------------------------
# Shared helpers used by multiple workers
# ---------------------------------------------------------------------------


def _make_all_gather_fn(world_size: int):
    """Return a function matching Lightning's self.all_gather API using dist."""

    def all_gather_fn(t: torch.Tensor) -> torch.Tensor:
        buf = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(buf, t.contiguous())
        return torch.stack(buf)  # (world_size, *t.shape)

    return all_gather_fn


def _make_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            'global_object': 'jets',
            'constituent_objects': ['tracks'],
            'pretrain': {
                '_target_': 'fm4tag.modules.ContrastiveDenoisingModule',
                'nce_temp': 0.07,
                'loss_type': 'out',
                'include_pos_in_denom': True,
                'lam_contrastive': 1.0,
                'lam_denoising_cat': 0.0,
                'lam_denoising_con': 0.0,
            },
            'optimizer': {'lr': 1e-3},
            'eval': {
                'enabled': True,
                'splits': ['val'],
                'n_samples': 8192,
                'metrics': ['uniformity', 'effective_rank'],
            },
        }
    )


def _make_encoders():
    from fm4tag.models import Encoder, GlobalEncoder

    return torch.nn.ModuleDict(
        {
            'jets': GlobalEncoder(num_features=2, dim=4),
            'tracks': Encoder(
                categories=[2, 3],
                num_continuous=2,
                dim=4,
                layers=[{'type': 'col', 'heads': 1, 'dim_head': 4}],
            ),
        }
    )


# ===========================================================================
# Test 1 – basic size-matched gather
# ===========================================================================


def _worker_basic_gather(rank: int, world_size: int) -> None:
    """Rank 0 has 7 rows, rank 1 has 5 rows → n_min=5, output shape (10, D)."""
    D = 8
    n_local = 7 if rank == 0 else 5
    # Fill each rank's tensor with a distinct value so we can verify slot order.
    z_local = torch.full((n_local, D), float(rank + 1))

    all_gather_fn = _make_all_gather_fn(world_size)
    z = gather_embeddings_sized(z_local, world_size, all_gather_fn, torch.device('cpu'))

    n_min = 5
    assert z is not None
    assert z.shape == (world_size * n_min, D), f'unexpected shape {z.shape}'

    # Rank 0's data (value 1.0) comes first; rank 1's data (value 2.0) second.
    assert torch.all(z[:n_min] == 1.0), 'first half should be rank-0 data'
    assert torch.all(z[n_min:] == 2.0), 'second half should be rank-1 data'

    # Both ranks must hold the identical tensor — verify via cross-rank sum.
    my_sum = torch.tensor([z.sum().item()])
    all_sums = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(all_sums, my_sum)
    for s in all_sums:
        assert abs(s.item() - my_sum.item()) < 1e-4, (
            'gathered tensor differs across ranks'
        )


@pytest.mark.ddp
def test_basic_size_matched_gather():
    run_ddp_test(_worker_basic_gather)


# ===========================================================================
# Test 2 – one rank with zero samples: no hang, returns None
# ===========================================================================


def _worker_zero_rank(rank: int, world_size: int) -> None:
    """Rank 0 has 100 rows, rank 1 has 0 → n_min=0, both ranks return None."""
    D = 8
    z_local = torch.randn(100, D) if rank == 0 else None

    all_gather_fn = _make_all_gather_fn(world_size)
    z = gather_embeddings_sized(z_local, world_size, all_gather_fn, torch.device('cpu'))

    assert z is None, f'expected None when any rank has zero data, got {type(z)}'


@pytest.mark.ddp
@pytest.mark.timeout(30)
def test_zero_rank_skips():
    """No deadlock when one rank contributes zero samples."""
    run_ddp_test(_worker_zero_rank)


# ===========================================================================
# Test 3 – multiple objects, asymmetric population
# ===========================================================================


def _worker_multi_object_asymmetric(rank: int, world_size: int) -> None:
    """3 objects; rank 1 has no 'vertices' data.

    All ranks must still enter the n_local collective for 'vertices'
    (with n_local=0 on rank 1) so neither rank deadlocks.  The result for
    'vertices' must be None on both ranks; 'global' and 'tracks' must be
    non-None.
    """
    D = 8
    object_names = ['global', 'tracks', 'vertices']
    # Rank 1 contributes nothing for 'vertices'.
    counts = {'global': 10, 'tracks': 8, 'vertices': 6 if rank == 0 else 0}

    all_gather_fn = _make_all_gather_fn(world_size)
    device = torch.device('cpu')
    results: dict[str, torch.Tensor | None] = {}

    for obj_name in object_names:
        n = counts[obj_name]
        z_local = torch.randn(n, D) if n > 0 else None
        results[obj_name] = gather_embeddings_sized(
            z_local, world_size, all_gather_fn, device
        )

    assert results['global'] is not None, "'global' should have data on both ranks"
    assert results['tracks'] is not None, "'tracks' should have data on both ranks"
    assert results['vertices'] is None, "'vertices' should be None (rank 1 had no data)"


@pytest.mark.ddp
@pytest.mark.timeout(30)
def test_multi_object_asymmetric():
    """Missing-object rank joins collectives in order; no deadlock."""
    run_ddp_test(_worker_multi_object_asymmetric)


# ===========================================================================
# Test 4 – end-to-end PretrainModule.on_validation_epoch_end
# ===========================================================================


def _worker_pretrain_e2e(rank: int, world_size: int) -> None:
    """Full on_validation_epoch_end path with synthetic embeddings.

    Both ranks share a fixed global dataset (seed 42).  Each rank holds its
    own slice; after the gather the logged metrics must match those computed
    on the full dataset.
    """
    from fm4tag.augmentations import Compose
    from fm4tag.metrics import effective_rank, uniformity
    from fm4tag.modules import ContrastiveDenoisingModule

    cfg = _make_cfg()
    encoders = _make_encoders()
    views = [Compose([]), Compose([])]
    module = ContrastiveDenoisingModule(encoders=encoders, views=views, cfg=cfg)

    # Wire up a minimal trainer stub (reads from _trainer in Lightning 2.x).
    module._trainer = SimpleNamespace(
        world_size=world_size, sanity_checking=False, current_epoch=0
    )

    # Capture logged values.
    logged: dict[str, float] = {}

    def _mock_log(name, value, **_kw):
        logged[name] = float(value)

    module.log = _mock_log
    module.print = lambda *a, **_kw: None

    # Build a fixed global dataset known to both ranks (same seed).
    D = 8
    N_total = 20
    N_per_rank = N_total // world_size
    torch.manual_seed(42)
    all_embs = {
        'jets': torch.randn(N_total, D),
        'tracks': torch.randn(N_total, D),
    }

    # Each rank populates its slice of the global dataset.
    start, end = rank * N_per_rank, (rank + 1) * N_per_rank
    for obj_name in ('jets', 'tracks'):
        module._val_emb_acc[obj_name].append(all_embs[obj_name][start:end].clone())

    # Provide a fake loss entry so the epoch-table formatting doesn't crash.
    module._val_acc['loss'].append(torch.tensor(0.5))

    # Replace all_gather with a real dist-backed version.
    all_gather_fn = _make_all_gather_fn(world_size)
    module.all_gather = all_gather_fn

    module.on_validation_epoch_end()

    # Compare against metrics computed on the full (gathered) dataset.
    for obj_name in ('jets', 'tracks'):
        z_all = all_embs[obj_name]  # (N_total, D)
        expected_u = float(uniformity(z_all))
        expected_er = float(effective_rank(z_all))

        key_u = f'val_{obj_name}/uniformity'
        key_er = f'val_{obj_name}/effective_rank'
        assert key_u in logged, f'{key_u} was not logged'
        assert key_er in logged, f'{key_er} was not logged'
        assert abs(logged[key_u] - expected_u) < 1e-4, (
            f'{key_u}: logged {logged[key_u]:.6f} != expected {expected_u:.6f}'
        )
        assert abs(logged[key_er] - expected_er) < 1e-4, (
            f'{key_er}: logged {logged[key_er]:.6f} != expected {expected_er:.6f}'
        )


@pytest.mark.ddp
def test_pretrain_module_e2e():
    """Logged uniformity/effective_rank matches manual computation on full data."""
    run_ddp_test(_worker_pretrain_e2e)


# ===========================================================================
# Test 5 – sanity_checking=True clears buffers without logging or hanging
# ===========================================================================


def test_sanity_check_skip():
    """sanity_checking=True branch clears buffers and logs nothing (no DDP)."""
    from fm4tag.augmentations import Compose
    from fm4tag.modules import ContrastiveDenoisingModule

    cfg = _make_cfg()
    encoders = _make_encoders()
    views = [Compose([]), Compose([])]
    module = ContrastiveDenoisingModule(encoders=encoders, views=views, cfg=cfg)

    module._trainer = SimpleNamespace(world_size=1, sanity_checking=True)
    module.print = lambda *a, **_kw: None

    # Populate both buffers with fake data.
    module._val_emb_acc['jets'].append(torch.randn(10, 8))
    module._val_acc['loss'].append(torch.tensor(0.5))

    logged: dict[str, float] = {}
    module.log = lambda name, value, **_kw: logged.update({name: float(value)})

    module.on_validation_epoch_end()

    assert len(module._val_emb_acc) == 0, '_val_emb_acc was not cleared'
    assert len(module._val_acc) == 0, '_val_acc was not cleared'
    assert len(logged) == 0, f'unexpected log calls during sanity check: {list(logged)}'
