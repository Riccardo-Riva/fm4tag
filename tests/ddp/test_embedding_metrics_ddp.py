"""DDP tests for the EmbeddingMetrics callback.

Successor of the ``ContrastiveDenoisingModule.on_validation_epoch_end`` DDP
test: each rank holds a slice of a fixed dataset; after the epoch-end gather
the logged metrics must equal those computed on the full dataset.

All tests use 2 CPU processes with the gloo backend (see ``tests/conftest``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.conftest import run_ddp_test
from fm4tag.callbacks import EmbeddingMetrics
from fm4tag.metrics import effective_rank, uniformity


def _make_all_gather_fn(world_size: int):
    """Return a function matching Lightning's self.all_gather API using dist."""

    def all_gather_fn(t: torch.Tensor) -> torch.Tensor:
        buf = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(buf, t.contiguous())
        return torch.stack(buf)  # (world_size, *t.shape)

    return all_gather_fn


def _worker_gathered_metrics(rank: int, world_size: int) -> None:
    """Logged metrics match a single-process computation on the full dataset."""
    D = 8
    N_total = 20
    N_per_rank = N_total // world_size

    # Fixed dataset known to every rank (same seed).
    torch.manual_seed(42)
    all_embs = {
        'jets_embedding': torch.randn(N_total, D),
        'aggregator': torch.randn(N_total, D),
    }

    cb = EmbeddingMetrics(metrics=['uniformity', 'effective_rank'], splits=['val'])

    # Each rank populates its slice directly (bypasses forward — the gather
    # path is what's under test here).
    start, end = rank * N_per_rank, (rank + 1) * N_per_rank
    for name, z in all_embs.items():
        cb._store['val'][name].append(z[start:end].clone())
    cb._names['val'] = sorted(all_embs)

    logged: dict[str, float] = {}
    pl_module = SimpleNamespace(
        log=lambda name, value, **_kw: logged.update({name: float(value)}),
        all_gather=_make_all_gather_fn(world_size),
        device=torch.device('cpu'),
    )
    trainer = SimpleNamespace(world_size=world_size, sanity_checking=False)

    cb.on_validation_epoch_end(trainer, pl_module)

    for name, z_full in all_embs.items():
        for metric, fn in (
            ('uniformity', uniformity),
            ('effective_rank', effective_rank),
        ):
            key = f'val/{name}/{metric}'
            expected = float(fn(z_full))
            assert key in logged, f'{key} was not logged'
            assert abs(logged[key] - expected) < 1e-4, (
                f'{key}: logged {logged[key]:.6f} != expected {expected:.6f}'
            )

    assert len(cb._store['val']) == 0, 'store was not cleared'


@pytest.mark.ddp
def test_gathered_metrics_match_full_dataset():
    run_ddp_test(_worker_gathered_metrics)


def _worker_asymmetric_names(rank: int, world_size: int) -> None:
    """A name with data on only one rank must not deadlock the collectives."""
    D = 4
    cb = EmbeddingMetrics(metrics=['uniformity'], splits=['val'])

    cb._store['val']['everywhere'].append(torch.randn(6, D))
    if rank == 0:  # 'partial' has data only on rank 0 → gather returns None
        cb._store['val']['partial'].append(torch.randn(6, D))
    cb._names['val'] = ['everywhere', 'partial']

    logged: dict[str, float] = {}
    pl_module = SimpleNamespace(
        log=lambda name, value, **_kw: logged.update({name: float(value)}),
        all_gather=_make_all_gather_fn(world_size),
        device=torch.device('cpu'),
    )
    trainer = SimpleNamespace(world_size=world_size, sanity_checking=False)

    cb.on_validation_epoch_end(trainer, pl_module)

    assert 'val/everywhere/uniformity' in logged
    assert 'val/partial/uniformity' not in logged


@pytest.mark.ddp
@pytest.mark.timeout(30)
def test_asymmetric_embedding_names_no_deadlock():
    run_ddp_test(_worker_asymmetric_names)
