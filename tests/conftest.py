"""Pytest configuration and shared DDP test helpers."""

from __future__ import annotations

import os
import socket
import traceback as _tb

import torch.distributed as dist
import torch.multiprocessing as mp

# Spawn-context manager — all cross-process primitives (Queue, etc.) must be
# created from the same context as the spawned processes.
_spawn_ctx = mp.get_context('spawn')


# ---------------------------------------------------------------------------
# pytest marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'ddp: mark test as requiring a DDP process group (gloo backend, CPU-only)',
    )


# ---------------------------------------------------------------------------
# DDP test runner
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _spawn_worker(rank, world_size, port, fn, error_queue, args, kwargs):
    """Worker bootstrap: init process group, run fn, tear down."""
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        fn(rank, world_size, *args, **kwargs)
    except Exception:
        error_queue.put((rank, _tb.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_ddp_test(fn, world_size=2, *args, **kwargs):
    """Spawn *world_size* gloo processes and run fn(rank, world_size, ...).

    Assertion errors raised inside any worker are captured via a
    multiprocessing Queue and re-raised in the main process with the full
    per-rank traceback, so pytest shows exactly which rank failed and why.
    """
    port = _find_free_port()
    error_queue = _spawn_ctx.Queue()

    try:
        mp.spawn(
            _spawn_worker,
            args=(world_size, port, fn, error_queue, args, kwargs),
            nprocs=world_size,
            join=True,
        )
    except Exception:
        # Drain worker tracebacks before re-raising.
        errors = []
        while True:
            try:
                errors.append(error_queue.get_nowait())
            except Exception:
                break
        if errors:
            msgs = '\n\n'.join(f'--- Rank {r} ---\n{tb}' for r, tb in errors)
            raise AssertionError(f'DDP worker failure(s):\n{msgs}') from None
        raise
