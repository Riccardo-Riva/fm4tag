"""Checkpoint choice, cache keys and chunk checks of scripts/roc_comparison.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import roc_comparison as rc  # noqa: E402


def _run_dir(tmp_path, val_loss, val_ce):
    ckpt_dir = tmp_path / 'exp' / 'version_0' / 'checkpoints'
    ckpt_dir.mkdir(parents=True)
    for e in range(len(val_loss)):
        (ckpt_dir / f'epoch{e:03d}-step{e}.ckpt').write_bytes(b'')
    rows = ['epoch,val/loss,val/head/loss_ce']
    rows += [f'{e},{a},{b}' for e, (a, b) in enumerate(zip(val_loss, val_ce))]
    (ckpt_dir.parent / 'metrics.csv').write_text('\n'.join(rows) + '\n')
    return tmp_path


def _cfg(phase):
    return OmegaConf.create(
        {
            'phase': phase,
            'callbacks': {'model_checkpoint': {'monitor': 'val/loss', 'mode': 'min'}},
            'callbacks_finetune': {'model_checkpoint': {'monitor': 'val/head/loss_ce'}},
        }
    )


def test_best_checkpoint_of_a_finetune_run_uses_the_ce_monitor(tmp_path):
    run_dir = _run_dir(tmp_path, val_loss=[3.0, 2.0, 1.0], val_ce=[0.60, 0.59, 0.61])
    assert 'epoch001' in rc.resolve_checkpoint(run_dir, _cfg('finetune'), 'best').name
    assert 'epoch002' in rc.resolve_checkpoint(run_dir, _cfg('pretrain'), 'best').name


def test_cache_is_keyed_on_precision(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return np.ones((4, 4))

    args = (tmp_path, 'm', Path('/c.ckpt'), Path('/t.h5'), 4, False)
    rc.cached_probs(*args, compute=compute, precision='fp32')
    rc.cached_probs(*args, compute=compute, precision='fp32')
    assert len(calls) == 1  # cache hit
    rc.cached_probs(*args, compute=compute, precision='bf16')
    assert len(calls) == 2  # different precision → recomputed


def test_old_caches_count_as_fp32(tmp_path):
    np.savez(
        tmp_path / 'probs_m.npz',
        probs=np.ones((4, 4)),
        ckpt_path='/c.ckpt',
        test_file='/t.h5',
    )
    out = rc.cached_probs(
        tmp_path,
        'm',
        Path('/c.ckpt'),
        Path('/t.h5'),
        4,
        False,
        compute=lambda: pytest.fail('should be a cache hit'),
    )
    assert out.shape == (4, 4)


def test_inference_batch_must_split_into_chunks():
    cfg = OmegaConf.create({'chunk_size': 256})
    with pytest.raises(ValueError, match='multiple of'):
        rc.run_inference(None, cfg, Path('/t.h5'), None, 1000, 0, 10, desc='x')
