"""Smoke test: pretrain + finetune (10 epochs each) on the global-only config.

Exercises the no-constituent code path end to end.  Pinned to a single GPU by
the caller via CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import glob

from omegaconf import OmegaConf

from fm4tag.runner import run

CFG = 'configs/jets_only_test.yaml'


def main() -> None:
    cfg = OmegaConf.load(CFG)

    print('\n========== PRETRAIN (10 epochs) ==========', flush=True)
    run(cfg, phase='pretrain', action='fit')

    ckpts = sorted(glob.glob('outputs/jets_only_test/version_*/checkpoints/*.ckpt'))
    print('\ncheckpoints found after pretrain:', ckpts, flush=True)
    last = [c for c in ckpts if c.endswith('last.ckpt')]
    enc_ckpt = last[0] if last else ckpts[0]
    print('Using encoder_ckpt for finetune:', enc_ckpt, flush=True)

    print('\n========== FINETUNE (10 epochs) ==========', flush=True)
    run(cfg, phase='finetune', action='fit', encoder_ckpt=enc_ckpt)

    print('\n========== ALL DONE ==========', flush=True)


if __name__ == '__main__':
    main()
