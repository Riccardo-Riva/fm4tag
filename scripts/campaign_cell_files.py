#!/usr/bin/env python
"""Compose one SSL-campaign cell and print the data files its training reads.

Used by experiments/2026-10_ssl-campaign/slurm/ssl_campaign.sh at submit time: a typo in a cell's overrides
fails here (non-zero exit) before anything is submitted, and the printed files
are read once sequentially at job start to warm the node's page cache (random
cold reads from NFS were the loader's bottleneck: ~6k vs ~50k jets/s/worker).

    scripts/campaign_cell_files.py arm=pretrain chunk_size=256 ...
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'src' / 'fm4tag' / 'configs'


def main() -> None:
    # encoder_ckpt=@<dir> is resolved by the job; any path composes the same.
    overrides = [
        'encoder_ckpt=/resolved/at/job/start' if o.startswith('encoder_ckpt=@') else o
        for o in sys.argv[1:]
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name='default', overrides=overrides)
    train = (
        cfg.pretrain_dataset_path if cfg.phase == 'pretrain' else cfg.train_dataset_path
    )
    for path in (train, cfg.val_dataset_path):
        if not Path(path).is_file():
            sys.exit(f'data file not found: {path}')
        print(path)


if __name__ == '__main__':
    main()
