"""Hydra entry point for fm4tag.

The actual workflow lives in :func:`fm4tag.runner.run.run`; the builder and
instantiation helpers live in :mod:`fm4tag.utils`.  This module is only the
thin Hydra wrapper that resolves the config and calls ``run``.

Run with Hydra (from the project root)::

    # Uses the default config (default.yaml) with its phase/action values.
    python -m fm4tag.engine

    # Switch config file — all keys can be overridden via dot-notation:
    python -m fm4tag.engine --config-name=saintV0 phase=pretrain action=fit
    python -m fm4tag.engine --config-name=saintV0 phase=finetune encoder_ckpt=/path/to/ckpt.pt

    # Or via the installed entry-point (equivalent):
    fm4tag --config-name=saintV0 phase=pretrain

    # Load a config file from OUTSIDE the repo's configs directory:
    fm4tag --config-path=/path/to/my/configs --config-name=my_experiment phase=finetune

For notebooks / scripts (Hydra not involved)::

    from omegaconf import OmegaConf
    from fm4tag.runner import run

    cfg = OmegaConf.load('src/fm4tag/configs/default.yaml')
    run(cfg, phase='pretrain', action='fit')
"""

from __future__ import annotations

import os

# Packed varlen batches change tensor sizes every step and fragment the CUDA
# caching allocator: a 30-epoch pretraining died with 4.3 GiB reserved but
# unusable.  Expandable segments avoid that; a setting in the environment wins.
# Read at CUDA initialisation, so setting it here (before any CUDA call) works.
if (
    'PYTORCH_ALLOC_CONF' not in os.environ
    and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ
):
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from fm4tag.runner.run import run  # noqa: E402


@hydra.main(version_base=None, config_path='../configs', config_name='default')
def main(cfg: DictConfig) -> None:
    run(cfg)
