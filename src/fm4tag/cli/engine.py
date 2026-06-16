"""Hydra entry point for fm4tag.

The actual workflow lives in :func:`fm4tag.runner.run.run`; the ``_build_*``
factory helpers live in :mod:`fm4tag.runner.builders`.  This module is only the
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

    cfg = OmegaConf.load('configs/default.yaml')
    run(cfg, phase='pretrain', action='fit')
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from fm4tag.runner.run import run


@hydra.main(version_base=None, config_path='configs', config_name='default')
def main(cfg: DictConfig) -> None:
    run(cfg)
