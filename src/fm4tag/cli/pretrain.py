"""Pretraining CLI entry point.

Run::

    fm4tag-pretrain                                   # all defaults
    fm4tag-pretrain encoder=saint_small               # swap encoder group
    fm4tag-pretrain augmentation=cutmix_only          # no latent mixup
    fm4tag-pretrain augmentation=none                 # no augmentation
    fm4tag-pretrain trainer.max_epochs=50             # Hydra dot-notation
    fm4tag-pretrain --config-path=/my/configs --config-name=my_pretrain

    # Load config from outside the repo:
    fm4tag-pretrain --config-path=/path/to/configs --config-name=experiment_A

On every run the fully-resolved config is written to:
    outputs/<experiment_name>/version_N/config.yaml
"""

from __future__ import annotations

import os

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from fm4tag.datamodules import PT_FT_DataModule
from fm4tag.models import PretrainModule
from fm4tag.utils.builders import (
    build_aug_pipeline,
    build_callbacks,
    build_encoders,
    build_profiler,
)


@hydra.main(version_base=None, config_path='../../../configs', config_name='pretrain')
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('high')
    L.seed_everything(cfg.get('seed', 42), workers=True)

    # ── Loggers ───────────────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(
        save_dir=cfg.get('output_dir', 'outputs'),
        name=cfg.get('experiment_name', 'fm4tag_pretrain'),
    )
    csv_logger = CSVLogger(
        save_dir=cfg.get('output_dir', 'outputs'),
        name=cfg.get('experiment_name', 'fm4tag_pretrain'),
        version=tb_logger.version,
    )

    # ── Save resolved config ──────────────────────────────────────────────────
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    with open(os.path.join(tb_logger.log_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # ── Build components ──────────────────────────────────────────────────────
    encoders = build_encoders(cfg)
    aug_pipeline = build_aug_pipeline(cfg)
    callbacks = build_callbacks(cfg, phase='pretrain')
    profiler = build_profiler(cfg)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        profiler=profiler,
        **trainer_kwargs,
    )

    # ── Data and model ────────────────────────────────────────────────────────
    dm = PT_FT_DataModule(cfg, phase='pretrain')
    module = PretrainModule(encoders, cfg, aug_pipeline=aug_pipeline)

    trainer.fit(module, dm)
