"""Finetuning CLI entry point.

A model config file must always be specified with --config-name.
The same config used for pretraining also drives finetuning — it contains
both pretrain and finetune sections; each CLI reads what it needs.

Run::

    fm4tag-finetune --config-name=atlas_tracks_saint
    fm4tag-finetune --config-name=atlas_tracks_saint encoder_ckpt=/path/to/pretrain.ckpt
    fm4tag-finetune --config-name=atlas_tracks_saint ckpt_path=/path/to/resume.ckpt
    fm4tag-finetune --config-name=atlas_tracks_saint action=test ckpt_path=/path/to/best.ckpt
    fm4tag-finetune --config-name=atlas_tracks_saint freeze_encoder=true

    # Config file outside the repo:
    fm4tag-finetune --config-path=/my/configs --config-name=my_model

On every run the fully-resolved config is written to:
    outputs/<experiment_name>/version_N/config.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from fm4tag.datamodules import PT_FT_DataModule
from fm4tag.models import FinetuneModule
from fm4tag.models.components.heads import MultiStreamClassifierHead
from fm4tag.utils.builders import (
    build_callbacks,
    build_encoders,
    build_profiler,
    load_pretrained_encoders,
)


def _build_head(cfg: DictConfig, encoders: torch.nn.ModuleDict) -> MultiStreamClassifierHead:
    """Construct the classifier head, inferring projection dims from the encoders."""
    head_cfg = cfg.head
    enc_cfg = cfg.encoder
    n_classes = len(cfg.variables[cfg.global_object].unique_labels)

    n_global = len(cfg.variables[cfg.global_object].inputs)
    global_proj_out = n_global * enc_cfg.dim

    const_proj_outs = []
    for obj_name in cfg.constituent_objects:
        n_feat = (
            len(cfg.variables[obj_name].inputs.continuous)
            + len(cfg.variables[obj_name].inputs.categorical)
        )
        proj_in = n_feat * enc_cfg.dim
        _proj_out = enc_cfg.get('proj_out') or proj_in // 2
        const_proj_outs.append(_proj_out)

    head = MultiStreamClassifierHead(
        global_proj_out=global_proj_out,
        const_proj_outs=const_proj_outs,
        y_dim=n_classes,
        mlp_dropout=head_cfg.get('mlp_dropout', 0.0),
        ff_dropout=head_cfg.get('ff_dropout', 0.0),
        attn_dropout=head_cfg.get('attn_dropout', 0.0),
        ff_mult=head_cfg.get('ff_mult', 4),
        heads=head_cfg.get('heads', 8),
        dim_head=head_cfg.get('dim_head', 16),
        depth=head_cfg.get('depth', 3),
    )

    # Sanity-check that inferred dims match actual encoder output sizes.
    enc_global_out = encoders[cfg.global_object].pt_mlp1.layers[-1].out_features
    assert enc_global_out == global_proj_out, (
        f'GlobalEncoder.pt_mlp1 output ({enc_global_out}) != inferred ({global_proj_out})'
    )
    for i, obj_name in enumerate(cfg.constituent_objects):
        enc_out = encoders[obj_name].pt_mlp1.layers[-1].out_features
        assert enc_out == const_proj_outs[i], (
            f'Encoder.pt_mlp1 output for {obj_name!r} ({enc_out}) != inferred ({const_proj_outs[i]})'
        )

    return head


_CONFIGS = str((Path(__file__).resolve().parent / "../../../configs").resolve())


@hydra.main(version_base=None, config_path=_CONFIGS, config_name=None)
def main(cfg: DictConfig) -> None:
    action = cfg.get('action', 'fit')
    enc_ckpt = cfg.get('encoder_ckpt')
    ckpt_path = cfg.get('ckpt_path')

    torch.set_float32_matmul_precision('high')
    L.seed_everything(cfg.get('seed', 42), workers=True)

    # ── Loggers ───────────────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(
        save_dir=cfg.get('output_dir', 'outputs'),
        name=cfg.get('experiment_name', 'fm4tag_finetune'),
    )
    csv_logger = CSVLogger(
        save_dir=cfg.get('output_dir', 'outputs'),
        name=cfg.get('experiment_name', 'fm4tag_finetune'),
        version=tb_logger.version,
    )

    # ── Save resolved config ──────────────────────────────────────────────────
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    with open(os.path.join(tb_logger.log_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # ── Build components ──────────────────────────────────────────────────────
    encoders = build_encoders(cfg)
    if enc_ckpt is not None:
        load_pretrained_encoders(encoders, enc_ckpt)

    head = _build_head(cfg, encoders)
    callbacks = build_callbacks(cfg, phase='finetune')
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
    dm = PT_FT_DataModule(cfg, phase='finetune')
    module = FinetuneModule(encoders, head, cfg)

    # weights_only=False: checkpoints may embed OmegaConf objects
    if action == 'fit':
        trainer.fit(module, dm, ckpt_path=ckpt_path, weights_only=False)

    elif action == 'test':
        trainer.test(module, dm, ckpt_path=ckpt_path or 'best', weights_only=False)

    elif action == 'predict':
        predictions = trainer.predict(
            module, dm, ckpt_path=ckpt_path or 'best', weights_only=False
        )
        out_dir = tb_logger.log_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(predictions, os.path.join(out_dir, 'predictions.pt'))

    else:
        raise ValueError(f"action must be 'fit', 'test', or 'predict', got {action!r}")
