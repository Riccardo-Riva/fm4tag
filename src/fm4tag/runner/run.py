"""Public ``run()`` entry point: build everything from config and drive Lightning.

This is the framework-agnostic workflow used by both the Hydra CLI
(:mod:`fm4tag.cli.engine`) and notebooks/scripts.  Runtime objects (encoders,
aggregator, views, datamodule) are built here and injected into the
Hydra-instantiated modules as kwarg overrides — the modules themselves never
see the config.  ``_convert_='all'`` is passed at the call sites so plain
dicts/lists cross the constructor boundary and ``save_hyperparameters`` keeps
checkpoints ``weights_only``-loadable.
"""

from __future__ import annotations

import os

import torch
import yaml
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

import lightning as L

from fm4tag.utils import (
    build_aggregator,
    build_encoders,
    build_head,
    instantiate_callbacks,
    instantiate_loggers,
    load_pretrained_aggregator,
    load_pretrained_encoders,
)


def run(
    cfg: DictConfig,
    *,
    phase: str | None = None,
    action: str | None = None,
    encoder_ckpt: str | None = None,
    ckpt_path: str | None = None,
    extra_callbacks: list | None = None,
) -> L.Trainer:
    """Run the training / evaluation / prediction workflow.

    Returns the :class:`~lightning.Trainer` (e.g. for
    ``trainer.checkpoint_callback.best_model_path`` or
    ``trainer.callback_metrics``).

    Args:
        cfg:          Fully resolved OmegaConf DictConfig (from a YAML config
                      file).  ``phase``, ``action``, ``encoder_ckpt``, and
                      ``ckpt_path`` inside the config are used as defaults and
                      can be overridden by the keyword arguments below.
        phase:        ``"pretrain"`` or ``"finetune"``.  Overrides
                      ``cfg.phase`` when provided.
        action:       ``"fit"``, ``"test"``, or ``"predict"``.  Overrides
                      ``cfg.action`` when provided.
        encoder_ckpt: Path to a pretraining checkpoint to load encoder weights
                      from (finetune only).  Overrides ``cfg.encoder_ckpt``.
        ckpt_path:    Lightning checkpoint path for resuming ``fit``, or for
                      ``test`` / ``predict``.  Overrides ``cfg.ckpt_path``.
        extra_callbacks: Additional Lightning callbacks appended after the
                      ones built from config.  Used by the HPO module to
                      inject :class:`~fm4tag.hpo._OptunaMetricCallback`.
    """
    _phase = phase or cfg.get('phase', 'finetune')
    _action = action or cfg.get('action', 'fit')
    _enc_ckpt = encoder_ckpt or cfg.get('encoder_ckpt')
    _ckpt = ckpt_path or cfg.get('ckpt_path')

    if _phase not in ('pretrain', 'finetune'):
        raise ValueError(f"phase must be 'pretrain' or 'finetune', got {_phase!r}")

    torch.set_float32_matmul_precision(cfg.get('float32_matmul_precision', 'high'))
    L.seed_everything(cfg.get('seed', 42), workers=True)

    # ── Trainer infrastructure ────────────────────────────────────────────
    logger = instantiate_loggers(cfg.get('loggers'))
    # Shared callbacks + phase-specific extras, merged in (not appended) so
    # that reusing a shared callback's key (e.g. ``model_checkpoint``) patches
    # its fields — such as ``monitor`` — instead of instantiating a second,
    # competing callback. Genuinely new keys (e.g. BackboneFinetuning, which
    # requires ``module.backbone`` and must not be built during pretraining)
    # are still added as normal.
    callbacks_cfg = cfg.get('callbacks')
    if _phase == 'finetune' and cfg.get('callbacks_finetune'):
        callbacks_cfg = OmegaConf.merge(callbacks_cfg, cfg.callbacks_finetune)
    callbacks = instantiate_callbacks(callbacks_cfg)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    profiler = hydra_instantiate(cfg.profiler) if cfg.get('profiler') else None

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
        **trainer_kwargs,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    dm = hydra_instantiate(
        cfg.datamodule,
        _convert_='all',
        train_dataset_path=(
            cfg.pretrain_dataset_path if _phase == 'pretrain' else cfg.train_dataset_path
        ),
    )

    # ── Models + module ───────────────────────────────────────────────────
    encoders = build_encoders(cfg)
    aggregator = build_aggregator(cfg, encoders)
    views = [hydra_instantiate(v) for v in cfg.get('views', [])]

    if _phase == 'pretrain':
        module: L.LightningModule = hydra_instantiate(
            cfg.pretrain,
            encoders=encoders,
            aggregator=aggregator,
            views=views,
            _convert_='all',
        )

    else:  # finetune
        head = build_head(cfg, aggregator)

        if _enc_ckpt is not None:
            load_pretrained_encoders(encoders, _enc_ckpt)
            # The aggregator is shared with pretraining; load it too unless
            # disabled (meaningful only if pretraining trained it).
            if cfg.get('load_aggregator', True):
                load_pretrained_aggregator(aggregator, _enc_ckpt)

        overrides: dict = {}
        if cfg.get('use_class_weights', False):
            # Per-class CE weights for the global object's label, taken from
            # the same class-dict file the datamodule uses.
            with open(cfg.class_dict_path) as f:
                class_dict = yaml.safe_load(f)
            label_var = cfg.variables[cfg.global_object].label
            overrides['class_weights'] = class_dict[cfg.global_object][label_var]

        n_classes = len(cfg.variables[cfg.global_object].unique_labels)
        module = hydra_instantiate(
            cfg.finetune,
            encoders=encoders,
            aggregator=aggregator,
            head=head,
            views=views,
            n_classes=n_classes,
            _convert_='all',
            **overrides,
        )

    # ── Action ────────────────────────────────────────────────────────────
    # Checkpoints written by the current modules contain only plain-typed
    # hyper-parameters and load under torch's weights_only=True default.
    # Pre-refactor checkpoints (omegaconf objects in hyper_parameters) need
    # an explicit weights_only=False here.
    if _action == 'fit':
        trainer.fit(module, dm, ckpt_path=_ckpt)

    elif _action == 'test':
        trainer.test(module, dm, ckpt_path=_ckpt or 'best')

    elif _action == 'predict':
        predictions = trainer.predict(module, dm, ckpt_path=_ckpt or 'best')
        out_dir = trainer.log_dir or cfg.get('output_dir', 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        torch.save(predictions, os.path.join(out_dir, 'predictions.pt'))

    else:
        raise ValueError(f"action must be 'fit', 'test', or 'predict', got {_action!r}")

    return trainer
