"""Core training / evaluation / prediction engine for fm4tag.

This module is the internal engine called by the CLI (``fm4tag.cli``).
It exposes a single public function :func:`run` that accepts a fully resolved
OmegaConf ``DictConfig`` (loaded from a YAML config file) together with
optional path overrides, and dispatches to the appropriate Lightning workflow.

It can also be imported and called directly from notebooks or scripts::

    from omegaconf import OmegaConf
    from fm4tag.train import run

    cfg = OmegaConf.load("src/fm4tag/configs/default.yaml")
    run(cfg, phase="pretrain", action="fit")
"""

from __future__ import annotations

import os

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import lightning as L
from lightning.pytorch.callbacks import (
    BackboneFinetuning,
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import CSVLogger

from fm4tag.data import PT_FT_DataModule
from fm4tag.models import FinetuneModule, PretrainModule
from fm4tag.models.components.encoder import saint_encoder
from fm4tag.models.components.heads import ClassifierHead


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _build_encoder(cfg: DictConfig) -> saint_encoder:
    """Instantiate a :class:`saint_encoder` from the config.

    The number of category classes and continuous features are derived
    automatically from ``cfg.variables`` so there is no duplication between
    the data config and the model config.
    """
    obj_name = list(cfg.constituent_objects)[0]
    obj_vars = cfg.variables[obj_name].inputs

    categories = [
        len(classes) for classes in obj_vars.cat_classes.values()
    ]
    num_continuous = len(obj_vars.continuous)

    enc = cfg.encoder
    return saint_encoder(
        categories=categories,
        num_continuous=num_continuous,
        dim=enc.dim,
        depth=enc.depth,
        heads=enc.heads,
        dim_head=enc.get("dim_head", 16),
        dim_row_head=enc.get("dim_row_head", 64),
        attn_dropout=enc.get("attn_dropout", 0.0),
        ff_dropout=enc.get("ff_dropout", 0.0),
        ff_mult=enc.get("ff_mult", 1),
        cont_embeddings=enc.get("cont_embeddings", "MLP"),
        attentiontype=enc.get("attentiontype", "col"),
        final_mlp_style=enc.get("final_mlp_style", "sep"),
    )


def _load_pretrained_encoder(encoder: saint_encoder, ckpt_path: str) -> saint_encoder:
    """Load encoder weights from a :class:`PretrainModule` checkpoint.

    Only the ``encoder.*`` keys in the Lightning state dict are extracted,
    so the checkpoint does not need to be a full :class:`PretrainModule`
    instance — a raw ``torch.save`` of the state dict also works if the
    keys are prefixed with ``encoder.``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    prefix = "encoder."
    enc_state = {
        k[len(prefix):]: v
        for k, v in state.items()
        if k.startswith(prefix)
    }

    if not enc_state:
        raise KeyError(
            f"No keys starting with '{prefix}' found in checkpoint {ckpt_path}. "
            "Make sure the checkpoint comes from a PretrainModule."
        )

    encoder.load_state_dict(enc_state)
    return encoder


def _build_callbacks(cfg: DictConfig) -> list:
    """Build the list of Lightning callbacks from the config."""
    cb_cfg = cfg.get("callbacks", {})
    callbacks = []

    # Default monitor: val/loss when a validation file is configured, else train/loss.
    _phase = cfg.get("phase", "finetune")
    _val_key = "pretrain_val_file" if _phase == "pretrain" else "val_file"
    _has_val = bool(cfg.get(_val_key))
    _default_monitor = "val/loss" if _has_val else "train/loss"

    # ── ModelSummary ────────────────────────────────────────────────────────
    ms = cb_cfg.get("model_summary", {})
    callbacks.append(ModelSummary(max_depth=ms.get("max_depth", 2)))

    # ── ModelCheckpoint ──────────────────────────────────────────────────────
    ckpt = cb_cfg.get("model_checkpoint", {})
    # If there is no validation set, force train/loss regardless of what the config says.
    _ckpt_monitor = _default_monitor if not _has_val else ckpt.get("monitor", _default_monitor)
    _metric_key = _ckpt_monitor.replace("/", "_")
    callbacks.append(
        ModelCheckpoint(
            monitor=_ckpt_monitor,
            save_top_k=ckpt.get("save_top_k", 3),
            mode=ckpt.get("mode", "min"),
            save_last=True,
            filename="{epoch:03d}-{" + _metric_key + ":.4f}",
            verbose=True,
        )
    )

    # ── EarlyStopping ────────────────────────────────────────────────────────
    es = cb_cfg.get("early_stopping", {})
    _es_monitor = _default_monitor if not _has_val else es.get("monitor", _default_monitor)
    callbacks.append(
        EarlyStopping(
            monitor=_es_monitor,
            patience=es.get("patience", 15),
            mode=es.get("mode", "min"),
            verbose=True,
            check_finite=True,
            # When monitoring a train metric there is no validation loop to hook into.
            check_on_train_epoch_end=not _has_val,
        )
    )

    # ── BackboneFinetuning (finetune phase + freeze_encoder only) ────────────
    if cfg.phase == "finetune" and cfg.get("freeze_encoder", False):
        bf = cb_cfg.get("backbone_finetuning", {})
        if bf.get("enabled", True):
            callbacks.append(
                BackboneFinetuning(
                    unfreeze_backbone_at_epoch=bf.get("unfreeze_backbone_at_epoch", 10),
                    backbone_initial_ratio_lr=bf.get("backbone_initial_ratio_lr", 0.1),
                    train_bn=bf.get("train_bn", False),
                )
            )

    return callbacks


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------

def run(
    cfg: DictConfig,
    *,
    phase: str | None = None,
    action: str | None = None,
    encoder_ckpt: str | None = None,
    ckpt_path: str | None = None,
) -> None:
    """Run the training / evaluation / prediction workflow.

    Args:
        cfg:          Fully resolved OmegaConf DictConfig (from a YAML config
                      file).  ``phase``, ``action``, ``encoder_ckpt``, and
                      ``ckpt_path`` inside the config are used as defaults and
                      can be overridden by the keyword arguments below.
        phase:        ``"pretrain"`` or ``"finetune"``.  Overrides
                      ``cfg.phase`` when provided.
        action:       ``"fit"``, ``"test"``, or ``"predict"``.  Overrides
                      ``cfg.action`` when provided.
        encoder_ckpt: Path to a :class:`PretrainModule` checkpoint to load
                      encoder weights from (finetune only).  Overrides
                      ``cfg.encoder_ckpt`` when provided.
        ckpt_path:    Lightning checkpoint path for resuming ``fit``, or for
                      ``test`` / ``predict``.  Overrides ``cfg.ckpt_path``.
    """
    # ── Resolve overrides ─────────────────────────────────────────────────────
    _phase = phase or cfg.get("phase", "finetune")
    _action = action or cfg.get("action", "fit")
    _enc_ckpt = encoder_ckpt or cfg.get("encoder_ckpt")
    _ckpt = ckpt_path or cfg.get("ckpt_path")

    L.seed_everything(cfg.get("seed", 42), workers=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = CSVLogger(
        save_dir=cfg.get("output_dir", "outputs"),
        name=cfg.get("experiment_name", "fm4tag"),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # BackboneFinetuning is only added when freeze_encoder=true and phase=finetune.
    # Temporarily patch phase so _build_callbacks sees the resolved value.
    with open_dict(cfg):
        cfg.phase = _phase
    callbacks = _build_callbacks(cfg)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_kwargs,
    )

    # ── Data module ───────────────────────────────────────────────────────────
    dm = PT_FT_DataModule(cfg, phase=_phase)

    # ── Lightning module ──────────────────────────────────────────────────────
    if _phase == "pretrain":
        encoder = _build_encoder(cfg)
        module: L.LightningModule = PretrainModule(encoder, cfg)

    elif _phase == "finetune":
        encoder = _build_encoder(cfg)

        if _enc_ckpt is not None:
            encoder = _load_pretrained_encoder(encoder, _enc_ckpt)

        head_cfg = cfg.head
        n_classes = len(cfg.variables[cfg.global_object].unique_labels)
        head = ClassifierHead(
            dim=encoder.dim,
            y_dim=n_classes,
            mlp_dropout=head_cfg.get("mlp_dropout", 0.0),
            ff_dropout=head_cfg.get("ff_dropout", 0.0),
            attn_dropout=head_cfg.get("attn_dropout", 0.0),
            ff_mult=head_cfg.get("ff_mult", 4),
            heads=head_cfg.get("heads", 8),
            dim_head=head_cfg.get("dim_head", 16),
            depth=head_cfg.get("depth", 3),
        )
        module = FinetuneModule(encoder, head, cfg)

    else:
        raise ValueError(f"phase must be 'pretrain' or 'finetune', got {_phase!r}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if _action == "fit":
        trainer.fit(module, dm, ckpt_path=_ckpt)

    elif _action == "test":
        trainer.test(module, dm, ckpt_path=_ckpt or "best")

    elif _action == "predict":
        predictions = trainer.predict(module, dm, ckpt_path=_ckpt or "best")
        out_dir = logger.log_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(predictions, os.path.join(out_dir, "predictions.pt"))

    else:
        raise ValueError(
            f"action must be 'fit', 'test', or 'predict', got {_action!r}"
        )
