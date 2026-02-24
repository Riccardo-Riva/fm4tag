"""Core engine for fm4tag.

Run with Hydra (from the project root)::

    # Uses the default config (default.yaml) with its phase/action values.
    python -m fm4tag.engine

    # Switch config file — all keys can be overridden via dot-notation:
    python -m fm4tag.engine --config-name=saintV0
    python -m fm4tag.engine --config-name=saintV0 phase=pretrain action=fit
    python -m fm4tag.engine --config-name=saintV0 phase=finetune encoder_ckpt=/path/to/ckpt.pt
    python -m fm4tag.engine --config-name=saintV0 phase=finetune ckpt_path=/path/to/ckpt.pt

    # Or via the installed entry-point (equivalent):
    fm4tag --config-name=saintV0 phase=pretrain

For notebooks / scripts (Hydra not involved)::

    from omegaconf import OmegaConf
    from fm4tag.engine import run

    cfg = OmegaConf.load('src/fm4tag/configs/saintV0.yaml')
    run(cfg, phase='pretrain', action='fit')
"""

from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.callbacks import (
    BackboneFinetuning,
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

from fm4tag.data import PT_FT_DataModule
from fm4tag.models import FinetuneModule, PretrainModule
from fm4tag.models.components.encoder import Encoder, GlobalEncoder
from fm4tag.models.components.heads import ClassifierHead


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _build_encoders(cfg: DictConfig) -> torch.nn.ModuleDict:
    """Build one encoder per object (global + all constituents).

    Returns a :class:`~torch.nn.ModuleDict` keyed by object name:

    * ``cfg.global_object``      → :class:`GlobalEncoder` (per-feature MLP)
    * each ``cfg.constituent_objects`` → :class:`Encoder` (transformer)
    """
    enc_cfg = cfg.encoder
    encoders: dict[str, torch.nn.Module] = {}

    # ── Global encoder (continuous-only, MLP) ────────────────────────────────
    global_name = cfg.global_object
    n_global = len(cfg.variables[global_name].inputs)
    encoders[global_name] = GlobalEncoder(num_features=n_global, dim=enc_cfg.dim)

    # ── Constituent encoders (transformer) ───────────────────────────────────
    for obj_name in cfg.constituent_objects:
        obj_vars = cfg.variables[obj_name].inputs
        categories = [len(classes) for classes in obj_vars.cat_classes.values()]
        num_continuous = len(obj_vars.continuous)
        encoders[obj_name] = Encoder(
            categories=categories,
            num_continuous=num_continuous,
            dim=enc_cfg.dim,
            depth=enc_cfg.depth,
            heads=enc_cfg.heads,
            dim_head=enc_cfg.get('dim_head', 16),
            dim_row_head=enc_cfg.get('dim_row_head', 64),
            attn_dropout=enc_cfg.get('attn_dropout', 0.0),
            ff_dropout=enc_cfg.get('ff_dropout', 0.0),
            ff_mult=enc_cfg.get('ff_mult', 1),
            cont_embeddings=enc_cfg.get('cont_embeddings', 'MLP'),
            attentiontype=enc_cfg.get('attentiontype', 'col'),
            final_mlp_style=enc_cfg.get('final_mlp_style', 'sep'),
        )

    return torch.nn.ModuleDict(encoders)


def _build_constituent_encoder(cfg: DictConfig, obj_name: str) -> Encoder:
    """Build a single constituent :class:`Encoder` (used by the finetune path)."""
    obj_vars = cfg.variables[obj_name].inputs
    categories = [len(classes) for classes in obj_vars.cat_classes.values()]
    num_continuous = len(obj_vars.continuous)
    enc_cfg = cfg.encoder
    return Encoder(
        categories=categories,
        num_continuous=num_continuous,
        dim=enc_cfg.dim,
        depth=enc_cfg.depth,
        heads=enc_cfg.heads,
        dim_head=enc_cfg.get('dim_head', 16),
        dim_row_head=enc_cfg.get('dim_row_head', 64),
        attn_dropout=enc_cfg.get('attn_dropout', 0.0),
        ff_dropout=enc_cfg.get('ff_dropout', 0.0),
        ff_mult=enc_cfg.get('ff_mult', 1),
        cont_embeddings=enc_cfg.get('cont_embeddings', 'MLP'),
        attentiontype=enc_cfg.get('attentiontype', 'col'),
        final_mlp_style=enc_cfg.get('final_mlp_style', 'sep'),
    )


def _load_pretrained_encoder(
    encoder: Encoder,
    ckpt_path: str,
    obj_name: str,
) -> Encoder:
    """Load constituent encoder weights from a :class:`PretrainModule` checkpoint.

    Supports both checkpoint formats:

    * **New** (multi-encoder): keys prefixed with ``encoders.<obj_name>.``
    * **Legacy** (single-encoder): keys prefixed with ``encoder.``

    Args:
        encoder:  The :class:`Encoder` instance to load weights into.
        ckpt_path: Path to a :class:`PretrainModule` Lightning checkpoint.
        obj_name:  Name of the constituent object whose encoder to extract.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    # Try new format first: encoders.<obj_name>.*
    prefix_new = f'encoders.{obj_name}.'
    enc_state = {
        k[len(prefix_new):]: v
        for k, v in state.items()
        if k.startswith(prefix_new)
    }

    # Fall back to legacy format: encoder.*
    if not enc_state:
        prefix_old = 'encoder.'
        enc_state = {
            k[len(prefix_old):]: v
            for k, v in state.items()
            if k.startswith(prefix_old)
        }

    if not enc_state:
        raise KeyError(
            f"Cannot find encoder weights for '{obj_name}' in checkpoint "
            f"'{ckpt_path}'.  Expected keys starting with "
            f"'encoders.{obj_name}.' (new format) or 'encoder.' (legacy)."
        )

    encoder.load_state_dict(enc_state)
    return encoder


def _build_callbacks(cfg: DictConfig, phase: str) -> list:
    """Build the list of Lightning callbacks from the config.

    ``phase`` is passed explicitly so this function does not need to read it
    from the config — it works correctly both when called from Hydra (where
    ``cfg.phase`` is already set) and when called from a notebook with an
    override.
    """
    cb_cfg = cfg.get('callbacks', {})
    callbacks = []

    # Choose the right monitor metric.  Fall back to train/loss when no
    # validation file is configured for the current phase.
    _val_key = 'pretrain_val_file' if phase == 'pretrain' else 'val_file'
    _has_val = bool(cfg.get(_val_key))
    _default_monitor = 'val_loss' if _has_val else 'train_loss'

    # ── ModelSummary ────────────────────────────────────────────────────────
    ms = cb_cfg.get('model_summary', {})
    callbacks.append(ModelSummary(max_depth=ms.get('max_depth', 2)))

    # ── ProgressBar ──────────────────────────────────────────────────────────
    pb = cb_cfg.get('progress_bar', {})
    callbacks.append(TQDMProgressBar(refresh_rate=pb.get('refresh_rate', 50)))

    # ── ModelCheckpoint ──────────────────────────────────────────────────────
    ckpt = cb_cfg.get('model_checkpoint', {})
    # When there is no val set, force train/loss regardless of the config value.
    _ckpt_monitor = (
        _default_monitor if not _has_val else ckpt.get('monitor', _default_monitor)
    )
    _metric_key = _ckpt_monitor.replace('/', '_')
    callbacks.append(
        ModelCheckpoint(
            monitor=_ckpt_monitor,
            save_top_k=ckpt.get('save_top_k', 3),
            mode=ckpt.get('mode', 'min'),
            save_last=True,
            filename='{epoch:03d}-{' + _metric_key + ':.4f}',
            verbose=True,
        )
    )

    # ── EarlyStopping ────────────────────────────────────────────────────────
    es = cb_cfg.get('early_stopping', {})
    _es_monitor = (
        _default_monitor if not _has_val else es.get('monitor', _default_monitor)
    )
    callbacks.append(
        EarlyStopping(
            monitor=_es_monitor,
            patience=es.get('patience', 15),
            mode=es.get('mode', 'min'),
            verbose=True,
            check_finite=True,
            # When monitoring a train metric there is no validation loop to hook into.
            check_on_train_epoch_end=not _has_val,
        )
    )

    # ── BackboneFinetuning (finetune phase + freeze_encoder only) ────────────
    if phase == 'finetune' and cfg.get('freeze_encoder', False):
        bf = cb_cfg.get('backbone_finetuning', {})
        if bf.get('enabled', True):
            callbacks.append(
                BackboneFinetuning(
                    unfreeze_backbone_at_epoch=bf.get('unfreeze_backbone_at_epoch', 10),
                    backbone_initial_ratio_lr=bf.get('backbone_initial_ratio_lr', 0.1),
                    train_bn=bf.get('train_bn', False),
                )
            )

    return callbacks


# ---------------------------------------------------------------------------
# Public API
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
    _phase = phase or cfg.get('phase', 'finetune')
    _action = action or cfg.get('action', 'fit')
    _enc_ckpt = encoder_ckpt or cfg.get('encoder_ckpt')
    _ckpt = ckpt_path or cfg.get('ckpt_path')

    L.seed_everything(cfg.get('seed', 42), workers=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = CSVLogger(
        save_dir=cfg.get('output_dir', 'outputs'),
        name=cfg.get('experiment_name', 'fm4tag'),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = _build_callbacks(cfg, _phase)

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
    if _phase == 'pretrain':
        encoders = _build_encoders(cfg)
        module: L.LightningModule = PretrainModule(encoders, cfg)

    elif _phase == 'finetune':
        # Finetune currently uses the first constituent encoder only.
        # (Multi-stream head support to be added later.)
        obj_name = list(cfg.constituent_objects)[0]
        encoder = _build_constituent_encoder(cfg, obj_name)

        if _enc_ckpt is not None:
            encoder = _load_pretrained_encoder(encoder, _enc_ckpt, obj_name=obj_name)

        head_cfg = cfg.head
        n_classes = len(cfg.variables[cfg.global_object].unique_labels)
        head = ClassifierHead(
            dim=encoder.dim,
            y_dim=n_classes,
            mlp_dropout=head_cfg.get('mlp_dropout', 0.0),
            ff_dropout=head_cfg.get('ff_dropout', 0.0),
            attn_dropout=head_cfg.get('attn_dropout', 0.0),
            ff_mult=head_cfg.get('ff_mult', 4),
            heads=head_cfg.get('heads', 8),
            dim_head=head_cfg.get('dim_head', 16),
            depth=head_cfg.get('depth', 3),
        )
        module = FinetuneModule(encoder, head, cfg)

    else:
        raise ValueError(f"phase must be 'pretrain' or 'finetune', got {_phase!r}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if _action == 'fit':
        trainer.fit(module, dm, ckpt_path=_ckpt)

    elif _action == 'test':
        trainer.test(module, dm, ckpt_path=_ckpt or 'best')

    elif _action == 'predict':
        predictions = trainer.predict(module, dm, ckpt_path=_ckpt or 'best')
        out_dir = logger.log_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(predictions, os.path.join(out_dir, 'predictions.pt'))

    else:
        raise ValueError(f"action must be 'fit', 'test', or 'predict', got {_action!r}")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='configs', config_name='default')
def main(cfg: DictConfig) -> None:
    run(cfg)
