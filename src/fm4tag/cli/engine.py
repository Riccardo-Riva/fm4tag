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

    # Load a config file from OUTSIDE the repo's configs directory:
    #   --config-path (-cp) sets the search directory (absolute, or relative to CWD)
    #   --config-name (-cn) is the filename without .yaml
    fm4tag --config-path=/path/to/my/configs --config-name=my_experiment phase=finetune
    fm4tag -cp /path/to/my/configs -cn my_experiment phase=finetune action=predict

    # On every run the fully-resolved config is written to:
    #   outputs/<experiment_name>/version_N/config.yaml

For notebooks / scripts (Hydra not involved)::

    from omegaconf import OmegaConf
    from fm4tag.engine import run

    cfg = OmegaConf.load('src/fm4tag/configs/saintV0.yaml')
    run(cfg, phase='pretrain', action='fit')
"""

from __future__ import annotations

import importlib
import os
import warnings

import hydra
import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.callbacks import (
    BackboneFinetuning,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)

from fm4tag.callbacks.callbacks import MemoryMonitorCallback, _PrecisionProgressBar
from fm4tag.datamodules.datamodule import CatConDataModule
from fm4tag.modules.finetune_module import FinetuneModule
from fm4tag.utils import instantiate


def _load_class(dotted_path: str) -> type:
    """Import and return a class from its fully-qualified dotted path."""
    module_path, cls_name = dotted_path.rsplit('.', 1)
    return getattr(importlib.import_module(module_path), cls_name)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _build_encoders(cfg: DictConfig) -> torch.nn.ModuleDict:
    """Build one encoder per object (global + all constituents).

    Reads architecture from ``cfg.backbone``; each encoder's class is selected
    by its ``_target_`` key:

    * ``cfg.backbone.global_encoder`` → e.g. :class:`~fm4tag.models.GlobalEncoder`
      or :class:`~fm4tag.models.GlobalTransformerEncoder`, with ``num_features``
      injected from the global object's variable definitions at runtime
    * ``cfg.backbone.constituents.<name>`` → e.g. :class:`~fm4tag.models.Encoder`
      (one per constituent type, with ``categories`` and ``num_continuous``
      injected from the variable definitions at runtime)
    """
    encoders: dict[str, torch.nn.Module] = {}

    global_name = cfg.global_object
    n_global = len(cfg.variables[global_name].inputs)
    encoders[global_name] = instantiate(
        cfg.backbone.global_encoder, num_features=n_global
    )

    for obj_name in cfg.constituent_objects:
        obj_vars = cfg.variables[obj_name].inputs
        categories = [len(classes) for classes in obj_vars.cat_classes.values()]
        num_continuous = len(obj_vars.continuous)
        encoders[obj_name] = instantiate(
            cfg.backbone.constituents[obj_name],
            categories=categories,
            num_continuous=num_continuous,
        )

    return torch.nn.ModuleDict(encoders)


def _load_pretrained_encoders(
    encoders: torch.nn.ModuleDict,
    ckpt_path: str,
) -> torch.nn.ModuleDict:
    """Load all encoder weights from a :class:`PretrainModule` checkpoint.

    For each encoder in the dict, tries the new checkpoint format
    (``encoders.<obj_name>.*``) first, then falls back to the legacy
    single-encoder format (``encoder.*``).  Emits a warning for any
    encoder whose weights cannot be found, leaving it randomly initialised.

    Args:
        encoders:  :class:`~torch.nn.ModuleDict` built by :func:`_build_encoders`.
        ckpt_path: Path to a :class:`PretrainModule` Lightning checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    for obj_name, encoder in encoders.items():
        prefix_new = f'encoders.{obj_name}.'
        enc_state = {
            k[len(prefix_new) :]: v
            for k, v in state.items()
            if k.startswith(prefix_new)
        }

        if not enc_state:
            prefix_old = 'encoder.'
            enc_state = {
                k[len(prefix_old) :]: v
                for k, v in state.items()
                if k.startswith(prefix_old)
            }

        if enc_state:
            encoder.load_state_dict(enc_state)
        else:
            warnings.warn(
                f"No pretrained weights found for encoder '{obj_name}' in "
                f"'{ckpt_path}'. Using random initialisation.",
                stacklevel=2,
            )

    return encoders


def _build_callbacks(cfg: DictConfig, phase: str) -> list:
    """Build the list of Lightning callbacks from the config.

    ``phase`` is passed explicitly so this function does not need to read it
    from the config — it works correctly both when called from Hydra (where
    ``cfg.phase`` is already set) and when called from a notebook with an
    override.
    """
    cb_cfg = cfg.get('callbacks', {})
    callbacks = []

    _val_key = 'pretrain_val_file' if phase == 'pretrain' else 'val_file'
    _has_val = bool(cfg.get(_val_key))
    _default_monitor = 'val_loss' if _has_val else 'train_loss'

    ms = cb_cfg.get('model_summary', {})
    callbacks.append(ModelSummary(max_depth=ms.get('max_depth', 2)))

    pb = cb_cfg.get('progress_bar', {})
    callbacks.append(_PrecisionProgressBar(refresh_rate=pb.get('refresh_rate', 50)))

    ckpt = cb_cfg.get('model_checkpoint', {})
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
            check_on_train_epoch_end=not _has_val,
        )
    )

    callbacks.append(LearningRateMonitor(logging_interval='step'))

    mm = cb_cfg.get('memory_monitor', {})
    if mm.get('enabled', False):
        log_every = mm.get(
            'log_every_n_steps', cfg.get('trainer', {}).get('log_every_n_steps', 100)
        )
        callbacks.append(MemoryMonitorCallback(log_every_n_steps=log_every))

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
# Profiler builder
# ---------------------------------------------------------------------------


def _build_profiler(cfg: DictConfig):
    """Build a Lightning profiler from config, or return ``None`` (disabled).

    Three profiler types are supported (``profiler.type``):

    * ``simple``   — wall-clock table per Lightning hook; near-zero overhead.
                     Good first-pass to spot which hooks dominate training time.
    * ``advanced`` — Python ``cProfile`` per hook; function-level call graph.
                     Use when ``simple`` points to a specific hook you want to
                     drill into.
    * ``pytorch``  — ``torch.profiler`` traces with GPU + CPU op-level timing
                     and optional memory stats.  Writes a Chrome trace JSON to
                     the log directory for visualisation in chrome://tracing or
                     https://ui.perfetto.dev/.

    Enable from the CLI without editing any YAML::

        fm4tag profiler.enabled=true profiler.type=simple
        fm4tag profiler.enabled=true profiler.type=pytorch profiler.profile_memory=true
    """
    p_cfg = cfg.get('profiler', {})
    if not p_cfg.get('enabled', False):
        return None

    ptype = p_cfg.get('type', 'simple')
    row_limit = p_cfg.get('row_limit', 25)

    if ptype == 'simple':
        return SimpleProfiler(filename='profiler-simple', extended=True)

    if ptype == 'advanced':
        return AdvancedProfiler(
            filename='profiler-advanced',
            line_count_restriction=row_limit,
        )

    if ptype == 'pytorch':
        return PyTorchProfiler(
            filename='profiler-pytorch',
            export_to_chrome=p_cfg.get('export_to_chrome', True),
            with_stack=p_cfg.get('with_stack', False),
            profile_memory=p_cfg.get('profile_memory', False),
            row_limit=row_limit,
        )

    raise ValueError(
        f"profiler.type must be 'simple', 'advanced', or 'pytorch', got {ptype!r}"
    )


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
    extra_callbacks: list | None = None,
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
        extra_callbacks: Additional Lightning callbacks appended after the
                      standard set built from config.  Used by the HPO module
                      to inject :class:`~fm4tag.hpo._OptunaMetricCallback`.
    """
    _phase = phase or cfg.get('phase', 'finetune')
    _action = action or cfg.get('action', 'fit')
    _enc_ckpt = encoder_ckpt or cfg.get('encoder_ckpt')
    _ckpt = ckpt_path or cfg.get('ckpt_path')

    torch.set_float32_matmul_precision('high')
    L.seed_everything(cfg.get('seed', 42), workers=True)

    tb_logger = TensorBoardLogger(
        save_dir=cfg.get('output_dir', 'outputs'),
        name=cfg.get('experiment_name', 'fm4tag'),
    )
    csv_logger = CSVLogger(
        save_dir=cfg.get('output_dir', 'outputs'),
        name=cfg.get('experiment_name', 'fm4tag'),
        version=tb_logger.version,
    )
    logger = [tb_logger, csv_logger]

    """
    # ── Save resolved config ──────────────────────────────────────────────────
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    with open(os.path.join(tb_logger.log_dir, 'config.yaml'), 'w') as _f:
        _f.write(OmegaConf.to_yaml(cfg, resolve=True))
    """

    callbacks = _build_callbacks(cfg, _phase)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    profiler = _build_profiler(cfg)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
        **trainer_kwargs,
    )

    dl_cfg = cfg.dataloader
    dm = CatConDataModule(
        train_dataset_path=cfg.train_dataset_path,
        val_dataset_path=cfg.val_dataset_path,
        test_dataset_path=cfg.test_dataset_path,
        variables=cfg.variables,
        global_object=cfg.global_object,
        constituent_objects=list(cfg.constituent_objects),
        norm_dict_path=cfg.get('norm_dict_path'),
        class_dict_path=cfg.get('class_dict_path'),
        batch_size=dl_cfg.batch_size,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.get('prefetch_factor', 2),
        pin_memory=dl_cfg.get('pin_memory', True),
    )

    if _phase == 'pretrain':
        encoders = _build_encoders(cfg)
        # Build view pipelines from the config's _target_-annotated list.
        views = [hydra_instantiate(v) for v in cfg.pretrain.views]
        # Select the module class via _target_ and instantiate directly
        # (encoders and cfg are runtime objects, not from YAML).
        module_cls = _load_class(cfg.pretrain._target_)
        module: L.LightningModule = module_cls(encoders=encoders, views=views, cfg=cfg)

    elif _phase == 'finetune':
        encoders = _build_encoders(cfg)

        if _enc_ckpt is not None:
            _load_pretrained_encoders(encoders, _enc_ckpt)

        n_classes = len(cfg.variables[cfg.global_object].unique_labels)

        # Read projection dimensions from the built encoders.
        global_proj_out = encoders[cfg.global_object].projector.layers[-1].out_features
        const_proj_outs = [
            encoders[obj_name].projector.layers[-1].out_features
            for obj_name in cfg.constituent_objects
        ]

        head = instantiate(
            cfg.head,
            global_proj_out=global_proj_out,
            const_proj_outs=const_proj_outs,
            y_dim=n_classes,
        )

        module = FinetuneModule(encoders, head, cfg)

    else:
        raise ValueError(f"phase must be 'pretrain' or 'finetune', got {_phase!r}")

    # weights_only=False: checkpoints saved by older Lightning embed omegaconf
    # objects (DictConfig, ContainerMetadata, …) which PyTorch 2.6+ rejects
    # under the new weights_only=True default.  The checkpoints are our own
    # files so this is safe.
    if _action == 'fit':
        trainer.fit(module, dm, ckpt_path=_ckpt, weights_only=False)

    elif _action == 'test':
        trainer.test(module, dm, ckpt_path=_ckpt or 'best', weights_only=False)

    elif _action == 'predict':
        predictions = trainer.predict(
            module, dm, ckpt_path=_ckpt or 'best', weights_only=False
        )
        out_dir = tb_logger.log_dir
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
