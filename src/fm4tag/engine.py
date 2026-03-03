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

import os
import warnings

import hydra
import psutil
import torch
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.callbacks import (
    BackboneFinetuning,
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)

from fm4tag.data import PT_FT_DataModule
from fm4tag.models import FinetuneModule, PretrainModule
from fm4tag.models.components.encoder import Encoder, GlobalEncoder
from fm4tag.models.components.heads import MultiStreamClassifierHead


# ---------------------------------------------------------------------------
# Memory monitor callback
# ---------------------------------------------------------------------------


class MemoryMonitorCallback(Callback):
    """Log CPU RSS and GPU VRAM usage each training step and validation epoch.

    Metrics logged (all in MiB):

    * ``mem/cpu_rss_MiB``  — resident set size of the main process (CPU RAM)
    * ``mem/gpu_alloc_MiB`` — GPU memory currently allocated by PyTorch tensors
    * ``mem/gpu_reserved_MiB`` — GPU memory reserved (cached) by the allocator

    Enable from config::

        callbacks:
          memory_monitor:
            enabled: true
            log_every_n_steps: 100   # optional, defaults to trainer.log_every_n_steps
    """

    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        self._log_every_n_steps = log_every_n_steps
        self._proc = psutil.Process(os.getpid())

    def _mem_stats(self) -> dict[str, float]:
        rss_mib = self._proc.memory_info().rss / 1024**2
        stats = {'mem/cpu_rss_MiB': rss_mib}
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            stats['mem/gpu_alloc_MiB'] = torch.cuda.memory_allocated(dev) / 1024**2
            stats['mem/gpu_reserved_MiB'] = torch.cuda.memory_reserved(dev) / 1024**2
        return stats

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self._log_every_n_steps == 0:
            pl_module.log_dict(self._mem_stats(), on_step=True, on_epoch=False)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        pl_module.log_dict(self._mem_stats(), on_step=False, on_epoch=True)


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
    encoders[global_name] = GlobalEncoder(
        num_features=n_global,
        dim=enc_cfg.dim,
    )

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
            proj_hidden=enc_cfg.get('proj_hidden', None),
            proj_out=enc_cfg.get('proj_out', None),
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
        # Try new format: encoders.<obj_name>.*
        prefix_new = f'encoders.{obj_name}.'
        enc_state = {
            k[len(prefix_new) :]: v
            for k, v in state.items()
            if k.startswith(prefix_new)
        }

        # Fall back to legacy single-encoder format: encoder.*
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


class _PrecisionProgressBar(TQDMProgressBar):
    """TQDMProgressBar that formats floats with 4 decimal places.

    Lightning's default ``Tqdm.format_num`` uses ``.3g`` (3 significant
    figures), which for loss values in the range 1-9 only preserves 2
    decimal digits, then pads with a trailing zero to a fixed width.
    Pre-converting floats to strings here bypasses that pipeline.
    """

    def get_metrics(self, trainer, pl_module):  # type: ignore[override]
        metrics = super().get_metrics(trainer, pl_module)
        return {
            k: f'{v:.4f}' if isinstance(v, float) else v
            for k, v in metrics.items()
            if not k.endswith('_step')
        }


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
    callbacks.append(_PrecisionProgressBar(refresh_rate=pb.get('refresh_rate', 50)))

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

    # ── LearningRateMonitor ───────────────────────────────────────────────────
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # ── MemoryMonitor ────────────────────────────────────────────────────────
    mm = cb_cfg.get('memory_monitor', {})
    if mm.get('enabled', False):
        log_every = mm.get(
            'log_every_n_steps', cfg.get('trainer', {}).get('log_every_n_steps', 100)
        )
        callbacks.append(MemoryMonitorCallback(log_every_n_steps=log_every))

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
        # Saves a human-readable table to <log_dir>/profiler-simple.txt
        return SimpleProfiler(filename='profiler-simple', extended=True)

    if ptype == 'advanced':
        # Saves cProfile output to <log_dir>/profiler-advanced.txt
        return AdvancedProfiler(
            filename='profiler-advanced',
            line_count_restriction=row_limit,
        )

    if ptype == 'pytorch':
        # Saves a Chrome trace to <log_dir>/profiler-pytorch.json (or .txt table)
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

    torch.set_float32_matmul_precision('high')
    L.seed_everything(cfg.get('seed', 42), workers=True)

    # ── Logger ────────────────────────────────────────────────────────────────
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

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = _build_callbacks(cfg, _phase)

    # ── Profiler ──────────────────────────────────────────────────────────────
    profiler = _build_profiler(cfg)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
        **trainer_kwargs,
    )

    # ── Data module ───────────────────────────────────────────────────────────
    dm = PT_FT_DataModule(cfg, phase=_phase)

    # ── Lightning module ──────────────────────────────────────────────────────
    if _phase == 'pretrain':
        encoders = _build_encoders(cfg)
        module: L.LightningModule = PretrainModule(encoders, cfg)

    elif _phase == 'finetune':
        encoders = _build_encoders(cfg)

        if _enc_ckpt is not None:
            _load_pretrained_encoders(encoders, _enc_ckpt)

        head_cfg = cfg.head
        n_classes = len(cfg.variables[cfg.global_object].unique_labels)

        n_global_features = len(cfg.variables[cfg.global_object].inputs)
        n_constituent_features = [
            len(cfg.variables[obj].inputs.continuous)
            + len(cfg.variables[obj].inputs.categorical)
            for obj in cfg.constituent_objects
        ]

        head = MultiStreamClassifierHead(
            dim=cfg.encoder.dim,
            n_global_features=n_global_features,
            n_constituent_features=n_constituent_features,
            y_dim=n_classes,
            cls_dim=head_cfg.get('cls_dim', None),
            mlp_dropout=head_cfg.get('mlp_dropout', 0.0),
            ff_dropout=head_cfg.get('ff_dropout', 0.0),
            attn_dropout=head_cfg.get('attn_dropout', 0.0),
            ff_mult=head_cfg.get('ff_mult', 4),
            heads=head_cfg.get('heads', 8),
            dim_head=head_cfg.get('dim_head', 16),
            depth=head_cfg.get('depth', 3),
        )
        module = FinetuneModule(encoders, head, cfg)

    else:
        raise ValueError(f"phase must be 'pretrain' or 'finetune', got {_phase!r}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
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
