"""Factory helpers that turn a resolved Hydra config into runtime objects.

These ``_build_*`` functions are used by :func:`fm4tag.runner.run.run` (and by
:mod:`fm4tag.eval_encoder`) to construct encoders, the jet aggregator, the
classifier head, Lightning callbacks, and the profiler from the config.
"""

from __future__ import annotations

import importlib
import warnings

import torch
from omegaconf import DictConfig

from lightning.pytorch.callbacks import (
    BackboneFinetuning,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)

from fm4tag.callbacks.callbacks import MemoryMonitorCallback, _PrecisionProgressBar
from fm4tag.models import JetAggregator, MultiStreamClassifierHead
from fm4tag.utils import instantiate, resolve_object_inputs


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
      (continuous count) and ``categories`` injected from the global object's
      variable definitions at runtime.  A legacy flat ``inputs`` list yields no
      categorical features, leaving the encoder identical to before.
    * ``cfg.backbone.constituents.<name>`` → e.g. :class:`~fm4tag.models.Encoder`
      (one per constituent type, with ``categories`` and ``num_continuous``
      injected from the variable definitions at runtime)
    """
    encoders: dict[str, torch.nn.Module] = {}

    global_name = cfg.global_object
    g_continuous, _, g_cat_classes = resolve_object_inputs(
        cfg.variables[global_name].inputs
    )
    g_categories = [len(classes) for classes in g_cat_classes.values()]
    encoders[global_name] = instantiate(
        cfg.backbone.global_encoder,
        num_features=len(g_continuous),
        categories=g_categories,
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


def _load_pretrained_aggregator(
    aggregator: JetAggregator,
    ckpt_path: str,
) -> JetAggregator:
    """Load aggregator weights from a pretraining checkpoint.

    Matches ``aggregator.*`` keys saved by a pretraining module.  These are
    meaningful only when the pretraining loss contained a jet-contrastive term
    (otherwise the aggregator received no gradient and its weights are the
    random init).  Emits a warning and leaves the aggregator randomly
    initialised if no matching weights are found or their shapes do not match
    the current architecture.

    Args:
        aggregator: :class:`JetAggregator` built by :func:`_build_aggregator`.
        ckpt_path:  Path to a pretraining-module Lightning checkpoint.
    """
    # A parameterless aggregator (e.g. no constituent objects → empty
    # const_transformer) has nothing to load; skip silently so the "no weights
    # found" warning below is reserved for genuine checkpoint mismatches.
    if sum(p.numel() for p in aggregator.parameters()) == 0:
        return aggregator

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    prefix = 'aggregator.'
    agg_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}

    if not agg_state:
        warnings.warn(
            f"No pretrained aggregator weights found in '{ckpt_path}'. "
            'Using random initialisation.',
            stacklevel=2,
        )
        return aggregator

    try:
        aggregator.load_state_dict(agg_state)
    except RuntimeError as exc:
        warnings.warn(
            f"Pretrained aggregator weights in '{ckpt_path}' do not match the "
            f'current aggregator architecture ({exc}). Using random '
            'initialisation.',
            stacklevel=2,
        )

    return aggregator


def _build_aggregator(
    cfg: DictConfig,
    encoders: torch.nn.ModuleDict,
) -> JetAggregator:
    """Build :class:`~fm4tag.models.JetAggregator` from cfg and built encoders.

    Output dims are read from the already-built encoders' projection heads::

        global_dim = encoders[cfg.global_object].projector.layers[-1].out_features
        const_dims = [encoders[obj].projector.layers[-1].out_features
                      for obj in cfg.constituent_objects]

    The transformer hyper-parameters (``depth``, ``heads``, ``dim_head``,
    ``ff_mult``, ``ff_dropout``, ``attn_dropout``) come from ``cfg.aggregator``.
    """
    global_dim = encoders[cfg.global_object].projector.layers[-1].out_features
    const_dims = [
        encoders[obj_name].projector.layers[-1].out_features
        for obj_name in cfg.constituent_objects
    ]

    agg_cfg = cfg.get('aggregator', {})
    return JetAggregator(
        global_dim=global_dim,
        const_dims=const_dims,
        depth=agg_cfg.get('depth', 3),
        heads=agg_cfg.get('heads', 8),
        dim_head=agg_cfg.get('dim_head', 16),
        ff_mult=agg_cfg.get('ff_mult', 4),
        ff_dropout=agg_cfg.get('ff_dropout', 0.0),
        attn_dropout=agg_cfg.get('attn_dropout', 0.0),
    )


def _build_head(
    cfg: DictConfig,
    aggregator: JetAggregator,
) -> MultiStreamClassifierHead:
    """Build the classifier head from ``cfg.head`` and the aggregator.

    ``jet_dim`` is taken from ``aggregator.out_dim`` and ``y_dim`` from the
    number of unique labels of the global object; all other parameters come
    from ``cfg.head``.
    """
    jet_dim = aggregator.out_dim
    y_dim = len(cfg.variables[cfg.global_object].unique_labels)

    head_cfg = cfg.get('head', {})
    return MultiStreamClassifierHead(
        jet_dim=jet_dim,
        y_dim=y_dim,
        mlp_hidden=head_cfg.get('mlp_hidden', None),
        mlp_dropout=head_cfg.get('mlp_dropout', 0.0),
    )


def _build_callbacks(cfg: DictConfig, phase: str) -> list:
    """Build the list of Lightning callbacks from the config.

    ``phase`` is passed explicitly so this function does not need to read it
    from the config — it works correctly both when called from Hydra (where
    ``cfg.phase`` is already set) and when called from a notebook with an
    override.
    """
    cb_cfg = cfg.get('callbacks', {})
    callbacks = []

    # Both pretrain and finetune build the val dataloader from val_dataset_path
    # (see datamodule.setup); use that same key here so the monitor falls back
    # to val_loss when a validation set is configured.
    _has_val = bool(cfg.get('val_dataset_path'))
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
