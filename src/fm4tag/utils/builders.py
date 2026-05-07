"""Builder utilities shared between pretrain and finetune CLI scripts."""

from __future__ import annotations

import warnings

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.pytorch.callbacks import (
    BackboneFinetuning,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler

from fm4tag.augmentations import AugmentationPipeline, MultiViewAugmentation
from fm4tag.models.components.encoder import Encoder, GlobalEncoder
from fm4tag.utils.callbacks import MemoryMonitorCallback, PrecisionProgressBar


# ---------------------------------------------------------------------------
# Encoder builder
# ---------------------------------------------------------------------------


def build_encoders(cfg: DictConfig) -> torch.nn.ModuleDict:
    """Build one encoder per object (global + all constituents).

    Returns a :class:`~torch.nn.ModuleDict` keyed by object name:

    * ``cfg.global_object``       → :class:`GlobalEncoder` (per-feature MLP)
    * each ``cfg.constituent_objects`` → :class:`Encoder` (transformer)
    """
    enc_cfg = cfg.encoder
    encoders: dict[str, torch.nn.Module] = {}

    global_name = cfg.global_object
    n_global = len(cfg.variables[global_name].inputs)
    encoders[global_name] = GlobalEncoder(
        num_features=n_global,
        dim=enc_cfg.dim,
    )

    for obj_name in cfg.constituent_objects:
        obj_vars = cfg.variables[obj_name].inputs
        categories = [len(classes) for classes in obj_vars.cat_classes.values()]
        num_continuous = len(obj_vars.continuous)
        encoders[obj_name] = Encoder(
            categories=categories,
            num_continuous=num_continuous,
            dim=enc_cfg.dim,
            depth=enc_cfg.depth,
            col_heads=enc_cfg.get('col_heads', 8),
            row_heads=enc_cfg.get('row_heads', 8),
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


def load_pretrained_encoders(
    encoders: torch.nn.ModuleDict,
    ckpt_path: str,
) -> torch.nn.ModuleDict:
    """Load all encoder weights from a :class:`PretrainModule` checkpoint.

    Tries the new format (``encoders.<obj_name>.*``) then falls back to the
    legacy single-encoder format (``encoder.*``).  Warns for missing encoders.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    for obj_name, encoder in encoders.items():
        prefix_new = f'encoders.{obj_name}.'
        enc_state = {k[len(prefix_new):]: v for k, v in state.items() if k.startswith(prefix_new)}

        if not enc_state:
            prefix_old = 'encoder.'
            enc_state = {k[len(prefix_old):]: v for k, v in state.items() if k.startswith(prefix_old)}

        if enc_state:
            encoder.load_state_dict(enc_state)
        else:
            warnings.warn(
                f"No pretrained weights found for encoder '{obj_name}' in "
                f"'{ckpt_path}'. Using random initialisation.",
                stacklevel=2,
            )

    return encoders


# ---------------------------------------------------------------------------
# Augmentation pipeline builder
# ---------------------------------------------------------------------------


def build_aug_module(
    cfg: DictConfig,
    dataset=None,
) -> MultiViewAugmentation:
    """Build a :class:`MultiViewAugmentation` from ``cfg.augmentation``.

    The augmentation config must contain a ``views`` list.  Each entry
    describes one augmented view and may have ``raw`` and ``latent`` sublists,
    each with ``_target_``-keyed dicts consumed by ``hydra.utils.instantiate``.

    If *dataset* is provided, :meth:`~MultiViewAugmentation.setup_for_dataset`
    is called immediately so that object-aware augmentations
    (:class:`~fm4tag.augmentations.ContinuousFeatureDilation`,
    :class:`~fm4tag.augmentations.CategoricalShift`) have their per-object
    feature indices resolved before training begins.

    Args:
        cfg:     Fully resolved Hydra config containing ``cfg.augmentation``.
        dataset: Optional :class:`~fm4tag.datasets.DatasetCatCon` instance
                 used to resolve feature names to column indices.
    """
    aug_cfg = cfg.get('augmentation', {})
    pipelines = []
    for view_cfg in aug_cfg.get('views', []):
        raw_augs = [instantiate(a) for a in view_cfg.get('raw', [])]
        latent_augs = [instantiate(a) for a in view_cfg.get('latent', [])]
        pipelines.append(AugmentationPipeline(raw=raw_augs, latent=latent_augs))
    module = MultiViewAugmentation(pipelines)
    if dataset is not None:
        module.setup_for_dataset(dataset)
    return module


def build_aug_pipeline(cfg: DictConfig) -> AugmentationPipeline:
    """Build a single :class:`AugmentationPipeline` from ``cfg.augmentation``.

    .. deprecated::
        Use :func:`build_aug_module` with the ``views:`` config format instead.
        This function is kept for backward compatibility with code that passes
        an ``AugmentationPipeline`` directly to :class:`PretrainModule`.

    Accepts both the old flat format (``raw:`` / ``latent:`` at top level) and
    the new ``views:`` format (returns the first view's pipeline).
    """
    aug_cfg = cfg.get('augmentation', {})
    if 'views' in aug_cfg and aug_cfg['views']:
        view_cfg = aug_cfg['views'][0]
        raw_augs = [instantiate(a) for a in view_cfg.get('raw', [])]
        latent_augs = [instantiate(a) for a in view_cfg.get('latent', [])]
    else:
        raw_augs = [instantiate(a) for a in aug_cfg.get('raw', [])]
        latent_augs = [instantiate(a) for a in aug_cfg.get('latent', [])]
    return AugmentationPipeline(raw=raw_augs, latent=latent_augs)


# ---------------------------------------------------------------------------
# Callback builder
# ---------------------------------------------------------------------------


def build_callbacks(cfg: DictConfig, phase: str) -> list:
    """Build the list of Lightning callbacks from the config.

    Args:
        cfg:   Fully resolved config.
        phase: ``"pretrain"`` or ``"finetune"``.
    """
    cb_cfg = cfg.get('callbacks', {})
    callbacks = []

    _val_key = 'pretrain_val_file' if phase == 'pretrain' else 'val_file'
    _has_val = bool(cfg.get(_val_key))
    _default_monitor = 'val_loss' if _has_val else 'train_loss'

    ms = cb_cfg.get('model_summary', {})
    callbacks.append(ModelSummary(max_depth=ms.get('max_depth', 2)))

    pb = cb_cfg.get('progress_bar', {})
    callbacks.append(PrecisionProgressBar(refresh_rate=pb.get('refresh_rate', 50)))

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


def build_profiler(cfg: DictConfig):
    """Build a Lightning profiler from config, or return ``None`` (disabled)."""
    p_cfg = cfg.get('profiler', {})
    if not p_cfg.get('enabled', False):
        return None

    ptype = p_cfg.get('type', 'simple')
    row_limit = p_cfg.get('row_limit', 25)

    if ptype == 'simple':
        return SimpleProfiler(filename='profiler-simple', extended=True)
    if ptype == 'advanced':
        return AdvancedProfiler(filename='profiler-advanced', line_count_restriction=row_limit)
    if ptype == 'pytorch':
        return PyTorchProfiler(
            filename='profiler-pytorch',
            export_to_chrome=p_cfg.get('export_to_chrome', True),
            with_stack=p_cfg.get('with_stack', False),
            profile_memory=p_cfg.get('profile_memory', False),
            row_limit=row_limit,
        )

    raise ValueError(f"profiler.type must be 'simple', 'advanced', or 'pytorch', got {ptype!r}")
