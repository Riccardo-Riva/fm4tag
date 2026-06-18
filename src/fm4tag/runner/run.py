"""Public ``run()`` entry point: build everything from config and drive Lightning.

This is the framework-agnostic workflow used by both the Hydra CLI
(:mod:`fm4tag.cli.engine`) and notebooks/scripts.  It builds the data module,
encoders, jet aggregator, loss, (optionally) the classifier head and views, then
runs ``fit`` / ``test`` / ``predict``.
"""

from __future__ import annotations

import os

import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from fm4tag.datamodules.datamodule import CatConDataModule
from fm4tag.modules.finetune_module import FinetuneModule

from .builders import (
    _build_aggregator,
    _build_callbacks,
    _build_encoders,
    _build_head,
    _build_profiler,
    _load_class,
    _load_pretrained_aggregator,
    _load_pretrained_encoders,
)


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
        persistent_workers=dl_cfg.get('persistent_workers', False)
    )

    if _phase == 'pretrain':
        encoders = _build_encoders(cfg)
        aggregator = _build_aggregator(cfg, encoders)
        # Build view pipelines from the config's _target_-annotated list.
        views = [hydra_instantiate(v) for v in cfg.pretrain.views]
        # Build the composable loss (terms instantiated recursively by Hydra).
        loss = hydra_instantiate(cfg.pretrain.loss)
        # Select the module class via _target_ and instantiate directly
        # (runtime objects are not built from YAML).
        module_cls = _load_class(cfg.pretrain._target_)
        module: L.LightningModule = module_cls(
            encoders=encoders,
            aggregator=aggregator,
            views=views,
            loss=loss,
            cfg=cfg,
        )

    elif _phase == 'finetune':
        encoders = _build_encoders(cfg)
        aggregator = _build_aggregator(cfg, encoders)
        head = _build_head(cfg, aggregator)
        loss = hydra_instantiate(cfg.finetune.loss)

        if _enc_ckpt is not None:
            _load_pretrained_encoders(encoders, _enc_ckpt)
            # The aggregator is shared with pretraining; load it too unless
            # disabled (meaningful only if pretraining trained it).
            if cfg.get('load_aggregator', True):
                _load_pretrained_aggregator(aggregator, _enc_ckpt)

        # Views are only needed for a jet-contrastive term; default to none.
        views = [hydra_instantiate(v) for v in cfg.finetune.get('views', [])]

        module = FinetuneModule(
            encoders=encoders,
            aggregator=aggregator,
            head=head,
            loss=loss,
            views=views,
            cfg=cfg,
        )

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
