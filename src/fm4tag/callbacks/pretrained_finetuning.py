"""Freeze/unfreeze schedule for the pretrained parts during fine-tuning."""

from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer


class PretrainedFinetuning(BaseFinetuning):
    """Freeze the pretrained parts (encoders **and** aggregator) initially.

    Like Lightning's :class:`~lightning.pytorch.callbacks.BackboneFinetuning`,
    but freezes ``pl_module.backbone`` *and* ``pl_module.aggregator`` — both
    carry pretrained weights, so both are protected while the randomly
    initialised classifier head warms up.

    :class:`~fm4tag.modules.FinetuneModule.configure_optimizers` detects this
    callback and excludes both parts from the initial optimiser; this callback
    adds them back as a new param group at ``unfreeze_at_epoch``.

    Set ``unfreeze_at_epoch >= trainer.max_epochs`` to keep the pretrained
    parts frozen for the whole run.  Remove the callback from the config to
    train everything from epoch 0.

    Args:
        unfreeze_at_epoch: Epoch at which the pretrained parts are unfrozen
                           and added to the optimiser.
        initial_ratio_lr:  Their learning rate at unfreeze time, as a fraction
                           of the head's current learning rate.
        train_bn:          Keep BatchNorm layers trainable while frozen.
    """

    def __init__(
        self,
        unfreeze_at_epoch: int = 10,
        initial_ratio_lr: float = 0.1,
        train_bn: bool = False,
    ) -> None:
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.initial_ratio_lr = initial_ratio_lr
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: L.LightningModule) -> None:
        self.freeze(
            [pl_module.backbone, pl_module.aggregator], train_bn=self.train_bn
        )

    def finetune_function(
        self,
        pl_module: L.LightningModule,
        current_epoch: int,
        optimizer: Optimizer,
    ) -> None:
        if current_epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=[pl_module.backbone, pl_module.aggregator],
                optimizer=optimizer,
                initial_denom_lr=1.0 / self.initial_ratio_lr,
                train_bn=self.train_bn,
            )
