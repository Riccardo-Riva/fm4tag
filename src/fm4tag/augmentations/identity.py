"""Identity augmentation — pass through unchanged.

Useful as the first view in a multi-view contrastive setup where you want
to preserve one unaugmented view alongside corrupted ones (matches the
original SimCLR setup with a clean ``view 1``).

The ``stage`` attribute can be set via the constructor since identity is
trivially valid in any stage; defaulting to :attr:`Stage.RAW` keeps it cheap.
"""

from __future__ import annotations

import torch

from .base import Augmentation, Stage, register


@register('identity')
class Identity(Augmentation):
    """No-op augmentation."""

    stage = Stage.RAW

    def __init__(self, stage: str | Stage = Stage.RAW) -> None:
        super().__init__()
        # Allow per-instance override so a user can place Identity in any stage
        # without subclassing.
        self.stage = Stage(stage) if isinstance(stage, str) else stage

    def forward(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        # Return a shallow copy so the caller can't accidentally see this
        # augmentation as identity-with-aliasing.
        return dict(data)
