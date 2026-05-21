"""Uniform continuous feature dilation — multiply all continuous features by a
scalar.

A simple multiplicative augmentation that stretches or shrinks the entire
continuous feature vector uniformly.  Categorical features are unchanged.
"""

from __future__ import annotations

import torch

from .base import Augmentation, Stage


class ContinuousDilation(Augmentation):
    """Scale all continuous features by a constant factor.

    Args:
        alpha: Scale factor applied to every continuous feature value.
    """

    stage = Stage.RAW

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        x_cont = data.get('continuous')
        if x_cont is None:
            return dict(data)
        out = dict(data)
        out['continuous'] = x_cont * self.alpha
        return out
