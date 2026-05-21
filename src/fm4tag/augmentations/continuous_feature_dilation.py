"""Per-feature continuous dilation — multiply a named subset of continuous
features by a scalar.

Feature names are resolved to column indices via :meth:`setup`, which must be
called once (by the training module or manually) before the first forward pass.
Features absent from the dataset's continuous-feature list are silently skipped.
"""

from __future__ import annotations

import torch

from .base import Augmentation, Stage


class ContinuousFeatureDilation(Augmentation):
    """Scale a named subset of continuous features by a constant factor.

    Args:
        features: Names of continuous features to dilate.
        alpha:    Scale factor applied to the selected features.
    """

    stage = Stage.RAW

    def __init__(self, features: list[str], alpha: float) -> None:
        super().__init__()
        self.feature_names = list(features)
        self.alpha = alpha
        self._indices: list[int] = []

    def setup(self, continuous_features: list[str], **kwargs) -> None:
        """Resolve feature names to column indices.

        Args:
            continuous_features: Ordered list of continuous feature names for
                the current dataset/object.
        """
        self._indices = [
            continuous_features.index(f)
            for f in self.feature_names
            if f in continuous_features
        ]

    def forward(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        x_cont = data.get('continuous')
        if x_cont is None or not self._indices:
            return dict(data)
        idx = torch.tensor(self._indices, device=x_cont.device)
        out = dict(data)
        x = x_cont.clone()
        x[..., idx] = x[..., idx] * self.alpha
        out['continuous'] = x
        return out
