"""Categorical feature shift — randomly nudge each class index by ±1.

Each categorical feature element is independently shifted by −1, 0, or +1
with probability ``p``, then clamped to the valid range ``[0, n_classes - 1]``.

Requires :meth:`setup` to be called once with the per-feature class counts so
that the clamp bounds are correct.  Before setup (or when ``x_categ`` is
``None``), the augmentation is a no-op.
"""

from __future__ import annotations

import torch

from .base import Augmentation, Stage


class CategoricalShift(Augmentation):
    """Randomly shift categorical feature values by ±1.

    Args:
        p: Per-element shift probability in ``[0, 1]``.
    """

    stage = Stage.RAW

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'p must be in [0, 1], got {p}')
        self.p = p
        self._max_vals: torch.Tensor | None = None

    def setup(self, n_classes: list[int], **kwargs) -> None:
        """Store max valid class index per feature.

        Args:
            n_classes: Number of classes for each categorical feature, in
                column order.
        """
        if n_classes:
            self._max_vals = torch.tensor([n - 1 for n in n_classes], dtype=torch.long)

    def forward(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        x_categ = data.get('categorical')
        if x_categ is None or self._max_vals is None:
            return dict(data)

        max_vals = self._max_vals.to(x_categ.device)
        delta = torch.randint(-1, 2, x_categ.shape, device=x_categ.device)
        mask = torch.bernoulli(
            self.p * torch.ones_like(x_categ, dtype=torch.float32)
        ).bool()
        min_vals = torch.zeros_like(max_vals)
        shifted = torch.clamp(x_categ + delta * mask, min=min_vals, max=max_vals)

        out = dict(data)
        out['categorical'] = shifted
        return out
