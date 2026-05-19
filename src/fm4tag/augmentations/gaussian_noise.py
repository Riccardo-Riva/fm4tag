"""Additive Gaussian noise on continuous features (or embeddings).

The same module can be placed in either :attr:`Stage.RAW` (noise on raw
continuous feature values, before embedding) or :attr:`Stage.EMBEDDING`
(noise on embedded tokens, after ``embed_data``).

Categorical features are left untouched in both cases — adding Gaussian
noise to a long-valued category index is meaningless.  In the embedding
stage, however, noise IS applied to embedded categorical tokens (they are
floats at that point).
"""

from __future__ import annotations

import torch

from .base import Augmentation, Stage


class GaussianNoise(Augmentation):
    """Add zero-mean Gaussian noise to feature tensors.

    Args:
        sigma:           Standard deviation of the added noise.
        space:           ``"raw"`` or ``"embedding"`` — which stage to place
                         this augmentation in.
        apply_to_categ:  Whether to add noise to the categorical key.
                         In the raw stage, categorical tensors are integer
                         class indices, so this is forced to ``False``
                         regardless of the constructor argument.
        apply_to_cont:   Whether to add noise to the continuous key.
    """

    def __init__(
        self,
        sigma: float = 0.05,
        space: str | Stage = Stage.RAW,
        apply_to_categ: bool = True,
        apply_to_cont: bool = True,
    ) -> None:
        super().__init__()
        if sigma < 0.0:
            raise ValueError(f'sigma must be >= 0, got {sigma}')
        self.sigma = sigma
        # space is a per-instance override (Augmentation.stage is a ClassVar
        # but we override it on the instance here).
        self.stage = Stage(space) if isinstance(space, str) else space
        # Forbid raw-categorical noise — meaningless on integer class indices.
        if self.stage == Stage.RAW:
            apply_to_categ = False
        self.apply_to_categ = apply_to_categ
        self.apply_to_cont = apply_to_cont

    def forward(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        out = dict(data)
        x_categ = out.get('categorical')
        x_cont = out.get('continuous')

        if self.sigma == 0.0:
            return out

        if self.apply_to_categ and x_categ is not None:
            out['categorical'] = x_categ + self.sigma * torch.randn_like(x_categ)
        if self.apply_to_cont and x_cont is not None:
            out['continuous'] = x_cont + self.sigma * torch.randn_like(x_cont)

        return out
