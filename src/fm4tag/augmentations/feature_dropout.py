"""SCARF-style feature corruption — resample a random subset of features
from the batch marginal.

For each row, pick a random subset of features (Bernoulli per feature) and
overwrite their values with values drawn from another randomly chosen row
of the batch.  Unlike :class:`CutMix`, this uses an independent random row
**per feature** rather than a single shuffled batch — which provides a more
diverse corruption since each feature can come from a different example.

References
----------
Bahri et al., SCARF (2021), arXiv:2106.15147.
"""

from __future__ import annotations

import torch

from .base import Augmentation, Stage


class FeatureDropout(Augmentation):
    """Random per-feature resampling from the batch marginal.

    Args:
        corrupt_frac: Probability that an individual feature is replaced.
            Following the SCARF paper, ``0.6`` is a common default; lower
            values give milder corruption.
    """

    stage = Stage.RAW

    def __init__(self, corrupt_frac: float = 0.6) -> None:
        super().__init__()
        if not 0.0 <= corrupt_frac <= 1.0:
            raise ValueError(
                f'corrupt_frac must be in [0, 1], got {corrupt_frac}'
            )
        self.corrupt_frac = corrupt_frac

    def forward(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        x_categ = data.get('categorical')
        x_cont = data.get('continuous')

        ref = x_categ if x_categ is not None else x_cont
        if ref is None:
            return dict(data)
        N = ref.size(0)
        device = ref.device
        if N <= 1:
            return dict(data)

        out = dict(data)
        for key, x in (('categorical', x_categ), ('continuous', x_cont)):
            if x is None:
                continue
            # Per-feature independent random row index, same shape as x.
            # gather(0, rand_rows)[i, j] == x[rand_rows[i, j], j].
            rand_rows = torch.randint(0, N, x.shape, device=device)
            corrupt_mask = (
                torch.rand(x.shape, device=device) < self.corrupt_frac
            )
            replacement = x.gather(0, rand_rows)
            out[key] = torch.where(corrupt_mask, replacement, x)

        return out
