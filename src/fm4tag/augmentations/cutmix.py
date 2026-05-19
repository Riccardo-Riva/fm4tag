"""CutMix augmentation — element-wise value swap between samples.

Replaces the existing functional :func:`add_noise` from
``augmentations.augmentations`` with a stateful :class:`Augmentation`.

For each feature position, a Bernoulli mask decides whether to replace the
value with the corresponding feature of a randomly chosen other row in the
batch.  Operates on flat ``(N, F)`` tensors at the :attr:`Stage.RAW` step.

This is the standard SAINT / TabTransformer pretraining corruption.  With
``lam`` close to 1 the corruption is mild; with ``lam`` close to 0 the row
is almost entirely overwritten from neighbours.

References
----------
Somepalli et al., SAINT (2021), arXiv:2106.01342, section 3.2.
"""

from __future__ import annotations

import torch

from .base import Augmentation, Stage


class CutMix(Augmentation):
    """Per-feature random swap with a shuffled copy of the batch.

    Args:
        lam: Probability that an individual feature is **kept** (i.e.
            ``Bernoulli(lam)`` mask).  Higher = less corruption.  Same as
            the ``lam`` in the legacy :func:`add_noise`.
        per_feature_mask: If ``True`` (default), draw an independent
            Bernoulli for every (row, feature) cell.  If ``False``, draw
            one Bernoulli per row and apply it to all features in that row
            (whole-row swap).
    """

    stage = Stage.RAW

    def __init__(self, lam: float = 0.7, per_feature_mask: bool = True) -> None:
        super().__init__()
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f'cutmix lam must be in [0, 1], got {lam}')
        self.lam = lam
        self.per_feature_mask = per_feature_mask

    @staticmethod
    def _swap(
        x: torch.Tensor, perm: torch.Tensor, keep_mask: torch.Tensor
    ) -> torch.Tensor:
        # keep_mask broadcasts to x's shape; True = keep original.
        return torch.where(keep_mask, x, x[perm])

    def forward(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        x_categ = data.get('categorical')
        x_cont = data.get('continuous')

        # Resolve N from whichever tensor is present.
        ref = x_categ if x_categ is not None else x_cont
        if ref is None:
            return dict(data)
        N = ref.size(0)
        device = ref.device
        if N <= 1:
            # No other rows to swap with — return unchanged.
            return dict(data)

        perm = torch.randperm(N, device=device)

        out = dict(data)
        for key, x in (('categorical', x_categ), ('continuous', x_cont)):
            if x is None:
                continue
            if self.per_feature_mask:
                keep = torch.bernoulli(
                    torch.full_like(x, self.lam, dtype=torch.float32)
                ).bool()
            else:
                # Single mask per row, broadcast over features.
                row_keep = torch.bernoulli(
                    torch.full((N,), self.lam, dtype=torch.float32, device=device)
                ).bool()
                keep = row_keep.view(N, *([1] * (x.dim() - 1))).expand_as(x)
            out[key] = self._swap(x, perm, keep)

        return out
