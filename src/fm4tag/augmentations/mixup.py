"""Embedding-space mixup — convex combination with a shuffled copy of the
batch in token-embedding space.

Replaces the functional :func:`mixup_data` from
``augmentations.augmentations`` with a stateful :class:`Augmentation`.

Operates at the :attr:`Stage.EMBEDDING` step on tensors of shape
``(N, F_cat, dim)`` and ``(N, F_con, dim)`` (the output of ``embed_data``,
before the transformer).
"""

from __future__ import annotations

import torch

from .base import Augmentation, Stage


class Mixup(Augmentation):
    """Linear interpolation of embedded tokens with a shuffled batch copy.

    For each sample :math:`i`::

        x'_i = (1 - lam) * x_i + lam * x_{perm[i]}

    where ``perm`` is a random permutation of the batch.  The same
    permutation is used for categorical and continuous embeddings so the
    two streams stay aligned per-sample.

    Args:
        lam: Noise level — mixing weight on the permuted sample (i.e.
            ``lam=0`` is identity, ``lam=1`` fully replaces with the permuted
            sample).  This is a fixed scalar; if you want a per-batch
            Beta-distributed lam, set ``random_lam=True`` and pass ``alpha``
            instead.
        alpha:      Beta-distribution parameter when ``random_lam=True``.
                    Ignored when ``random_lam=False``.
        random_lam: Sample ``lam ~ Beta(alpha, alpha)`` per forward pass.
    """

    stage = Stage.EMBEDDING

    def __init__(
        self,
        lam: float = 0.2,
        alpha: float = 0.4,
        random_lam: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f'lam must be in [0, 1], got {lam}')
        if alpha <= 0.0:
            raise ValueError(f'alpha must be > 0, got {alpha}')
        self.lam = lam
        self.alpha = alpha
        self.random_lam = random_lam

    def _sample_lam(self, device: torch.device) -> float:
        if not self.random_lam:
            return self.lam
        # Beta(alpha, alpha) symmetric — only need one sample, scalar.
        dist = torch.distributions.Beta(self.alpha, self.alpha)
        return float(dist.sample().to(device).item())

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

        lam = self._sample_lam(device)
        perm = torch.randperm(N, device=device)

        out = dict(data)
        for key, x in (('categorical', x_categ), ('continuous', x_cont)):
            if x is None:
                continue
            out[key] = (1.0 - lam) * x + lam * x[perm]

        return out
