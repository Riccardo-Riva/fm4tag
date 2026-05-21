"""Uniformity metric (Wang & Isola, NeurIPS 2020)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import register_metric


@register_metric('uniformity')
@torch.no_grad()
def uniformity(
    z: torch.Tensor,
    t: float = 2.0,
    max_samples: int = 4096,
) -> torch.Tensor:
    """Log-average Gaussian kernel on the unit hypersphere.

    More negative = more uniform = better. Near 0 indicates collapse.

    ``torch.pdist`` is O(N²) in memory; if ``N > max_samples`` a random subset
    is drawn so the metric remains tractable on large constituent batches.

    Args:
        z:           ``(N, D)`` embeddings (need not be pre-normalised).
        t:           Gaussian kernel bandwidth (default 2.0, as in the paper).
        max_samples: Maximum number of embeddings to use.
    """
    z = F.normalize(z, dim=-1)
    if z.size(0) > max_samples:
        idx = torch.randperm(z.size(0), device=z.device)[:max_samples]
        z = z[idx]
    sq_dists = torch.pdist(z.float(), p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log()
