"""Effective rank metric (Roy & Vetterli, 2007)."""

from __future__ import annotations

import torch

from .registry import register_metric


@register_metric('effective_rank')
@torch.no_grad()
def effective_rank(z: torch.Tensor) -> torch.Tensor:
    """exp(H) where H is the entropy of the normalised singular value spectrum.

    Higher = less collapsed. Range: [1, D].

    Args:
        z: ``(N, D)`` embedding matrix.
    """
    z_f = z.float()
    z_f = z_f - z_f.mean(0, keepdim=True)
    if not torch.isfinite(z_f).all():
        return torch.tensor(float('nan'))
    s = torch.linalg.svdvals(z_f)
    s = s[s > 0]
    p = s / s.sum()
    return torch.exp(-(p * p.log()).sum())
