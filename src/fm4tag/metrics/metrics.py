"""Representation-quality metrics for evaluating pretrained encoders.

These are *monitoring* metrics — they carry no gradients and are never used as
training objectives.  Import and call them inside ``torch.no_grad()`` blocks.

Currently implemented
---------------------
uniformity
    Measures how uniformly embeddings are spread on the unit hypersphere
    (Wang & Isola, NeurIPS 2020).  More negative = more uniform = better.
effective_rank
    exp(H) where H is the entropy of the normalised singular value spectrum
    (Roy & Vetterli 2007).  Higher = less collapsed.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """Effective rank of the embedding matrix (Roy & Vetterli 2007).

    Args:
        z: ``(N, D)`` embedding matrix.

    Returns:
        Effective rank scalar (in [1, D]).
    """
    z_f = z.float()
    z_f = z_f - z_f.mean(0, keepdim=True)
    if not torch.isfinite(z_f).all():
        return float('nan')
    s = torch.linalg.svdvals(z_f)
    s = s[s > 0]
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()).item())


@torch.no_grad()
def uniformity(z: torch.Tensor, t: float = 2.0, max_samples: int = 4096) -> torch.Tensor:
    """Uniformity metric (Wang & Isola, NeurIPS 2020).

    Measures how uniformly embeddings are spread on the unit hypersphere.
    More negative = more uniform = better. A value near 0 indicates collapse.

    ``torch.pdist`` is O(N²) in memory; if ``N > max_samples`` a random subset
    is drawn so the metric remains tractable on large constituent batches.

    Args:
        z:           ``(N, D)`` embeddings (need not be pre-normalised).
        t:           Gaussian kernel bandwidth (default 2.0, as in the paper).
        max_samples: Maximum number of embeddings to use; subsamples if N is
                     larger.

    Returns:
        Scalar uniformity value.
    """
    z = F.normalize(z, dim=-1)
    if z.size(0) > max_samples:
        idx = torch.randperm(z.size(0), device=z.device)[:max_samples]
        z = z[idx]
    sq_dists = torch.pdist(z.float(), p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log()
