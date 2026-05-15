"""Distributed-data-parallel utilities for cross-rank embedding gathering."""

from __future__ import annotations

from typing import Callable

import torch


def gather_embeddings_sized(
    z_local: torch.Tensor | None,
    world_size: int,
    all_gather_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> torch.Tensor | None:
    """Gather same-shaped embeddings across DDP ranks, returning n_min rows each.

    All ranks must call this function in the same order so that the two
    all_gather collectives stay synchronised.  If any rank has zero local
    samples (n_min == 0), every rank returns None and the second collective
    is skipped, preventing a hang on the rank with no data.

    Args:
        z_local:        ``(N_local, D)`` local embeddings, or ``None`` if this
                        rank contributed no samples for this object.
        world_size:     Total number of ranks in the process group.
        all_gather_fn:  Callable matching Lightning's ``self.all_gather``
                        signature: takes a ``(*shape,)`` tensor and returns a
                        ``(world_size, *shape)`` tensor with each rank's copy
                        stacked along a new leading dimension.
        device:         Device for auxiliary tensors (must match the process
                        group's device, e.g. ``module.device``).

    Returns:
        ``(world_size * n_min, D)`` CPU tensor, or ``None`` if any rank had
        zero samples.
    """
    if world_size <= 1:
        return z_local

    n_local = torch.tensor(
        [z_local.size(0) if z_local is not None else 0],
        device=device,
    )
    all_n = all_gather_fn(n_local).view(-1)  # (world_size,)
    n_min = int(all_n.min().item())
    if n_min == 0:
        return None

    z = all_gather_fn(z_local[:n_min].to(device)).flatten(0, 1).cpu()
    return z
