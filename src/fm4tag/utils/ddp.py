"""Distributed-data-parallel utilities for cross-rank embedding gathering."""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed
import torch.distributed.nn.functional as dist_nn_functional


def all_gather_with_grad(z: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Differentiably gather ``z`` from all DDP ranks.

    Returns ``(z_all, local_start, local_end)`` where ``z_all`` is the
    concatenation of every rank's tensor in rank order, and
    ``z_all[local_start:local_end]`` is this rank's shard.

    Unlike a plain ``torch.distributed.all_gather`` (which detaches every
    rank's contribution), this uses ``torch.distributed.nn.functional.
    all_gather``, whose backward **reduce-scatters** the gradient: each rank
    receives the summed gradient for its own shard from *all* anchors on
    *all* ranks.  Combined with DDP's gradient averaging, this reproduces the
    exact full-(global-)batch gradient, even when the loss is computed only
    over the local-anchor slice.  See ``tests/ddp/test_supcon_gradient.py``.

    Ranks may contribute **different numbers of rows** (e.g. flattened
    variable-length constituents): each rank pads its tensor to the largest
    per-rank size, gathers, and strips the padding again, so ``z_all`` always
    holds exactly the union of every rank's real rows.  A contrastive loss
    computed on ``z_all`` therefore sees the full global batch as its pool
    regardless of world size or sharding.

    Falls back to ``(z, 0, N)`` on a single GPU or when the process group is
    not initialised.
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() == 1
    ):
        return z, 0, z.size(0)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    local_n = torch.tensor([z.size(0)], device=z.device)
    all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
    torch.distributed.all_gather(all_n, local_n)
    sizes = [int(n.item()) for n in all_n]

    n_max = max(sizes)
    if n_max == 0:
        return z, 0, z.size(0)

    if min(sizes) == n_max:
        # Equal sizes: gather directly.
        # Backward reduce-scatters grads to each rank's shard (see above).
        gathered = dist_nn_functional.all_gather(z.contiguous())
        z_all = torch.cat(gathered, dim=0)
        local_start = rank * z.size(0)
    else:
        # Unequal sizes: pad to n_max so the collective is well-shaped, then
        # strip each rank's padding after the gather.  Padded rows are sliced
        # away before any use, so they receive zero gradient.
        pad = z.new_zeros((n_max - z.size(0), *z.shape[1:]))
        z_padded = torch.cat([z, pad], dim=0).contiguous()
        gathered = dist_nn_functional.all_gather(z_padded)
        z_all = torch.cat([g[:n] for g, n in zip(gathered, sizes)], dim=0)
        local_start = sum(sizes[:rank])

    return z_all, local_start, local_start + z.size(0)


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
