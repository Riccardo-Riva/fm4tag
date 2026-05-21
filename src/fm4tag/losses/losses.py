from __future__ import annotations

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (NT-Xent) contrastive loss.

    Given two sets of projected embeddings ``z1`` and ``z2`` from two
    augmented views of the same data, this loss encourages representations
    of the same sample to be similar and those of different samples to be
    dissimilar.

    In DDP training, embeddings are gathered from all ranks before computing
    the similarity matrix, so each rank sees the full batch as negatives.
    Gradients only flow through the local rank's shard.  When constituent-level
    embeddings are used (variable ``N_valid`` across ranks), gathering is
    skipped and each rank computes a local InfoNCE instead.

    Reference: Chen et al., "A Simple Framework for Contrastive Learning
    of Visual Representations" (SimCLR), ICML 2020.
    """

    def __init__(self, temperature: float = 0.7) -> None:
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _all_gather_with_grad(z: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Gather ``z`` from all ranks, preserving gradients for the local shard.

        Returns:
            ``(z_all, local_start, local_end)`` where ``z_all[local_start:local_end]``
            corresponds to the local rank's entries and retains a gradient path.
            Falls back to ``(z, 0, N)`` on single GPU or when ranks have
            different leading-dimension sizes (e.g. variable-length constituents).
        """
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return z, 0, z.size(0)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # Verify all ranks have the same leading dimension.
        local_n = torch.tensor([z.size(0)], device=z.device)
        all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
        torch.distributed.all_gather(all_n, local_n)
        if not all(n.item() == local_n.item() for n in all_n):
            # Variable sizes across ranks (e.g. constituent-level) — local fallback.
            return z, 0, z.size(0)

        # All-gather; replace the local shard with the original to preserve gradients.
        gathered = [torch.zeros_like(z) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, z.contiguous())
        gathered[rank] = z
        z_all = torch.cat(gathered, dim=0)

        local_start = rank * z.size(0)
        local_end = local_start + z.size(0)
        return z_all, local_start, local_end

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute the symmetric InfoNCE loss.

        Args:
            z1: ``(N, D)`` projected embeddings from view 1.
            z2: ``(N, D)`` projected embeddings from view 2.

        Returns:
            Scalar loss tensor.
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z1_all, local_start, local_end = self._all_gather_with_grad(z1)
        z2_all, _, _ = self._all_gather_with_grad(z2)

        N = z1_all.size(0)
        logits = (z1_all @ z2_all.t()) / self.temperature  # (N, N)
        targets = torch.arange(N, device=logits.device)

        # Loss computed only over the local shard so gradients flow to local z1/z2.
        loss = 0.5 * (
            F.cross_entropy(
                logits[local_start:local_end], targets[local_start:local_end]
            )
            + F.cross_entropy(
                logits.t()[local_start:local_end], targets[local_start:local_end]
            )
        )
        return loss


class MultiViewSupConLoss(nn.Module):
    """Multi-view contrastive loss generalising SupCon to arbitrary numbers of views.

    For a batch of N samples with V views each, all V views of sample *i* are
    positive pairs; every view of every other sample is a negative.

    In DDP training, embeddings are gathered from all ranks (when all ranks
    have the same leading dimension) so the full cross-device batch acts as
    negatives.  Gradients flow only through the local shard.  When constituent
    embeddings have variable N across ranks, gathering is skipped and each rank
    computes a local loss.

    Reference: Khosla et al., "Supervised Contrastive Learning",
    NeurIPS 2020. https://arxiv.org/abs/2004.11362

    Args:
        temperature:          Logit scale.
        loss_type:            ``'out'`` — L_out from the paper: per-positive
                              log-probs are summed outside the log, then averaged.
                              ``'in'``  — L_in from the paper: positive similarities
                              are summed inside a single log before dividing by Z.
        include_pos_in_denom: If ``True`` (default, matches the paper), the
                              softmax denominator Z contains all other
                              view-instances (positives + negatives).  If
                              ``False``, positives are excluded from Z so the
                              loss only pushes negatives away.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        loss_type: str = 'out',
        include_pos_in_denom: bool = True,
    ) -> None:
        super().__init__()
        if loss_type not in ('out', 'in'):
            raise ValueError(f"loss_type must be 'out' or 'in', got {loss_type!r}")
        self.temperature = temperature
        self.loss_type = loss_type
        self.include_pos_in_denom = include_pos_in_denom

    @staticmethod
    def _all_gather_with_grad(z: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Gather ``z`` from all ranks, preserving gradients for the local shard.

        Returns ``(z_all, local_start, local_end)``.  Falls back to
        ``(z, 0, N)`` on a single GPU or when ranks have different sizes.
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
        if not all(n.item() == local_n.item() for n in all_n):
            return z, 0, z.size(0)

        gathered = [torch.zeros_like(z) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, z.contiguous())
        gathered[rank] = z
        z_all = torch.cat(gathered, dim=0)
        local_start = rank * z.size(0)
        local_end = local_start + z.size(0)
        return z_all, local_start, local_end

    def forward(self, z_list: list[torch.Tensor]) -> torch.Tensor:
        """Compute multi-view SupCon loss.

        Args:
            z_list: V tensors each of shape ``(N, D)`` — one projected embedding
                    tensor per view, in the same sample order.

        Returns:
            Scalar loss.
        """
        V = len(z_list)
        if V < 2:
            raise ValueError("MultiViewSupConLoss requires at least 2 views.")

        N = z_list[0].size(0)
        # Stack to (N, V, D) then flatten to (N*V, D).
        # Ordering: sample0_view0, sample0_view1, …, sample1_view0, …
        z = torch.stack(z_list, dim=1).reshape(N * V, -1)
        z = F.normalize(z, dim=-1)

        z_all, local_start, local_end = self._all_gather_with_grad(z)
        total = z_all.size(0)   # world_size * N * V
        total_N = total // V

        # Label for index i: which sample it belongs to.
        # After DDP gather, rank r contributes items [r*N*V … (r+1)*N*V),
        # interleaved as sample0_view0, sample0_view1, … so label[i] = i // V.
        labels = torch.arange(total_N, device=z_all.device).repeat_interleave(V)

        sim = (z_all @ z_all.T) / self.temperature   # (total, total)

        mask_self = torch.eye(total, dtype=torch.bool, device=z_all.device)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self

        # Build denominator: always exclude self; optionally exclude positives.
        denom_mask = mask_self if self.include_pos_in_denom else (mask_self | mask_pos)
        log_Z = torch.logsumexp(sim.masked_fill(denom_mask, float('-inf')), dim=-1)

        if self.loss_type == 'out':
            # L_out = mean_i [ mean_{p in P(i)} (log_Z[i] - sim[i,p]) ]
            n_pos = mask_pos.float().sum(1)
            loss_per_anchor = (
                log_Z - (sim * mask_pos.float()).sum(1) / n_pos.clamp(min=1)
            )
        else:
            # L_in = mean_i [ log_Z[i] - log(sum_{p in P(i)} exp(sim[i,p])) ]
            log_sum_pos = torch.logsumexp(
                sim.masked_fill(~mask_pos, float('-inf')), dim=-1
            )
            loss_per_anchor = log_Z - log_sum_pos

        return loss_per_anchor[local_start:local_end].mean()


class DenoisingLoss(nn.Module):
    """Multi-task denoising reconstruction loss.

    * **Categorical features** – cross-entropy between predicted logits and
      the original (uncorrupted) integer class indices.  All categorical
      features are reconstructed (including index 0).
    * **Continuous features** – MSE between the concatenated scalar predictions
      and the original continuous values.
    """

    def forward(
        self,
        cat_outs: list[torch.Tensor],
        x_categ: torch.Tensor,
        con_outs: list[torch.Tensor],
        x_cont: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute categorical and continuous denoising losses.

        Args:
            cat_outs: List of ``F_cat`` tensors each of shape ``(N, n_classes_j)``,
                      one reconstruction logit tensor per categorical feature,
                      as returned by :class:`sep_MLP`.
            x_categ:  ``(N, F_cat)`` long tensor of original (uncorrupted,
                      pre-offset) categorical indices.
            con_outs: List of ``F_con`` tensors each of shape ``(N, 1)``,
                      one scalar prediction per continuous feature,
                      as returned by :class:`sep_MLP`.
            x_cont:   ``(N, F_con)`` float tensor of original continuous values.

        Returns:
            ``(loss_cat, loss_con)`` – two scalar tensors.
        """
        # Categorical loss — reconstruct all features including index 0.
        loss_cat = x_categ.new_zeros(())
        for j in range(x_categ.shape[-1]):
            loss_cat = loss_cat + F.cross_entropy(cat_outs[j], x_categ[:, j])

        # Continuous loss.
        loss_con = x_cont.new_zeros(())
        if con_outs:
            con_pred = torch.cat(con_outs, dim=1)  # (N, F_con)
            loss_con = F.mse_loss(con_pred, x_cont)

        return loss_cat, loss_con
