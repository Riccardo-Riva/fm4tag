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
            F.cross_entropy(logits[local_start:local_end], targets[local_start:local_end])
            + F.cross_entropy(logits.t()[local_start:local_end], targets[local_start:local_end])
        )
        return loss


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
