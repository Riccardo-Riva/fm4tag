import torch
import torch.nn.functional as F
from torch import nn


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (NT-Xent) contrastive loss.

    Given two sets of projected embeddings ``z1`` and ``z2`` from two
    augmented views of the same data, this loss encourages representations
    of the same sample to be similar and those of different samples to be
    dissimilar.

    Reference: Chen et al., "A Simple Framework for Contrastive Learning
    of Visual Representations" (SimCLR), ICML 2020.
    """

    def __init__(self, temperature: float = 0.7) -> None:
        super().__init__()
        self.temperature = temperature

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

        logits = (z1 @ z2.t()) / self.temperature  # (N, N)
        targets = torch.arange(logits.size(0), device=logits.device)

        loss = 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
        )
        return loss


class DenoisingLoss(nn.Module):
    """Multi-task denoising reconstruction loss.

    * **Categorical features** – cross-entropy between predicted logits and
      the original (uncorrupted) integer class indices.
    * **Continuous features** – MSE between the concatenated scalar predictions
      and the original continuous values.

    The first categorical feature (index 0 along ``F_cat``) is treated as a
    summary / CLS token and is intentionally **not** reconstructed, matching
    the convention used in the classifier head and the original SAINT code.
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
        # Categorical loss — skip index 0 (summary token, not reconstructed).
        loss_cat = x_categ.new_zeros(1, dtype=torch.float).squeeze()
        for j in range(1, x_categ.shape[-1]):
            loss_cat = loss_cat + F.cross_entropy(cat_outs[j], x_categ[:, j])

        # Continuous loss.
        loss_con = x_cont.new_zeros(1, dtype=torch.float).squeeze()
        if con_outs:
            con_pred = torch.cat(con_outs, dim=1)  # (N, F_con)
            loss_con = F.mse_loss(con_pred, x_cont)

        return loss_cat, loss_con
