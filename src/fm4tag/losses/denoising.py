"""Denoising (reconstruction) losses for the pretraining objective.

Two independent components, matching the ``denoising_cat`` / ``denoising_con``
entries of the pretraining module's ``loss_weights``:

* **Categorical features** – cross-entropy between predicted logits and the
  original (uncorrupted) integer class indices.  All categorical features are
  reconstructed (including index 0).
* **Continuous features** – MSE between the concatenated scalar predictions
  and the original continuous values.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def denoising_cat_loss(
    cat_outs: list[torch.Tensor],
    x_categ: torch.Tensor,
) -> torch.Tensor:
    """Categorical reconstruction loss (sum of per-feature cross-entropies).

    Args:
        cat_outs: List of ``F_cat`` tensors each of shape ``(N, n_classes_j)``,
                  one reconstruction logit tensor per categorical feature,
                  as returned by :class:`sep_MLP`.
        x_categ:  ``(N, F_cat)`` long tensor of original (uncorrupted,
                  pre-offset) categorical indices.

    Returns:
        Scalar loss tensor (zero if there are no categorical features).
    """
    loss = x_categ.new_zeros((), dtype=torch.float32)
    for j in range(x_categ.shape[-1]):
        loss = loss + F.cross_entropy(cat_outs[j], x_categ[:, j])
    return loss


def denoising_con_loss(
    con_outs: list[torch.Tensor],
    x_cont: torch.Tensor,
) -> torch.Tensor:
    """Continuous reconstruction loss (MSE over all continuous features).

    Args:
        con_outs: List of ``F_con`` tensors each of shape ``(N, 1)``,
                  one scalar prediction per continuous feature,
                  as returned by :class:`sep_MLP`.
        x_cont:   ``(N, F_con)`` float tensor of original continuous values.

    Returns:
        Scalar loss tensor (zero if there are no continuous features).
    """
    if not con_outs:
        return x_cont.new_zeros(())
    con_pred = torch.cat(con_outs, dim=1)  # (N, F_con)
    return F.mse_loss(con_pred, x_cont)


class DenoisingLoss(nn.Module):
    """Legacy combined wrapper — used only by the legacy
    ``ContrastiveDenoisingModule`` loss adapters; goes away together with it.
    """

    def forward(
        self,
        cat_outs: list[torch.Tensor],
        x_categ: torch.Tensor,
        con_outs: list[torch.Tensor],
        x_cont: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            denoising_cat_loss(cat_outs, x_categ),
            denoising_con_loss(con_outs, x_cont),
        )
