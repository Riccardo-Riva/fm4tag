"""Composable fine-tuning loss built from a weighted list of loss terms.

``FinetuneLoss`` shares the exact list-of-terms design and scope-aware fan-out
of :class:`~fm4tag.modules.losses.pretrain_loss.PretrainLoss` — a term fires
only when every required parameter of its ``forward`` is present in the call.
The fine-tuning module calls it with ``logits``/``labels`` (always) and,
optionally, ``z_jet_list`` so that :class:`JetContrastiveFinetuneTerm` can add
a jet-level contrastive objective on top of cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .pretrain_loss import JetContrastiveTermAdapter, PretrainLoss


class FinetuneLoss(PretrainLoss):
    """Composable fine-tuning loss: ``total = Σ weight_i * term_i(...)``.

    Same constructor, validation, and scope-aware fan-out as
    :class:`~fm4tag.modules.losses.pretrain_loss.PretrainLoss`; only the set of
    terms differs (cross-entropy, optional jet contrastive).
    """


class CrossEntropyTerm(nn.Module):
    """Supervised classification term (cross-entropy on the class logits)."""

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Args:
            logits:        ``(B, n_classes)`` class logits.
            labels:        ``(B,)`` integer class labels.
            class_weights: Optional per-class weights for the loss.
        """
        loss_val = F.cross_entropy(logits, labels, weight=class_weights)
        return loss_val, {'loss_ce': loss_val}


class JetContrastiveFinetuneTerm(nn.Module):
    """Jet-level (POINT B) contrastive term used during fine-tuning.

    Thin wrapper around
    :class:`~fm4tag.modules.losses.pretrain_loss.JetContrastiveTermAdapter`, so
    the fine-tune jet contrastive is logged under the same
    ``jet_embedding/loss_contrastive`` key as in pretraining.

    Args:
        temperature:          Contrastive temperature.
        loss_type:            ``'out'`` (L_out) or ``'in'`` (L_in).
        include_pos_in_denom: SupCon denominator includes positives if ``True``.
    """

    def __init__(
        self,
        temperature: float,
        loss_type: str = 'out',
        include_pos_in_denom: bool = True,
    ) -> None:
        super().__init__()
        self.adapter = JetContrastiveTermAdapter(
            temperature=temperature,
            loss_type=loss_type,
            include_pos_in_denom=include_pos_in_denom,
        )

    def forward(
        self, z_jet_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Args:
            z_jet_list: One ``(B, jet_dim)`` jet embedding per view.
        """
        return self.adapter(z_jet_list)
