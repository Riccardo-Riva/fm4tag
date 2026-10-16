"""Multi-view contrastive loss (SupCon generalised to unlabelled views)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fm4tag.models.attention import RowGroups
from fm4tag.utils.ddp import all_gather_with_grad


class MultiViewSupConLoss(nn.Module):
    """Multi-view contrastive loss generalising SupCon to arbitrary numbers of views.

    For a batch of N samples with V views each, all V views of sample *i* are
    positive pairs; every view of every other sample is a negative.

    In DDP training, embeddings are gathered from all ranks so the full
    cross-device batch acts as the pool — also when ranks contribute different
    numbers of rows (flattened constituents), via padding inside
    :func:`all_gather_with_grad`.  The gather is differentiable, so although
    the loss is reduced over the local-anchor slice, gradients flow back to
    every rank's rows and — combined with DDP gradient averaging — reproduce
    the exact full-batch gradient for equal per-rank counts.  With unequal
    counts the effective objective is the mean of per-rank anchor means
    (each rank weighted equally, not each anchor); the residual world-size
    dependence is O(ΔN/N) and negligible at typical occupancies.

    With ``groups`` (:class:`~fm4tag.models.attention.RowGroups`, the jet
    chunks of the intersample attention) positives and negatives come from the
    anchor's own chunk only and nothing is gathered across ranks: a chunk
    lives on one rank, so the loss is world-size independent by construction.
    Similarities are then computed in fp32 and the loss is the mean over all
    anchors of all chunks.

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
        include_same_view_negatives: If ``True`` (default, matches the paper),
                              other samples from the anchor's *own* view count
                              as negatives in Z.  If ``False``, only instances
                              from *other* views contribute to Z (positives are
                              cross-view already, so they are unaffected).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        loss_type: str = 'out',
        include_pos_in_denom: bool = True,
        include_same_view_negatives: bool = True,
    ) -> None:
        super().__init__()
        if loss_type not in ('out', 'in'):
            raise ValueError(f"loss_type must be 'out' or 'in', got {loss_type!r}")
        self.temperature = temperature
        self.loss_type = loss_type
        self.include_pos_in_denom = include_pos_in_denom
        self.include_same_view_negatives = include_same_view_negatives

    def forward(
        self, z_list: list[torch.Tensor], groups: RowGroups | None = None
    ) -> torch.Tensor:
        """Compute multi-view SupCon loss.

        Args:
            z_list: V tensors each of shape ``(N, D)`` — one projected embedding
                    tensor per view, in the same sample order.
            groups: Optional chunks of the ``N`` axis; ``None`` = one pool
                    gathered across ranks.

        Returns:
            Scalar loss.
        """
        V = len(z_list)
        if V < 2:
            raise ValueError('MultiViewSupConLoss requires at least 2 views.')
        if groups is not None:
            return self._chunked(z_list, groups)

        N = z_list[0].size(0)
        # Stack to (N, V, D) then flatten to (N*V, D).
        # Ordering: sample0_view0, sample0_view1, …, sample1_view0, …
        z = torch.stack(z_list, dim=1).reshape(N * V, -1)
        z = F.normalize(z, dim=-1)

        z_all, local_start, local_end = all_gather_with_grad(z)
        total = z_all.size(0)  # world_size * N * V
        total_N = total // V

        # Label for index i: which sample it belongs to.
        # After DDP gather, rank r contributes items [r*N*V … (r+1)*N*V),
        # interleaved as sample0_view0, sample0_view1, … so label[i] = i // V.
        labels = torch.arange(total_N, device=z_all.device).repeat_interleave(V)

        sim = (z_all @ z_all.T) / self.temperature  # (total, total)

        mask_self = torch.eye(total, dtype=torch.bool, device=z_all.device)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self

        # Build denominator: always exclude self; optionally exclude positives.
        denom_mask = mask_self if self.include_pos_in_denom else (mask_self | mask_pos)

        if not self.include_same_view_negatives:
            # Item i belongs to view i % V: samples are interleaved view-fastest
            # (sample0_view0, sample0_view1, …), and each rank's gathered block
            # has length N*V — a multiple of V — so this holds across ranks too.
            view_ids = torch.arange(total, device=z_all.device) % V
            mask_same_view = view_ids.unsqueeze(0) == view_ids.unsqueeze(1)
            # Same view + same sample is only the anchor itself (already masked);
            # everything else same-view is a negative → drop it from Z.
            denom_mask = denom_mask | mask_same_view

        log_Z = torch.logsumexp(sim.masked_fill(denom_mask, float('-inf')), dim=-1)

        if self.loss_type == 'out':
            # L_out = mean_i [ mean_{p in P(i)} (log_Z[i] - sim[i,p]) ]
            n_pos = mask_pos.float().sum(1)
            loss_per_anchor = log_Z - (sim * mask_pos.float()).sum(1) / n_pos.clamp(
                min=1
            )
        else:
            # L_in = mean_i [ log_Z[i] - log(sum_{p in P(i)} exp(sim[i,p])) ]
            log_sum_pos = torch.logsumexp(
                sim.masked_fill(~mask_pos, float('-inf')), dim=-1
            )
            loss_per_anchor = log_Z - log_sum_pos

        loss_local = loss_per_anchor[local_start:local_end]
        if loss_local.numel() == 0:
            # Rank with no local anchors: mean() would be NaN; return a
            # graph-connected zero instead.
            return sim.sum() * 0.0
        return loss_local.mean()

    def _chunked(self, z_list: list[torch.Tensor], groups: RowGroups) -> torch.Tensor:
        z = torch.stack(z_list, dim=1)  # (N, V, D)
        if z.dtype in (torch.float16, torch.bfloat16):
            z = z.float()  # bf16 logits at T~0.1 are too coarse
        z = F.normalize(z, dim=-1)
        bounds = groups.cu_seqlens.tolist()
        losses = [
            self._chunk_loss(z[start:end])
            for start, end in zip(bounds[:-1], bounds[1:])
            if end > start
        ]
        if not losses:
            return z.sum() * 0.0
        losses = torch.cat(losses)
        return losses.mean() if losses.numel() else z.sum() * 0.0

    def _chunk_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Per-anchor losses of one chunk; ``z`` is ``(n, V, D)``, normalised."""
        n, V, _ = z.shape
        T = self.temperature
        flat = z.reshape(n * V, -1)  # sample-major, view-fastest
        sim = (flat @ flat.T).view(n, V, n, V) / T

        same_sample = torch.eye(n, dtype=torch.bool, device=z.device)[:, None, :, None]
        same_view = torch.eye(V, dtype=torch.bool, device=z.device)[None, :, None, :]
        excluded = same_sample & same_view  # self
        if not self.include_pos_in_denom:
            excluded = excluded | same_sample
        if not self.include_same_view_negatives:
            excluded = excluded | same_view
        log_Z = torch.logsumexp(
            sim.masked_fill(excluded, float('-inf')).view(n * V, n * V), dim=-1
        )

        # Positives are the anchor's other views: (n, V, V) dot products.
        pos = torch.einsum('svd,swd->svw', z, z) / T
        off_diag = ~torch.eye(V, dtype=torch.bool, device=z.device)
        if self.loss_type == 'out':
            pos_term = pos.masked_fill(~off_diag, 0.0).sum(-1) / (V - 1)
        else:
            pos_term = torch.logsumexp(pos.masked_fill(~off_diag, float('-inf')), -1)
        loss = log_Z - pos_term.reshape(n * V)
        # An anchor with an empty denominator (one-sample chunk, positives
        # excluded) carries no contrastive signal.
        return loss[torch.isfinite(log_Z)]
