"""Track / constituent dropout — randomly mark valid constituents as invalid.

Operates at the :attr:`Stage.PRE_FLATTEN` step: each currently-valid
constituent is independently set to invalid with probability ``drop_prob``.
The subsequent flatten-by-valid step in the encoder pipeline then removes
the dropped constituents naturally — no further changes are needed
downstream.

This augmentation has no effect on global-object batches (no ``valid``
mask present); it returns the input unchanged in that case.
"""

from __future__ import annotations

import torch

from .base import Augmentation, Stage, register


@register('track_dropout')
@register('constituent_dropout')
class TrackDropout(Augmentation):
    """Drop a random subset of valid constituents per jet.

    Args:
        drop_prob:   Probability that each currently-valid constituent is
                     marked invalid.
        min_valid:   Minimum number of valid constituents that must remain
                     per jet.  If dropping would take a jet below this
                     count, fewer constituents are dropped from that jet.
                     ``0`` disables this safeguard.
    """

    stage = Stage.PRE_FLATTEN

    def __init__(self, drop_prob: float = 0.15, min_valid: int = 1) -> None:
        super().__init__()
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f'drop_prob must be in [0, 1], got {drop_prob}')
        if min_valid < 0:
            raise ValueError(f'min_valid must be >= 0, got {min_valid}')
        self.drop_prob = drop_prob
        self.min_valid = min_valid

    def forward(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        valid = data.get('valid')
        if valid is None or self.drop_prob == 0.0:
            return dict(data)

        # Draw a fresh Bernoulli mask of "keep" decisions, restricted to
        # currently-valid slots.  Invalid slots stay invalid.
        keep = torch.bernoulli(
            torch.full_like(valid, 1.0 - self.drop_prob, dtype=torch.float32)
        ).bool()
        new_valid = valid & keep

        if self.min_valid > 0:
            # For any jet that dipped below the floor, restore enough of the
            # originally-valid (now-dropped) constituents to reach min_valid.
            counts = new_valid.sum(dim=1)  # (B,)
            deficient = counts < self.min_valid
            if deficient.any():
                # For each deficient row, randomly re-enable previously-valid
                # constituents that we had dropped, in priority order.
                deficient_rows = torch.nonzero(deficient, as_tuple=False).squeeze(1)
                for b in deficient_rows.tolist():
                    needed = self.min_valid - int(counts[b].item())
                    # Candidates: originally valid AND currently dropped.
                    candidates = torch.nonzero(
                        valid[b] & ~new_valid[b], as_tuple=False
                    ).squeeze(1)
                    if candidates.numel() == 0:
                        continue
                    n_restore = min(needed, candidates.numel())
                    pick = candidates[
                        torch.randperm(candidates.numel(), device=valid.device)[:n_restore]
                    ]
                    new_valid[b, pick] = True

        out = dict(data)
        out['valid'] = new_valid
        return out
