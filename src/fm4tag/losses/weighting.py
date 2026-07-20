"""Validation of loss-component weights."""

from __future__ import annotations


def validate_loss_weights(weights: dict[str, float]) -> dict[str, float]:
    """Validate loss-component weights, applied to the loss terms as-is.

    Zero-weight components stay zero — the training modules use that to skip
    computing (and logging) the corresponding loss term entirely.

    Args:
        weights: Mapping component name → non-negative weight.

    Returns:
        The same dict, unchanged.

    Raises:
        ValueError: If any weight is negative, or all weights are zero.
    """
    negative = {k: w for k, w in weights.items() if w < 0}
    if negative:
        raise ValueError(f'Loss weights must be non-negative, got {negative}.')

    if sum(weights.values()) == 0:
        raise ValueError(
            f'At least one loss weight must be > 0, got all zeros: {weights}.'
        )

    return weights
