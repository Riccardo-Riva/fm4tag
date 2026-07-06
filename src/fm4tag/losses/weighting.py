"""Validation and normalisation of loss-component weights."""

from __future__ import annotations


def normalize_loss_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalise loss-component weights so they sum to 1.

    Only the relative balance between components matters, so
    ``{'a': 2, 'b': 1}`` and ``{'a': 1, 'b': 0.5}`` yield the same result.
    Zero-weight components stay zero — the training modules use that to skip
    computing (and logging) the corresponding loss term entirely.

    Args:
        weights: Mapping component name → non-negative weight.

    Returns:
        New dict with the same keys and weights scaled to unit sum.

    Raises:
        ValueError: If any weight is negative, or all weights are zero.
    """
    negative = {k: w for k, w in weights.items() if w < 0}
    if negative:
        raise ValueError(f'Loss weights must be non-negative, got {negative}.')

    total = sum(weights.values())
    if total == 0:
        raise ValueError(
            f'At least one loss weight must be > 0, got all zeros: {weights}.'
        )

    return {k: w / total for k, w in weights.items()}
