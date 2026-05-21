"""Metric registry for online representation-quality monitoring."""

from __future__ import annotations

from typing import Callable

import torch

_REGISTRY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}


def register_metric(name: str) -> Callable:
    """Decorator that registers a metric function under ``name``.

    Usage::

        @register_metric('my_metric')
        def my_metric(z: torch.Tensor) -> torch.Tensor:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator


def compute_metric(name: str, z: torch.Tensor) -> torch.Tensor:
    """Call a registered metric by name.

    Raises:
        ValueError: if ``name`` is not in the registry.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown metric {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](z)
