"""Representation-quality metrics with a registration system.

Built-in metrics
----------------
* ``uniformity``      – Wang & Isola (NeurIPS 2020): spread on the unit hypersphere.
* ``effective_rank``  – Roy & Vetterli (2007): entropy of singular value spectrum.

Extending
---------
Use :func:`register_metric` as a decorator in any module that gets imported::

    from fm4tag.metrics import register_metric

    @register_metric('my_metric')
    @torch.no_grad()
    def my_metric(z: torch.Tensor) -> torch.Tensor:
        ...

Config usage
------------
::

    eval:
      metrics: [uniformity, effective_rank]
"""

from __future__ import annotations

# Import submodules first to trigger @register_metric decorations.
from . import effective_rank as _er  # noqa: F401
from . import uniformity as _u  # noqa: F401
from .effective_rank import effective_rank
from .registry import compute_metric, register_metric
from .uniformity import uniformity

__all__ = ['register_metric', 'compute_metric', 'uniformity', 'effective_rank']
