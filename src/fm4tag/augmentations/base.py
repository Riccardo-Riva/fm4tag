"""Augmentation base class and dispatch infrastructure.

Augmentations are :class:`~torch.nn.Module` subclasses that transform a data
dict in one of three positions in the encoding pipeline:

* :attr:`Stage.PRE_FLATTEN` — operates on the un-flattened constituents
  ``(B, C, F)`` together with the ``valid`` mask ``(B, C)``.  Use this when an
  augmentation needs the per-jet constituent structure (e.g. dropping entire
  constituents by flipping the mask).

* :attr:`Stage.RAW` — operates on flat raw features after the
  flatten-by-valid step: ``categorical (N, F_cat)`` and ``continuous
  (N, F_con)``.  Use this for value-level corruptions (cutmix, SCARF,
  additive noise on raw inputs).

* :attr:`Stage.EMBEDDING` — operates on embedded tokens
  ``(N, F, dim)`` produced by ``embed_data``.  Use this when the augmentation
  is naturally defined in embedding space (mixup, noise on embeddings).

Each :class:`Augmentation` declares its stage via the class attribute
:attr:`stage`.  The composer :class:`Compose` groups a list of augmentations
by stage so the pretraining module can apply them at the right point.

For the **global object** (continuous-only, no constituents) the
:attr:`Stage.PRE_FLATTEN` step is skipped — the global path has no valid
mask.  Augmentations targeting raw or embedding stages still apply.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

import torch
from torch import nn


class Stage(str, Enum):
    """Position in the encoding pipeline at which an augmentation operates."""

    PRE_FLATTEN = 'pre_flatten'
    RAW = 'raw'
    EMBEDDING = 'embedding'


class Augmentation(nn.Module):
    """Base class for all augmentations.

    Subclasses must:

    1. Set the class attribute :attr:`stage` to one of :class:`Stage`'s values.
    2. Implement :meth:`forward(data)` that returns a transformed dict with
       the same keys.

    The ``data`` dict layout depends on the stage:

    +----------------+----------------------------------------------------------------+
    | Stage          | Dict shape                                                     |
    +================+================================================================+
    | PRE_FLATTEN    | ``categorical``: ``(B, C, F_cat)`` long or ``None``            |
    |                | ``continuous``:  ``(B, C, F_con)`` float or ``None``           |
    |                | ``valid``:       ``(B, C)`` bool or ``None``                   |
    +----------------+----------------------------------------------------------------+
    | RAW            | ``categorical``: ``(N, F_cat)`` long or ``None``               |
    |                | ``continuous``:  ``(N, F_con)`` float or ``None``              |
    +----------------+----------------------------------------------------------------+
    | EMBEDDING      | ``categorical``: ``(N, F_cat, dim)`` float or ``None``         |
    |                | ``continuous``:  ``(N, F_con, dim)`` float or ``None``         |
    +----------------+----------------------------------------------------------------+

    Augmentations must return a NEW dict (or new tensors); the caller may
    depend on the original being unmodified.
    """

    stage: ClassVar[Stage]

    def forward(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        raise NotImplementedError


class Compose(nn.Module):
    """Sequentially apply a list of augmentations grouped by stage.

    The composer keeps three ``nn.ModuleList`` slots and applies each in the
    order ``PRE_FLATTEN → RAW → EMBEDDING``.  Within a stage, augmentations
    are applied in the order they were listed in the config.

    Use the explicit per-stage dispatch methods (:meth:`apply_pre_flatten`,
    :meth:`apply_raw`, :meth:`apply_embedding`) from the pretraining module —
    the bare :meth:`forward` is not used because each stage is invoked at a
    different point in the encoding pipeline.
    """

    def __init__(self, augmentations: list[Augmentation]) -> None:
        super().__init__()
        self.pre_flatten = nn.ModuleList(
            [a for a in augmentations if a.stage == Stage.PRE_FLATTEN]
        )
        self.raw = nn.ModuleList(
            [a for a in augmentations if a.stage == Stage.RAW]
        )
        self.embedding = nn.ModuleList(
            [a for a in augmentations if a.stage == Stage.EMBEDDING]
        )

    @staticmethod
    def _apply(
        modules: nn.ModuleList,
        data: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        for m in modules:
            data = m(data)
        return data

    def apply_pre_flatten(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        return self._apply(self.pre_flatten, data)

    def apply_raw(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        return self._apply(self.raw, data)

    def apply_embedding(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        return self._apply(self.embedding, data)

    def __repr__(self) -> str:
        lines = ['Compose(']
        for stage_name, modules in [
            ('pre_flatten', self.pre_flatten),
            ('raw', self.raw),
            ('embedding', self.embedding),
        ]:
            if len(modules) == 0:
                continue
            lines.append(f'  {stage_name}:')
            for m in modules:
                lines.append(f'    {m}')
        lines.append(')')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Registry — maps config-string names to Augmentation classes
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Augmentation]] = {}


def register(name: str):
    """Class decorator that registers an :class:`Augmentation` subclass under
    ``name`` so it can be looked up from a YAML config."""

    def _decorator(cls: type[Augmentation]) -> type[Augmentation]:
        if name in _REGISTRY:
            raise ValueError(
                f'Augmentation {name!r} is already registered '
                f'(existing: {_REGISTRY[name].__name__})'
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get(name: str) -> type[Augmentation]:
    """Look up a registered :class:`Augmentation` class by its config name."""
    if name not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY))
        raise KeyError(
            f'Augmentation {name!r} not in registry. Available: {available}'
        )
    return _REGISTRY[name]


def build_from_config(
    aug_configs: list[dict] | None,
) -> Compose:
    """Build a :class:`Compose` from a list of config entries.

    Each entry must have a ``name`` key matching a registered augmentation;
    any other keys are passed as ``__init__`` kwargs.

    Example YAML::

        augmentations:
          - name: track_dropout
            drop_prob: 0.15
          - name: cutmix
            lam: 0.7
          - name: gaussian_noise
            space: embedding
            sigma: 0.05

    An empty or ``None`` list returns an empty :class:`Compose`.
    """
    if not aug_configs:
        return Compose([])
    built: list[Augmentation] = []
    for entry in aug_configs:
        entry = dict(entry)  # local copy — don't mutate the omegaconf node
        name = entry.pop('name')
        cls = get(name)
        built.append(cls(**entry))
    return Compose(built)
