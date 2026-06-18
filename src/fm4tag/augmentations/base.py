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

For the **global object** (no constituents, so no valid mask) the
:attr:`Stage.PRE_FLATTEN` step is skipped.  Its categorical + continuous
features are otherwise handled like constituents: RAW augmentations run on the
raw values and EMBEDDING augmentations run on the embedded tokens *before* the
transformer.

Compose instances are built from config via Hydra's ``instantiate``, which
handles the nested ``_target_`` list recursively::

    views:
      - _target_: fm4tag.augmentations.Compose
        augmentations:
          - _target_: fm4tag.augmentations.CutMix
            lam: 0.7
          - _target_: fm4tag.augmentations.GaussianNoise
            space: embedding
            sigma: 0.05
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
        self.raw = nn.ModuleList([a for a in augmentations if a.stage == Stage.RAW])
        self.embedding = nn.ModuleList(
            [a for a in augmentations if a.stage == Stage.EMBEDDING]
        )

    @staticmethod
    def _run_stage(
        modules: nn.ModuleList,
        data: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        for m in modules:
            data = m(data)
        return data

    def apply_pre_flatten(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        return self._run_stage(self.pre_flatten, data)

    def apply_raw(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        return self._run_stage(self.raw, data)

    def apply_embedding(
        self, data: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        return self._run_stage(self.embedding, data)

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
