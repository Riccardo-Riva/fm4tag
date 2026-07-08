"""Augmentation framework for fm4tag.

Public API
----------

* :class:`Augmentation`  — base class
* :class:`Stage`         — enum: ``PRE_FLATTEN``, ``RAW``, ``EMBEDDING``
* :class:`Compose`       — applies a list of augmentations grouped by stage
* :func:`collect_view_outputs` — per-view augmented data for inspection/plots

Built-in augmentations
----------------------

* :class:`Identity`
* :class:`CutMix`
* :class:`FeatureDropout`
* :class:`GaussianNoise`
* :class:`TrackDropout`
* :class:`Mixup`
* :class:`ContinuousDilation`
* :class:`ContinuousFeatureDilation`
* :class:`CategoricalShift`

Config usage
------------

Views are specified as a top-level Hydra ``_target_`` list (shared by the
pretrain and finetune phases) and instantiated recursively::

    views:
      - _target_: fm4tag.augmentations.Compose
        augmentations: []                           # clean / identity view
      - _target_: fm4tag.augmentations.Compose
        augmentations:
          - _target_: fm4tag.augmentations.TrackDropout
            drop_prob: 0.15
          - _target_: fm4tag.augmentations.CutMix
            lam: 0.3
      - _target_: fm4tag.augmentations.Compose
        augmentations:
          - _target_: fm4tag.augmentations.FeatureDropout
            corrupt_frac: 0.4
          - _target_: fm4tag.augmentations.GaussianNoise
            space: embedding
            sigma: 0.05
"""

from __future__ import annotations

from .base import Augmentation, Compose, Stage
from .inspection import collect_view_outputs
from .categorical_shift import CategoricalShift
from .continuous_dilation import ContinuousDilation
from .continuous_feature_dilation import ContinuousFeatureDilation
from .cutmix import CutMix
from .feature_dropout import FeatureDropout
from .gaussian_noise import GaussianNoise
from .identity import Identity
from .mixup import Mixup
from .track_dropout import TrackDropout

__all__ = [
    'Augmentation',
    'Compose',
    'Stage',
    'collect_view_outputs',
    'CategoricalShift',
    'ContinuousDilation',
    'ContinuousFeatureDilation',
    'CutMix',
    'FeatureDropout',
    'GaussianNoise',
    'Identity',
    'Mixup',
    'TrackDropout',
]
