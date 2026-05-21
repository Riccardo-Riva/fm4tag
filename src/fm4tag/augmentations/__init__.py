"""Augmentation framework for fm4tag.

Public API
----------

* :class:`Augmentation`  — base class
* :class:`Stage`         — enum: ``PRE_FLATTEN``, ``RAW``, ``EMBEDDING``
* :class:`Compose`       — applies a list of augmentations grouped by stage

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

Views are specified as Hydra ``_target_`` lists and instantiated recursively::

    pretrain:
      views:
        - _target_: fm4tag.augmentations.Compose
          augmentations: []                           # clean / identity view
        - _target_: fm4tag.augmentations.Compose
          augmentations:
            - _target_: fm4tag.augmentations.TrackDropout
              drop_prob: 0.15
            - _target_: fm4tag.augmentations.CutMix
              lam: 0.7
        - _target_: fm4tag.augmentations.Compose
          augmentations:
            - _target_: fm4tag.augmentations.FeatureDropout
              corrupt_frac: 0.4
            - _target_: fm4tag.augmentations.GaussianNoise
              space: embedding
              sigma: 0.05
        - _target_: fm4tag.augmentations.Compose
          augmentations:
            - _target_: fm4tag.augmentations.ContinuousDilation
              alpha: 1.05
            - _target_: fm4tag.augmentations.CategoricalShift
              p: 0.3
"""

from __future__ import annotations

from .base import Augmentation, Compose, Stage
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
