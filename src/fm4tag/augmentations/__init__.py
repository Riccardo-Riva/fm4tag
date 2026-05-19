"""Augmentation framework for fm4tag.

Public API
----------

* :class:`Augmentation`     — base class
* :class:`Stage`            — enum: ``PRE_FLATTEN``, ``RAW``, ``EMBEDDING``
* :class:`Compose`          — applies a list of augmentations grouped by stage
* :func:`build_from_config` — construct a :class:`Compose` from a YAML list

Built-in augmentations
----------------------

* :class:`Identity`
* :class:`CutMix`
* :class:`FeatureDropout`
* :class:`GaussianNoise`
* :class:`TrackDropout`
* :class:`Mixup`

Example YAML config
-------------------

::

    # configs/pretrain_experiment.yaml
    pretrain:
      views:
        - augmentations: []                     # clean view
        - augmentations:
            - _target_: fm4tag.augmentations.TrackDropout
              drop_prob: 0.15
            - _target_: fm4tag.augmentations.CutMix
              lam: 0.7
            - _target_: fm4tag.augmentations.Mixup
              lam: 0.8
        - augmentations:
            - _target_: fm4tag.augmentations.FeatureDropout
              corrupt_frac: 0.4
            - _target_: fm4tag.augmentations.GaussianNoise
              space: embedding
              sigma: 0.05

Each view's ``augmentations`` list is passed to
:func:`build_from_config` to produce a :class:`Compose` that the
pretraining module applies at the appropriate point in the encoder pipeline.
"""

from __future__ import annotations

from .base import Augmentation, Compose, Stage, build_from_config
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
    'build_from_config',
    'CutMix',
    'FeatureDropout',
    'GaussianNoise',
    'Identity',
    'Mixup',
    'TrackDropout',
]
