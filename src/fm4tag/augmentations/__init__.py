"""Augmentation framework for fm4tag.

Public API
----------

* :class:`Augmentation`     — base class
* :class:`Stage`            — enum: ``PRE_FLATTEN``, ``RAW``, ``EMBEDDING``
* :class:`Compose`          — applies a list of augmentations grouped by stage
* :func:`build_from_config` — construct a :class:`Compose` from a YAML list
* :func:`register`, :func:`get` — registry helpers (decorate custom augs)

Built-in augmentations (registered names)
-----------------------------------------

* ``identity``                                  — :class:`Identity`
* ``cutmix``                                    — :class:`CutMix`
* ``feature_dropout`` / ``scarf``               — :class:`FeatureDropout`
* ``gaussian_noise``                            — :class:`GaussianNoise`
* ``track_dropout`` / ``constituent_dropout``   — :class:`TrackDropout`
* ``mixup``                                     — :class:`Mixup`

Example YAML config
-------------------

::

    # configs/pretrain_experiment.yaml
    pretrain:
      views:
        - augmentations: []                     # clean view
        - augmentations:
            - name: track_dropout
              drop_prob: 0.15
            - name: cutmix
              lam: 0.7
            - name: mixup
              lam: 0.8
        - augmentations:
            - name: scarf
              corrupt_frac: 0.4
            - name: gaussian_noise
              space: embedding
              sigma: 0.05

Each view's ``augmentations`` list is passed to
:func:`build_from_config` to produce a :class:`Compose` that the
pretraining module applies at the appropriate point in the encoder pipeline.

The legacy functional helpers :func:`embed_data`, :func:`add_noise`,
:func:`mixup_data` previously living in ``augmentations.augmentations``
remain available there for backward compatibility — the new
:class:`Augmentation` system is layered on top, not a replacement of
``embed_data`` (which is still the right place to call the encoder's
embedding layers).
"""

from __future__ import annotations

from .base import Augmentation, Compose, Stage, build_from_config, get, register

# Importing the concrete modules below triggers their ``@register(...)``
# decorators, populating the registry.  Keep these imports even if their
# names aren't used directly elsewhere.
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
    'get',
    'register',
    'CutMix',
    'FeatureDropout',
    'GaussianNoise',
    'Identity',
    'Mixup',
    'TrackDropout',
]
