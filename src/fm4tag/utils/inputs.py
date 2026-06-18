"""Helpers for reading an object's feature ``inputs`` config.

Both the global (jet) object and constituent objects declare their features
under ``variables.<obj>.inputs``.  Two schemas are accepted:

* **Dict** (preferred, used by tracks and now jets)::

      inputs:
        continuous:  [feat_a, feat_b, ...]
        categorical: [cat_a, cat_b, ...]
        cat_classes:
          cat_a: [0, 1, 2]
          cat_b: [0, 1]

* **Flat list** (legacy, global-only)::

      inputs: [feat_a, feat_b, ...]   # all continuous, no categorical

:func:`resolve_object_inputs` normalises either form to the same triple so
callers (dataset, encoder builder, model-summary script) share one code path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def resolve_object_inputs(inputs) -> tuple[list[str], list[str], dict]:
    """Return ``(continuous, categorical, cat_classes)`` for an object's inputs.

    Accepts either a flat list (legacy → all continuous, no categorical) or a
    dict with ``continuous`` / ``categorical`` / ``cat_classes`` keys.  Works for
    plain Python containers and OmegaConf ``ListConfig`` / ``DictConfig``.

    Args:
        inputs: The ``variables.<obj>.inputs`` node.

    Returns:
        ``(continuous, categorical, cat_classes)`` — two lists of feature names
        and a mapping ``{categorical_feature: [class values]}`` (empty when there
        are no categorical features).
    """
    # A Mapping (dict / DictConfig) uses the explicit schema; anything else that
    # is a non-string sequence (list / ListConfig) is the legacy continuous-only
    # flat list.
    if isinstance(inputs, Mapping):
        continuous = list(inputs.get('continuous', []) or [])
        categorical = list(inputs.get('categorical', []) or [])
        cat_classes = dict(inputs.get('cat_classes', {}) or {})
        return continuous, categorical, cat_classes

    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        return list(inputs), [], {}

    raise TypeError(
        f'Unsupported inputs schema: expected a mapping or a list, got {type(inputs)!r}.'
    )
