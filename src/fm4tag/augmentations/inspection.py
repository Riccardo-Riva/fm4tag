"""Inspect augmentation pipelines by collecting their per-view outputs.

Standalone successor of the old pretraining module's ``predict_step``: it only
needs the views and a batch — no model, no Trainer — so it can be called
directly from a notebook to plot feature distributions after each pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from einops import rearrange

from .base import Compose


@torch.no_grad()
def collect_view_outputs(
    views: Sequence[Compose],
    batch: dict,
    global_object: str,
    constituent_objects: Sequence[str],
) -> dict:
    """Apply each view's PRE_FLATTEN / RAW augmentations and collect the results.

    Unlike the training loss (which shares the original valid mask across
    views), each view here uses its **own** valid mask, so mask-modifying
    augmentations (e.g. :class:`TrackDropout`) are visible in the output.

    Args:
        views:               Augmentation pipelines, one per view.
        batch:               Batch dict as produced by the datamodule
                             (``'global'`` + ``'constituents'``).
        global_object:       Name of the global object (e.g. ``'jets'``).
        constituent_objects: Names of the constituent objects.

    Returns a dict structured as::

        {
            '<global_object>': {
                'original': Tensor (B, F_g),
                'views': [{'raw': Tensor (B, F_g)}, ...],
            },
            'constituents': {
                '<obj>': {
                    'original': {
                        'categorical': Tensor (N, F_cat),
                        'continuous':  Tensor (N, F_con),
                    },
                    'views': [
                        {
                            'pre_flatten': {
                                'categorical': Tensor (N_v, F_cat),
                                'continuous':  Tensor (N_v, F_con),
                                'valid':       Tensor (B, C),
                            },
                            'raw': {
                                'categorical': Tensor (N_v, F_cat),
                                'continuous':  Tensor (N_v, F_con),
                            },
                        },
                        ...
                    ],
                },
            },
        }
    """
    result: dict = {}

    # ── Global object ────────────────────────────────────────────────────
    x_global = batch['global']
    view_results = []
    for view in views:
        raw_out = view.apply_raw({'continuous': x_global.clone()})
        view_results.append({'raw': raw_out['continuous'].cpu()})
    result[global_object] = {
        'original': x_global.cpu(),
        'views': view_results,
    }

    # ── Constituent objects ───────────────────────────────────────────────
    result['constituents'] = {}
    for obj_name in constituent_objects:
        const = batch['constituents'][obj_name]

        # Original flattened using the unaugmented valid mask.
        valids_orig = rearrange(const['valid'], 'b c -> (b c)')
        x_categ_orig = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_orig]
        x_cont_orig = rearrange(const['continuous'], 'b c f -> (b c) f')[valids_orig]

        view_results = []
        for view in views:
            # PRE_FLATTEN — use the view's own valid mask here.
            data_pre = view.apply_pre_flatten(
                {
                    'categorical': const['categorical'].clone(),
                    'continuous': const['continuous'].clone(),
                    'valid': const['valid'].clone(),
                }
            )
            valids_v = rearrange(data_pre['valid'], 'b c -> (b c)')
            x_cat_v = rearrange(data_pre['categorical'], 'b c f -> (b c) f')[valids_v]
            x_con_v = rearrange(data_pre['continuous'], 'b c f -> (b c) f')[valids_v]

            # RAW stage
            data_raw = view.apply_raw({'categorical': x_cat_v, 'continuous': x_con_v})

            view_results.append(
                {
                    'pre_flatten': {
                        'categorical': x_cat_v.cpu(),
                        'continuous': x_con_v.cpu(),
                        'valid': data_pre['valid'].cpu(),
                    },
                    'raw': {
                        'categorical': data_raw['categorical'].cpu(),
                        'continuous': data_raw['continuous'].cpu(),
                    },
                }
            )

        result['constituents'][obj_name] = {
            'original': {
                'categorical': x_categ_orig.cpu(),
                'continuous': x_cont_orig.cpu(),
            },
            'views': view_results,
        }

    return result
