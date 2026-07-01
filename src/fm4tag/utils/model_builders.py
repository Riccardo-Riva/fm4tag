from __future__ import annotations

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate as hydra_instantiate

from fm4tag.models.aggregator import TransformerAggregator

def build_encoders(cfg: DictConfig) -> torch.nn.ModuleDict:
    """Build one encoder per object (global + all constituents).

    Reads architecture from ``cfg.encoders``; each encoder's class is selected
    by its ``_target_`` key:

    * ``cfg.encoders.global_encoder`` → e.g. :class:`~fm4tag.models.GlobalEncoder`
      or :class:`~fm4tag.models.GlobalTransformerEncoder`, with ``num_features``
      injected from the global object's variable definitions at runtime
    * ``cfg.encoders.constituents.<name>`` → e.g. :class:`~fm4tag.models.Encoder`
      (one per constituent type, with ``categories`` and ``num_continuous``
      injected from the variable definitions at runtime)
    """
    encoders: dict[str, torch.nn.Module] = {}

    global_name = cfg.global_object
    n_global = len(cfg.variables[global_name].inputs)
    encoders[global_name] = hydra_instantiate(
        cfg.encoders.global_encoder, num_features=n_global
    )

    for obj_name in cfg.constituent_objects:
        obj_vars = cfg.variables[obj_name].inputs
        categories = [len(classes) for classes in obj_vars.cat_classes.values()]
        num_continuous = len(obj_vars.continuous)
        encoders[obj_name] = hydra_instantiate(
            cfg.encoders.constituents[obj_name],
            categories=categories,
            num_continuous=num_continuous,
        )

    return torch.nn.ModuleDict(encoders)

def build_aggregator(
    cfg: DictConfig,
    encoders: torch.nn.ModuleDict,
) -> TransformerAggregator:
    """Build the jet aggregator from ``cfg.aggregator`` and built encoders.

    Mirrors :func:`build_encoders`: the class is selected by
    ``cfg.aggregator._target_`` (e.g. :class:`~fm4tag.models.TransformerAggregator`)
    and its transformer hyper-parameters (``depth``, ``heads``, ``dim_head``,
    ``ff_mult``, ``ff_dropout``, ``attn_dropout``) come from the same config node.

    The output dims are injected at runtime from the already-built encoders'
    projection heads::

        global_dim = encoders[cfg.global_object].projector.layers[-1].out_features
        const_dims = [encoders[obj].projector.layers[-1].out_features
                      for obj in cfg.constituent_objects]
    """
    global_dim = encoders[cfg.global_object].projector.layers[-1].out_features
    const_dims = [
        encoders[obj_name].projector.layers[-1].out_features
        for obj_name in cfg.constituent_objects
    ]

    return hydra_instantiate(
        cfg.aggregator,
        global_dim=global_dim,
        const_dims=const_dims,
    )
