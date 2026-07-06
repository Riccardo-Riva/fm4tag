"""Loading pretrained encoder / aggregator weights from Lightning checkpoints."""

from __future__ import annotations

import warnings

import torch


def load_pretrained_encoders(
    encoders: torch.nn.ModuleDict,
    ckpt_path: str,
) -> torch.nn.ModuleDict:
    """Load all encoder weights from a pretraining-module checkpoint.

    For each encoder in the dict, tries the current checkpoint format
    (``encoders.<obj_name>.*``) first, then falls back to the legacy
    single-encoder format (``encoder.*``).  Emits a warning for any
    encoder whose weights cannot be found, leaving it randomly initialised.

    Args:
        encoders:  :class:`~torch.nn.ModuleDict` built by
                   :func:`~fm4tag.utils.model_builders.build_encoders`.
        ckpt_path: Path to a pretraining-module Lightning checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    for obj_name, encoder in encoders.items():
        prefix_new = f'encoders.{obj_name}.'
        enc_state = {
            k[len(prefix_new) :]: v
            for k, v in state.items()
            if k.startswith(prefix_new)
        }

        if not enc_state:
            prefix_old = 'encoder.'
            enc_state = {
                k[len(prefix_old) :]: v
                for k, v in state.items()
                if k.startswith(prefix_old)
            }

        if enc_state:
            encoder.load_state_dict(enc_state)
        else:
            warnings.warn(
                f"No pretrained weights found for encoder '{obj_name}' in "
                f"'{ckpt_path}'. Using random initialisation.",
                stacklevel=2,
            )

    return encoders


def load_pretrained_aggregator(
    aggregator: torch.nn.Module,
    ckpt_path: str,
) -> torch.nn.Module:
    """Load aggregator weights from a pretraining checkpoint.

    Matches ``aggregator.*`` keys saved by a pretraining module.  These are
    meaningful only when the pretraining loss contained a jet-contrastive term
    (otherwise the aggregator received no gradient and its weights are the
    random init).  Emits a warning and leaves the aggregator randomly
    initialised if no matching weights are found or their shapes do not match
    the current architecture.

    Args:
        aggregator: Aggregator built by
                    :func:`~fm4tag.utils.model_builders.build_aggregator`.
        ckpt_path:  Path to a pretraining-module Lightning checkpoint.
    """
    # A parameterless aggregator (e.g. no constituent objects → empty
    # const_transformer) has nothing to load; skip silently so the "no weights
    # found" warning below is reserved for genuine checkpoint mismatches.
    if sum(p.numel() for p in aggregator.parameters()) == 0:
        return aggregator

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    prefix = 'aggregator.'
    agg_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}

    if not agg_state:
        warnings.warn(
            f"No pretrained aggregator weights found in '{ckpt_path}'. "
            'Using random initialisation.',
            stacklevel=2,
        )
        return aggregator

    try:
        aggregator.load_state_dict(agg_state)
    except RuntimeError as exc:
        warnings.warn(
            f"Pretrained aggregator weights in '{ckpt_path}' do not match the "
            f'current aggregator architecture ({exc}). Using random '
            'initialisation.',
            stacklevel=2,
        )

    return aggregator
