"""Print the Lightning ModelSummary for an fm4tag config without running training.

Usage::

    uv run python scripts/model_summary.py --config src/fm4tag/configs/default.yaml
    uv run python scripts/model_summary.py --config src/fm4tag/configs/model_0.yaml --phase pretrain
    uv run python scripts/model_summary.py --config src/fm4tag/configs/default.yaml --max-depth 3
"""

from __future__ import annotations

import argparse
import sys

from omegaconf import OmegaConf
from lightning.pytorch.utilities.model_summary import ModelSummary

# Make sure the package is importable when running from project root.
sys.path.insert(0, 'src')

from fm4tag.engine import _build_encoders
from fm4tag.models import FinetuneModule, PretrainModule
from fm4tag.models.components.heads import MultiStreamClassifierHead


def build_module(cfg, phase: str):
    encoders = _build_encoders(cfg)

    if phase == 'pretrain':
        return PretrainModule(encoders, cfg)

    if phase == 'finetune':
        head_cfg = cfg.head
        n_classes = len(cfg.variables[cfg.global_object].unique_labels)
        from fm4tag.utils import resolve_object_inputs

        _g_con, _g_cat, _ = resolve_object_inputs(
            cfg.variables[cfg.global_object].inputs
        )
        n_global_features = len(_g_con) + len(_g_cat)
        n_constituent_features = [
            len(cfg.variables[obj].inputs.continuous)
            + len(cfg.variables[obj].inputs.categorical)
            for obj in cfg.constituent_objects
        ]
        head = MultiStreamClassifierHead(
            dim=cfg.encoder.dim,
            n_global_features=n_global_features,
            n_constituent_features=n_constituent_features,
            y_dim=n_classes,
            cls_dim=head_cfg.get('cls_dim', None),
            mlp_dropout=head_cfg.get('mlp_dropout', 0.0),
            ff_dropout=head_cfg.get('ff_dropout', 0.0),
            attn_dropout=head_cfg.get('attn_dropout', 0.0),
            ff_mult=head_cfg.get('ff_mult', 4),
            heads=head_cfg.get('heads', 8),
            dim_head=head_cfg.get('dim_head', 16),
            depth=head_cfg.get('depth', 3),
        )
        return FinetuneModule(encoders, head, cfg)

    raise ValueError(f"phase must be 'pretrain' or 'finetune', got {phase!r}")


def main():
    parser = argparse.ArgumentParser(
        description='Print model summary from a config file.'
    )
    parser.add_argument('--config', required=True, help='Path to a YAML config file.')
    parser.add_argument(
        '--phase',
        default=None,
        choices=['pretrain', 'finetune'],
        help='Override the phase in the config (pretrain | finetune).',
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help='Override callbacks.model_summary.max_depth from the config.',
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    phase = args.phase or cfg.get('phase', 'finetune')
    max_depth = args.max_depth or cfg.get('callbacks', {}).get('model_summary', {}).get(
        'max_depth', 2
    )

    module = build_module(cfg, phase)
    print(ModelSummary(module, max_depth=max_depth))


if __name__ == '__main__':
    main()
