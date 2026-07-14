"""Print Lightning's ModelSummary for the model a config builds — no data, no GPU.

This is the same table the ``ModelSummary`` callback prints at the start of
training, because it is the same class applied to the same module: the model is
built with the runner's own ``build_encoders`` / ``build_aggregator`` /
``build_head``, so the counts are the ones training will instantiate.

The config is composed through Hydra, not read with ``OmegaConf.load``, so the
``defaults:`` list is resolved and the selected ``encoders`` group is actually
applied — and every Hydra override works, which is the quick way to compare
architectures without editing YAML.

Usage::

    scripts/model_summary.py                         # default.yaml, pretrain
    scripts/model_summary.py --phase finetune        # adds the classifier head
    scripts/model_summary.py --config-name model_0   # a different config

    scripts/model_summary.py encoders=col_v0         # compare architectures
    scripts/model_summary.py encoders.constituents.tracks.layers.0.num_inds=64

    scripts/model_summary.py --max-depth 4           # deeper module tree
    scripts/model_summary.py --breakdown             # + per-component % shares
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate as hydra_instantiate
from lightning.pytorch.utilities.model_summary import ModelSummary

from fm4tag.utils.model_builders import build_aggregator, build_encoders, build_head

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'src' / 'fm4tag' / 'configs')


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _fmt(n: int) -> str:
    """Human-readable parameter count: 763776 -> '763.8 K'."""
    if n >= 1_000_000:
        return f'{n / 1e6:.2f} M'
    if n >= 1_000:
        return f'{n / 1e3:.1f} K'
    return str(n)


def _print_tree(
    module: torch.nn.Module,
    total: int,
    *,
    name: str = '(model)',
    max_depth: int = 2,
    level: int = 0,
    prefix: str = '',
) -> None:
    """Print named_children with parameter counts and share of the total."""
    n = _count(module)
    if n == 0:
        return

    share = 100.0 * n / total if total else 0.0
    bar = '#' * int(round(share / 4))  # 25 chars == 100%
    print(f'  {prefix + name:<46} {_fmt(n):>9} {share:>6.1f}%  {bar}')

    if level >= max_depth:
        return

    children = list(module.named_children())
    for i, (child_name, child) in enumerate(children):
        last = i == len(children) - 1
        child_prefix = prefix.replace('|- ', '|  ').replace('`- ', '   ')
        _print_tree(
            child,
            total,
            name=child_name,
            max_depth=max_depth,
            level=level + 1,
            prefix=child_prefix + ('`- ' if last else '|- '),
        )


def _print_geometry(cfg, encoders: torch.nn.ModuleDict) -> None:
    """Token geometry per object — what actually drives the parameter counts.

    Row attention now runs per token at width ``dim`` (not on a flattened
    ``N*dim`` vector), so ``dim`` is the width of both the column and the row step.
    """
    print('\nEncoder geometry')
    print(f'  {"object":<12} {"N tokens":>9} {"dim":>5}   layers')
    constituents = cfg.encoders.get('constituents', {})
    for obj_name, enc in encoders.items():
        if hasattr(enc, 'num_categories'):  # constituent Encoder
            n_tokens = enc.num_categories + enc.num_continuous
        else:  # GlobalEncoder / GlobalTransformerEncoder
            n_tokens = enc.num_features
        dim = enc.dim

        node = (
            constituents[obj_name]
            if obj_name in constituents
            else cfg.encoders.global_encoder
        )
        parts = []
        for lc in node.get('layers', []):
            desc = f'{lc["type"]}x{lc.get("depth", 1)}'
            if 'num_inds' in lc:
                inds = lc['num_inds']
                desc += f' (ISAB m={inds})' if inds is not None else ' (all-pairs row)'
            parts.append(desc)
        print(f'  {obj_name:<12} {n_tokens:>9} {dim:>5}   {", ".join(parts) or "-"}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--phase', choices=['pretrain', 'finetune'], default='pretrain')
    ap.add_argument(
        '--config-name', default='default', help='config name, without .yaml'
    )
    ap.add_argument('--config-path', default=CONFIG_DIR)
    ap.add_argument('--max-depth', type=int, default=2, help='module-tree depth')
    ap.add_argument(
        '--breakdown',
        action='store_true',
        help='also print the per-component parameter shares and encoder geometry',
    )
    ap.add_argument(
        'overrides', nargs='*', help='hydra overrides, e.g. encoders=col_v0'
    )
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=args.config_path):
        cfg = compose(
            config_name=args.config_name,
            overrides=[f'phase={args.phase}', *args.overrides],
        )

    encoders = build_encoders(cfg)
    aggregator = build_aggregator(cfg, encoders)
    views = [hydra_instantiate(v) for v in cfg.get('views', [])]

    if args.phase == 'pretrain':
        module = hydra_instantiate(
            cfg.pretrain,
            encoders=encoders,
            aggregator=aggregator,
            views=views,
            _convert_='all',
        )
    else:
        module = hydra_instantiate(
            cfg.finetune,
            encoders=encoders,
            aggregator=aggregator,
            head=build_head(cfg, aggregator),
            views=views,
            n_classes=len(cfg.variables[cfg.global_object].unique_labels),
            _convert_='all',
        )

    print()
    print(f'  config    {args.config_name}.yaml    phase={args.phase}')
    print(
        f'  objects   {cfg.global_object} (global) + {", ".join(cfg.constituent_objects)}'
    )
    if args.overrides:
        print(f'  overrides {" ".join(args.overrides)}')
    print()

    # The same table the ModelSummary callback prints at the start of training.
    print(ModelSummary(module, max_depth=args.max_depth))

    if args.breakdown:
        total = _count(module)
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        _print_geometry(cfg, encoders)

        print(f'\nParameter share (tree depth {args.max_depth})')
        _print_tree(module, total, max_depth=args.max_depth)

        print('-' * 72)
        print(f'  {"total":<46} {_fmt(total):>9}')
        print(f'  {"trainable":<46} {_fmt(trainable):>9}')
        if trainable != total:
            print(f'  {"frozen":<46} {_fmt(total - trainable):>9}')
        print(f'  {"size (fp32)":<46} {total * 4 / 1e6:>7.2f} MB')
        print('-' * 72)


if __name__ == '__main__':
    main()
