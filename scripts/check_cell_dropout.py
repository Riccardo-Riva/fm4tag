"""Check that every dropout of the model a campaign cell builds is at the expected probability.

Composes each cell of a cells file like ``campaign_cell_files.py`` (``${VAR}`` filled from the environment,
``encoder_ckpt=@...`` replaced by a placeholder), builds encoders + aggregator + head on the CPU and walks the
modules: ``nn.Dropout.p`` plus the ``dropout`` attribute of every attention.  Prints the counts per part and exits
non-zero if any dropout differs from ``--expected`` (or there is none).

Usage::

    ENCODERS=rowcol_concat_v0 CHUNK=512 python scripts/check_cell_dropout.py \
        --cells scripts/configs/campaign/D_drop.cells --expected 0.2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from torch import nn

from fm4tag.utils.model_builders import build_aggregator, build_encoders, build_head

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'src' / 'fm4tag' / 'configs'


def dropouts(module: nn.Module) -> list[float]:
    found = [m.p for m in module.modules() if isinstance(m, nn.Dropout)]
    found += [
        m.dropout
        for m in module.modules()
        if hasattr(m, 'dropout') and not isinstance(m, nn.Dropout)
    ]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--cells', type=Path, required=True)
    parser.add_argument('--expected', type=float, required=True)
    args = parser.parse_args()

    bad = 0
    for line in args.cells.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, *tokens = line.split()
        overrides = []
        for tok in tokens:
            if tok.startswith('@'):
                continue
            tok = re.sub(r'\$\{(\w+)\}', lambda m: os.environ[m.group(1)], tok)
            overrides.append(
                'encoder_ckpt=/placeholder' if tok.startswith('encoder_ckpt=@') else tok
            )
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(config_name='campaign_ssl', overrides=overrides)
        encoders = build_encoders(cfg)
        aggregator = build_aggregator(cfg, encoders)
        head = build_head(cfg, aggregator)
        parts = {
            'encoders': dropouts(encoders),
            'aggregator': dropouts(aggregator),
            'head': dropouts(head),
        }
        summary = ', '.join(
            f'{k}: {len(v)} x {sorted(set(v))}' for k, v in parts.items()
        )
        wrong = [
            k
            for k, v in parts.items()
            if not v or any(abs(p - args.expected) > 1e-12 for p in v)
        ]
        bad += bool(wrong)
        print(f'{"FAIL" if wrong else "ok  "} {name}: {summary}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
