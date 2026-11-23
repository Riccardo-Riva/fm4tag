#!/usr/bin/env python
"""One table out of a directory of profile_timing.py reports.

Reads every ``<point>/profile_<phase>.json`` under the given directory (as
written by scripts/submit_s0_benchmark.sh) and prints, per point: workload
overrides, row mode, chunk size, batch size, ms/step, jets/s, peak GPU memory
and the share of the step spent in the main components.  Points without a
report (OOM, crash) are listed with the last line of their log.  Writes
``<dir>/table.csv``.

    scripts/profile_table.py experiments/2026-07_kernel-profiling/outputs/s0_<TS>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# FinetuneModule names its encoders `backbone`, PretrainModule `encoders`.
COMPONENTS = {
    'tracks': ('backbone.tracks', 'encoders.tracks'),
    'jets': ('backbone.jets', 'encoders.jets'),
    'agg': ('aggregator',),
}


def _override(overrides: list[str], key: str, default=None):
    for o in overrides:
        k, _, v = o.partition('=')
        if k.lstrip('+') == key:
            return v
    return default


def _row(point: Path) -> dict:
    reports = sorted(point.glob('profile_*.json'))
    if not reports:
        log = point / 'log.txt'
        last = log.read_text().strip().splitlines()[-1:] if log.is_file() else ['']
        return {'point': point.name, 'error': last[0][:120]}
    r = json.loads(reports[0].read_text())
    ov = r['overrides']
    ms = r['total_ms_per_step']
    mods = {m['name']: m['fwd_ms'] + m['bwd_ms'] for m in r['modules']}
    row = {
        'point': point.name,
        'phase': r['phase'],
        'arm': _override(ov, 'arm'),
        'jc_down': _override(ov, 'finetune.loss_weights.jet_contrastive', '0'),
        'jc_pre': _override(ov, 'pretrain.loss_weights.jet_contrastive', '0'),
        'encoders': _override(ov, 'encoders'),
        'chunk': _override(ov, 'chunk_size', 'null'),
        'batch': r['batch_size'],
        'ms_step': round(ms, 1),
        'jets_s': round(r['batch_size'] / ms * 1e3),
        'peak_gib': round(r['peak_mem_gb'], 2),
        'data_ms': round(r['dataloader'].get('steady_mean_ms', float('nan')), 1),
    }
    for short, names in COMPONENTS.items():
        row[f'{short}_%'] = round(100 * sum(mods.get(n, 0.0) for n in names) / ms)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    points = sorted(p for p in args.root.iterdir() if p.is_dir())
    table = pd.DataFrame([_row(p) for p in points])
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(table.to_string(index=False))
    out = args.root / 'table.csv'
    table.to_csv(out, index=False)
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()
