"""Largest batch size per campaign cell that fits the GPU with dropout, and what dropout costs per step.

For every cell of a cells file (first pretraining seed of each arm), one short training (``--steps`` steps, 3
validation batches) is run per candidate batch size, from the cell's own size down in steps of 10%
(rounded down to a multiple of the chunk size), until one finishes with a peak allocated memory of at most
``--max-peak-gib`` (the highest peak of the no-dropout baselines was 41.0 GiB of 44.4).  The accepted size is
then timed with ``--time-steps`` and ``--time-steps`` + 100 steps (s/step = difference of the wall times), and
so is the cell at its own batch size WITHOUT dropout (the dropout keys removed), so the cost of dropout is a
ratio measured the same way.  Two workers, one per GPU the job was allocated (the first two entries of the
CUDA_VISIBLE_DEVICES slurm set; never a fixed 0 and 1, which can be another job's cards).

Run inside a 2-GPU slurm job (slurm/dropout_batch_sweep.sh)::

    python scripts/dropout_batch_sweep.py --cells scripts/configs/campaign/D_drop.cells --out <dir>

``--cell-names`` lists the cells to sweep (default: the six downstream arms of D_drop.cells), dealt round robin
to the two cards, so put the longest first; the cells need ``datamodule.batch_size=`` and ``chunk_size=`` as
literals or with ``${ENCODERS}`` / ``${CHUNK}`` (the only variables filled by default).

To only MEASURE cells at their own batch size (peak memory and s/step, e.g. the WSD cells at batch 12800)::

    ... --fractions 1.0 --max-peak-gib 45 --skip-baseline --extra finetune.warmup_epochs=0.2
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / 'src' / 'fm4tag' / 'configs'
DROPOUT_KEYS = (
    'encoders.global_encoder.layers.0.attn_dropout',
    'encoders.global_encoder.layers.0.ff_dropout',
    'encoders.constituents.tracks.layers.0.attn_dropout',
    'encoders.constituents.tracks.layers.0.ff_dropout',
    'aggregator.attn_dropout',
    'aggregator.ff_dropout',
    'head.mlp_dropout',
)
GIB = 1024.0
EXTRA: list[
    str
] = []  # --extra: overrides appended to every run (after the cell's, so they win)


def read_cells(path: Path, env: dict[str, str]) -> dict[str, list[str]]:
    """``{cell: overrides}`` with ``${VAR}`` filled, ``@`` tokens dropped, ``encoder_ckpt=@dir`` resolved."""
    cells = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, *tokens = line.split()
        overrides = []
        for tok in tokens:
            if tok.startswith('@'):
                continue
            tok = re.sub(r'\$\{(\w+)\}', lambda m: env[m.group(1)], tok)
            if tok.startswith('encoder_ckpt=@'):
                ckpts = sorted(
                    glob.glob(
                        f'{tok.removeprefix("encoder_ckpt=@")}/*/version_*/checkpoints/last.ckpt'
                    ),
                    key=os.path.getmtime,
                )
                if not ckpts:
                    raise FileNotFoundError(f'{name}: no last.ckpt under {tok}')
                tok = f'encoder_ckpt={ckpts[-1]}'
            overrides.append(tok)
        cells[name] = overrides
    return cells


def with_batch(overrides: list[str], batch: int) -> list[str]:
    return [
        f'datamodule.batch_size={batch}'
        if o.startswith('datamodule.batch_size=')
        else o
        for o in overrides
    ]


def without_dropout(overrides: list[str]) -> list[str]:
    return [o for o in overrides if o.split('=')[0] not in DROPOUT_KEYS]


def batch_of(overrides: list[str]) -> int:
    return int(
        next(o for o in overrides if o.startswith('datamodule.batch_size=')).split('=')[
            1
        ]
    )


def build_cmd(run_dir: Path, overrides: list[str], steps: int) -> list[str]:
    """The fm4tag command of a ``steps``-step, 1-epoch training of a cell."""
    return [
        str(REPO / '.venv' / 'bin' / 'fm4tag'),
        f'--config-path={CONFIG_DIR}',
        '--config-name=campaign_ssl',
        'action=fit',
        'trainer.max_epochs=1',
        f'+trainer.limit_train_batches={steps}',
        '+trainer.limit_val_batches=3',
        'datamodule.num_workers=6',
        *overrides,
        *EXTRA,
        'seed=4242',
        'experiment_name=dropout_sweep',
        f'output_dir={run_dir}',
    ]


def run_once(
    gpu: int, run_dir: Path, overrides: list[str], steps: int, dry: bool
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(run_dir, overrides, steps)
    if dry:
        print(' '.join(cmd))
        return {'rc': 0, 'secs': 0.0, 'peak_gib': float('nan'), 'finite': True}
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'WANDB_MODE': 'disabled'}
    t0 = time.time()
    with (
        open(run_dir / 'rank_0.out', 'w') as out,
        open(run_dir / 'rank_0.err', 'w') as err,
    ):
        rc = subprocess.run(
            cmd, cwd=run_dir, env=env, stdout=out, stderr=err
        ).returncode
    secs = time.time() - t0
    peak, finite = float('nan'), False
    files = glob.glob(str(run_dir / '**' / 'metrics.csv'), recursive=True)
    if rc == 0 and files:
        df = pd.concat([pd.read_csv(f) for f in files])
        if 'mem/gpu_peak_MiB' in df:
            peak = float(df['mem/gpu_peak_MiB'].max()) / GIB
        loss = (
            df['train/loss_step'].dropna()
            if 'train/loss_step' in df
            else pd.Series(dtype=float)
        )
        finite = bool(
            len(loss) and loss.map(pd.notna).all() and (loss.abs() < 1e6).all()
        )
    return {'rc': rc, 'secs': secs, 'peak_gib': peak, 'finite': finite}


def sec_per_step(gpu, run_dir, overrides, short, dry) -> float:
    a = run_once(gpu, run_dir / f'time{short}', overrides, short, dry)
    b = run_once(gpu, run_dir / f'time{short + 100}', overrides, short + 100, dry)
    return (
        (b['secs'] - a['secs']) / 100.0
        if a['rc'] == 0 and b['rc'] == 0
        else float('nan')
    )


def work(gpu, cells, names, args, results, lock) -> None:
    for name in names:
        ov = cells[name]
        b0, chunk = (
            batch_of(ov),
            int(next(o for o in ov if o.startswith('chunk_size=')).split('=')[1]),
        )
        base_dir = args.out / name
        row = {'cell': name, 'batch_baseline': b0}
        # baseline timing, no dropout, at the cell's own batch size
        row['s_per_step_nodrop'] = (
            float('nan')
            if args.skip_baseline
            else sec_per_step(
                gpu,
                base_dir / 'nodrop',
                without_dropout(ov),
                args.time_steps,
                args.dry_run,
            )
        )
        accepted = None
        for frac in args.fractions:
            batch = int(b0 * frac) // chunk * chunk
            res = run_once(
                gpu,
                base_dir / f'B{batch}',
                with_batch(ov, batch),
                args.steps,
                args.dry_run,
            )
            print(f'[gpu {gpu}] {name} B={batch}: {res}', flush=True)
            row.setdefault('tried', []).append(
                (batch, res['rc'], round(res['peak_gib'], 1))
            )
            if (
                res['rc'] == 0
                and res['finite']
                and (args.dry_run or res['peak_gib'] <= args.max_peak_gib)
            ):
                accepted, row['peak_gib'] = batch, res['peak_gib']
                break
        row['batch_dropout'] = accepted
        if accepted:
            row['s_per_step_drop'] = sec_per_step(
                gpu,
                base_dir / f'B{accepted}',
                with_batch(ov, accepted),
                args.time_steps,
                args.dry_run,
            )
        with lock:
            results.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--cells', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100, help='steps of a fit test')
    parser.add_argument(
        '--time-steps',
        type=int,
        default=40,
        help='short run of the timing pair (long = +100)',
    )
    parser.add_argument('--max-peak-gib', type=float, default=41.0)
    parser.add_argument(
        '--fractions',
        type=float,
        nargs='+',
        default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    )
    parser.add_argument(
        '--cell-names',
        nargs='+',
        default=[
            'ftjc_p1s42_d02',
            'ft_p0s42_d02',
            'sc_d02',
            'scjc_clip1_d02',
            'frjc_p1s42_d02',
            'fr_p0s42_d02',
        ],
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='no timing of the cell without dropout (measuring cells as they are)',
    )
    parser.add_argument(
        '--extra',
        nargs='*',
        default=[],
        help='overrides appended to every run, e.g. finetune.warmup_epochs=0.2 for '
        'WSD cells (their 1.6-epoch warmup does not fit a 1-epoch test)',
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    EXTRA.extend(args.extra)

    env = {**os.environ}
    env.setdefault('ENCODERS', 'rowcol_concat_v0')
    env.setdefault('CHUNK', '512')
    cells = read_cells(args.cells, env)
    args.out.mkdir(parents=True, exist_ok=True)
    # two workers, the long cells spread over both GPUs
    devices = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d]
    if len(devices) < 2 and not args.dry_run:
        raise SystemExit(
            f'need the 2 GPUs of the job in CUDA_VISIBLE_DEVICES, got {devices}'
        )
    devices = (devices + ['0', '1'])[:2]
    # round robin over the two cards, so name the longest cells first
    plan = {gpu: args.cell_names[i::2] for i, gpu in enumerate(devices)}
    results, lock = [], threading.Lock()
    threads = [
        threading.Thread(target=work, args=(gpu, cells, names, args, results, lock))
        for gpu, names in plan.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    table = pd.DataFrame(results)
    if 'batch_dropout' in table:
        table['step_cost'] = table['s_per_step_drop'] / table['s_per_step_nodrop']
        table['jets_per_s_ratio'] = (
            table['batch_dropout'] / table['s_per_step_drop']
        ) / (table['batch_baseline'] / table['s_per_step_nodrop'])
    table.to_csv(args.out / 'results.csv', index=False)
    print(table.to_string())


if __name__ == '__main__':
    main()
