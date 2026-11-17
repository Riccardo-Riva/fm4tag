"""How much slower is a training when other trainings share its GPU (docs/gpu_packing.md)?

For each scenario, the co-tenant cells are started on the card as long trainings (``--cotenant-steps``) and kept
running while the measured cell is timed with the same method as scripts/dropout_batch_sweep.py: a ``--short``-step
and a ``--short`` + 100-step run one after the other, s/step = the difference of their wall times / 100.  Then the
co-tenants are stopped.  The s/step of each cell ALONE comes from the earlier measurement (``--alone``, results.csv of
dropout_batch_sweep.py).  A card's throughput gain from sharing is sum over its trainings of (alone / shared s/step),
assuming every co-tenant slows down like the measured cell of the same kind.

nvidia-smi is sampled every 10 s into ``<out>/nvidia_smi.csv`` (memory used per card, utilisation).

Run inside a 2-GPU slurm job (slurm/share_measure.sh); the two cards run their scenario lists in parallel.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dropout_batch_sweep as sweep  # noqa: E402

# scenario = (measured cell, co-tenant cells): CE + CE (the tightest memory), CE / frozen + JC / frozen in a mixed card
SCENARIOS = [
    [
        ('sc_d02_lr75', ['ft_p0_d02_lr75_bb10']),
        ('sc_d02_lr75', ['frjc_p1_d02_lr75', 'fr_p0_d02_lr75']),
    ],
    [
        ('frjc_p1_d02_lr75', ['sc_d02_lr75', 'fr_p0_d02_lr75']),
        ('fr_p0_d02_lr75', ['sc_d02_lr75', 'frjc_p1_d02_lr75']),
    ],
]


def scenario(gpu, card, idx, measured, cotenants, cells, args, rows, lock) -> None:
    out = args.out / f'card{card}_scenario{idx}'
    procs = []
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'WANDB_MODE': 'disabled'}
    for j, cell in enumerate(cotenants):
        run_dir = out / f'cotenant{j}_{cell}'
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = sweep.build_cmd(run_dir, cells[cell], args.cotenant_steps)
        procs.append(
            subprocess.Popen(
                cmd,
                cwd=run_dir,
                env=env,
                stdout=open(run_dir / 'rank_0.out', 'w'),
                stderr=open(run_dir / 'rank_0.err', 'w'),
            )
        )
        time.sleep(20)
    time.sleep(args.settle)  # co-tenants past their start-up, training
    alive = [p.poll() is None for p in procs]
    a = sweep.run_once(
        gpu,
        out / f'measured_{measured}_{args.short}',
        cells[measured],
        args.short,
        False,
    )
    b = sweep.run_once(
        gpu,
        out / f'measured_{measured}_{args.short + 100}',
        cells[measured],
        args.short + 100,
        False,
    )
    alive_end = [p.poll() is None for p in procs]
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    row = {
        'card': card,
        'scenario': idx,
        'measured': measured,
        'cotenants': ' + '.join(cotenants),
        's_per_step_shared': (b['secs'] - a['secs']) / 100.0
        if a['rc'] == 0 and b['rc'] == 0
        else float('nan'),
        'rc': f'{a["rc"]}/{b["rc"]}',
        'peak_gib': b['peak_gib'],
        'cotenants_alive_throughout': all(alive) and all(alive_end),
    }
    print(f'[gpu {gpu}] {row}', flush=True)
    with lock:
        rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--cells', type=Path, required=True)
    parser.add_argument('--alone', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--short', type=int, default=40)
    parser.add_argument('--cotenant-steps', type=int, default=5000)
    parser.add_argument('--settle', type=int, default=90)
    parser.add_argument('--extra', nargs='*', default=[])
    args = parser.parse_args()
    sweep.EXTRA.extend(args.extra)

    cells = sweep.read_cells(args.cells, dict(os.environ))
    devices = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d]
    if len(devices) < 2:
        raise SystemExit(f'need the 2 GPUs of the job, got {devices}')
    args.out.mkdir(parents=True, exist_ok=True)
    smi = subprocess.Popen(
        [
            'nvidia-smi',
            f'--id={",".join(devices[:2])}',
            '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu',
            '--format=csv,nounits',
            '-l',
            '10',
        ],
        stdout=open(args.out / 'nvidia_smi.csv', 'w'),
    )
    rows, lock = [], threading.Lock()

    def card_worker(card: int) -> None:
        for idx, (measured, cotenants) in enumerate(SCENARIOS[card]):
            scenario(
                devices[card], card, idx, measured, cotenants, cells, args, rows, lock
            )

    threads = [threading.Thread(target=card_worker, args=(c,)) for c in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    smi.terminate()

    alone = pd.read_csv(args.alone).set_index('cell')['s_per_step_drop']
    table = pd.DataFrame(rows).sort_values(['card', 'scenario'])
    table['s_per_step_alone'] = table['measured'].map(alone)
    table['slowdown'] = table['s_per_step_shared'] / table['s_per_step_alone']
    table.to_csv(args.out / 'results.csv', index=False)
    print(table.to_string())


if __name__ == '__main__':
    main()
