"""Memory of one pretraining step on CPU, per cell of a cells file (dev test, 2026-11-26).

For each cell: compose it with a base config, build the pretraining module as the runner does, take the first
validation batch of the real data (same jets for every cell), and run one forward + backward of the training loss on
CPU in float32.  Each cell runs in its own process.  A thread samples the resident memory every 20 ms; the step's
memory is the highest sample minus the level just before the forward.  One JSON line per cell goes to stdout and to
``--out``.

The CPU number is not the GPU number (float32 instead of bf16-mixed, other attention kernels).  It is used as a ratio:
the GPU peak of a new cell is estimated from a calibration cell whose GPU peak was measured (README).

    .venv/bin/python mem_estimate_cpu.py --cells <cells file> --config-name <base> --batch 2560 --out <jsonl>
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / 'src' / 'fm4tag' / 'configs'
# In-process loading and no pinned memory: the measured process holds only the model and one batch.
LOADER_OVERRIDES = [
    'datamodule.num_workers=0',
    'datamodule.persistent_workers=false',
    'datamodule.pin_memory=false',
]


def read_cells(path: str) -> list[tuple[str, list[str]]]:
    """Return (name, hydra overrides) per cell; launcher tokens (``@...``) are dropped."""
    cells = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, *tokens = line.split()
        cells.append((name, [t for t in tokens if not t.startswith('@')]))
    return cells


def rss_gib() -> float:
    """Current resident set size of this process in GiB."""
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1024**3


class PeakSampler(threading.Thread):
    """Samples the resident memory until stopped and keeps the highest value."""

    def __init__(self, interval: float = 0.02) -> None:
        super().__init__(daemon=True)
        self.interval, self.peak, self._stop_event = (
            interval,
            rss_gib(),
            threading.Event(),
        )

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak = max(self.peak, rss_gib())
            time.sleep(self.interval)

    def stop(self) -> float:
        self._stop_event.set()
        self.join()
        return max(self.peak, rss_gib())


def measure(config_name: str, name: str, overrides: list[str]) -> dict:
    """Build the cell's pretraining module, run one loss + backward on a real batch, return the memory numbers."""
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from fm4tag.utils.model_builders import build_aggregator, build_encoders

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides + LOADER_OVERRIDES)
    torch.manual_seed(cfg.seed)

    dm = instantiate(
        cfg.datamodule, _convert_='all', train_dataset_path=cfg.pretrain_dataset_path
    )
    dm.setup('fit')
    batch = next(iter(dm.val_dataloader()))
    encoders = build_encoders(cfg)
    aggregator = build_aggregator(cfg, encoders)
    views = [instantiate(v) for v in cfg.get('views', [])]
    module = instantiate(
        cfg.pretrain,
        encoders=encoders,
        aggregator=aggregator,
        views=views,
        _convert_='all',
        **OmegaConf.to_container(cfg.optimizer, resolve=True),
    )
    module.train()

    before = rss_gib()
    sampler = PeakSampler()
    sampler.start()
    t0 = time.time()
    loss, _ = module._compute_loss(batch)
    loss.backward()
    seconds = time.time() - t0
    peak = sampler.stop()
    if not torch.isfinite(loss):
        raise FloatingPointError(f'{name}: loss is {loss.item()}')

    valid = batch['constituents']['tracks']['valid']
    return {
        'cell': name,
        'config_name': config_name,
        'overrides': overrides,
        'batch': int(valid.shape[0]),
        'valid_tracks': int(valid.sum()),
        'trainable_M': round(
            sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6, 3
        ),
        'rss_before_GiB': round(before, 3),
        'step_GiB': round(peak - before, 3),
        'seconds': round(seconds, 1),
        'loss': round(loss.item(), 4),
        'threads': torch.get_num_threads(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--cells', required=True, help='cells file')
    p.add_argument(
        '--config-name',
        required=True,
        help='base config (default = model B, saint_small_RC, ...)',
    )
    p.add_argument(
        '--batch',
        type=int,
        required=True,
        help='jets per batch (a multiple of chunk_size)',
    )
    p.add_argument('--only', nargs='*', help='run only these cells (default: all)')
    p.add_argument('--out', help='append one JSON line per cell here')
    p.add_argument(
        '--child', help=argparse.SUPPRESS
    )  # internal: measure this one cell in this process
    args = p.parse_args()

    cells = dict(read_cells(args.cells))
    if args.child:
        overrides = cells[args.child] + [f'datamodule.batch_size={args.batch}']
        print(
            'RESULT ' + json.dumps(measure(args.config_name, args.child, overrides)),
            flush=True,
        )
        return

    failed = []
    for name in args.only or list(cells):
        cmd = [
            sys.executable,
            __file__,
            '--cells',
            args.cells,
            '--config-name',
            args.config_name,
            '--batch',
            str(args.batch),
            '--child',
            name,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines = [
            ln[len('RESULT ') :]
            for ln in proc.stdout.splitlines()
            if ln.startswith('RESULT ')
        ]
        if proc.returncode or not lines:
            failed.append(name)
            print(
                f'FAILED {name} (exit {proc.returncode}):\n{proc.stderr[-3000:]}',
                file=sys.stderr,
                flush=True,
            )
            continue
        print(lines[0], flush=True)
        if args.out:
            with open(args.out, 'a') as f:
                f.write(lines[0] + '\n')
    if failed:
        sys.exit(f'failed cells: {failed}')


if __name__ == '__main__':
    main()
