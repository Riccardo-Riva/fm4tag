"""Per-module timing profile of one training step (pretrain or finetune).

Runs a manual training loop (no Lightning Trainer) on a single device and
measures, with CUDA events:

* step segments — data fetch, host-to-device copy, forward+loss, backward,
  grad clip, optimizer step;
* per-module forward AND backward time, via hooks on the interesting modules
  (encoders, transformer blocks, RowMixer internals, attention/FFN, projector,
  reconstructors, aggregator, head, loss modules).

The model/data are assembled exactly like ``fm4tag.runner.run`` (same Hydra
config, same builders), so the numbers are representative of real training —
minus DDP gradient all-reduce and the cross-rank SupCon gather (single
process).

Usage (typically inside an sbatch wrapper — see submit_profile_timing.sh):

    python scripts/profile_timing.py --phase pretrain  [hydra overrides ...]
    python scripts/profile_timing.py --phase finetune datamodule.batch_size=512
    python scripts/profile_timing.py --phase pretrain --synthetic --device cpu

Results are printed as tables and saved as JSON next to --out.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate as hydra_instantiate

from fm4tag.utils.model_builders import build_aggregator, build_encoders, build_head

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'src' / 'fm4tag' / 'configs')

# Module classes worth timing individually (subsets of the components above —
# they overlap with their parents, see the report note).
_HOT_CLASSES = {
    'ColTransformer',
    'RowTransformer',
    'RowColTransformer',
    'RowMixer',
    'Attention',
    'RowAttention',
    'ChunkedRowAttention',
    'FeedForward',
    'simple_MLP',
    'sep_MLP',
    'Classifier_Transformer',
}


# ──────────────────────────────────────────────────────────────────────────────
# Timing machinery
# ──────────────────────────────────────────────────────────────────────────────


class _Timer:
    """CUDA-event (or wall-clock on CPU) interval timer."""

    def __init__(self, device: str) -> None:
        self.cuda = device.startswith('cuda')

    def start(self):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def stop(self, start):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return (start, ev)  # elapsed resolved after sync
        return (time.perf_counter() - start) * 1e3

    @staticmethod
    def elapsed_ms(pair) -> float:
        if isinstance(pair, tuple):
            s, e = pair
            return s.elapsed_time(e)
        return pair


class ModuleTimers:
    """Forward+backward CUDA-event hooks on selected named submodules."""

    def __init__(self, root: torch.nn.Module, device: str) -> None:
        self.timer = _Timer(device)
        self.pending_fwd: dict[str, list] = defaultdict(list)
        self.pending_bwd: dict[str, list] = defaultdict(list)
        self.fwd_ms: dict[str, list[float]] = defaultdict(list)
        self.bwd_ms: dict[str, list[float]] = defaultdict(list)
        self.calls: dict[str, int] = defaultdict(int)
        self.classes: dict[str, str] = {}
        self.handles = []
        self._starts: dict[int, list] = defaultdict(list)  # id(module) stack
        self._bwd_starts: dict[int, list] = defaultdict(list)

        for name, mod in root.named_modules():
            if not name:
                continue
            cls = type(mod).__name__
            if name.count('.') <= 1 or cls in _HOT_CLASSES:
                if cls in ('ModuleList', 'ModuleDict', 'Sequential', 'Compose'):
                    continue
                self.classes[name] = cls
                self._register(name, mod)

    def _register(self, name: str, mod: torch.nn.Module) -> None:
        def pre(m, args, kwargs=None, _n=name):
            self._starts[id(m)].append(self.timer.start())

        def post(m, args, out, _n=name):
            if self._starts[id(m)]:
                s = self._starts[id(m)].pop()
                self.pending_fwd[_n].append(self.timer.stop(s))
                self.calls[_n] += 1

        def bpre(m, grad_output, _n=name):
            self._bwd_starts[id(m)].append(self.timer.start())

        def bpost(m, grad_input, grad_output, _n=name):
            if self._bwd_starts[id(m)]:
                s = self._bwd_starts[id(m)].pop()
                self.pending_bwd[_n].append(self.timer.stop(s))

        self.handles += [
            mod.register_forward_pre_hook(pre),
            mod.register_forward_hook(post),
            mod.register_full_backward_pre_hook(bpre),
            mod.register_full_backward_hook(bpost),
        ]

    def flush_step(self) -> None:
        """Resolve pending events into per-step ms (call after a device sync)."""
        for name, pairs in self.pending_fwd.items():
            self.fwd_ms[name].append(sum(_Timer.elapsed_ms(p) for p in pairs))
        for name, pairs in self.pending_bwd.items():
            self.bwd_ms[name].append(sum(_Timer.elapsed_ms(p) for p in pairs))
        self.pending_fwd.clear()
        self.pending_bwd.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


# ──────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ──────────────────────────────────────────────────────────────────────────────


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def synthetic_batch(cfg, B: int) -> dict:
    """Build a batch with the shapes the datamodule would produce."""
    g = len(cfg.variables[cfg.global_object].inputs)
    batch: dict = {
        'global': torch.randn(B, g),
        'label': torch.randint(0, len(cfg.variables[cfg.global_object].unique_labels), (B,)),
        'constituents': {},
    }
    for obj in cfg.constituent_objects:
        v = cfg.variables[obj].inputs
        C = 40  # constituents per jet (dataset average is ~20 valid of 40 slots)
        n_cat, n_con = len(v.cat_classes), len(v.continuous)
        valid = torch.rand(B, C) < 0.5
        valid[:, 0] = True
        batch['constituents'][obj] = {
            'categorical': torch.stack(
                [torch.randint(0, len(c), (B, C)) for c in v.cat_classes.values()], dim=-1
            ),
            'continuous': torch.randn(B, C, n_con),
            'valid': valid,
        }
    return batch


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['pretrain', 'finetune'], required=True)
    ap.add_argument('--steps', type=int, default=30, help='measured steps')
    ap.add_argument('--warmup', type=int, default=10, help='untimed warmup steps')
    ap.add_argument('--data-batches', type=int, default=50, help='batches for the dataloader benchmark')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--synthetic', action='store_true', help='synthetic batches; skips the real dataloader')
    ap.add_argument('--out', default=None, help='output dir for the JSON report')
    ap.add_argument('overrides', nargs='*', help='hydra overrides, e.g. encoders=col_v0')
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name='default', overrides=[f'phase={args.phase}', *args.overrides])

    device = args.device
    torch.set_float32_matmul_precision(cfg.get('float32_matmul_precision', 'high'))
    B = cfg.datamodule.batch_size

    # ── Model, assembled like runner/run.py ─────────────────────────────
    encoders = build_encoders(cfg)
    aggregator = build_aggregator(cfg, encoders)
    views = [hydra_instantiate(v) for v in cfg.get('views', [])]
    if args.phase == 'pretrain':
        module = hydra_instantiate(
            cfg.pretrain, encoders=encoders, aggregator=aggregator, views=views, _convert_='all'
        )
    else:
        head = build_head(cfg, aggregator)
        n_classes = len(cfg.variables[cfg.global_object].unique_labels)
        module = hydra_instantiate(
            cfg.finetune, encoders=encoders, aggregator=aggregator, head=head,
            views=views, n_classes=n_classes, _convert_='all',
        )
    module = module.to(device).train()
    params = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    n_params = sum(p.numel() for p in module.parameters()) / 1e6
    print(f'phase={args.phase}  device={device}  batch_size={B}  params={n_params:.2f}M')
    print(f'overrides: {args.overrides or "(none)"}\n')

    # ── Dataloader benchmark (real data only) ───────────────────────────
    data_report = {}
    if args.synthetic:
        batch_source = lambda: synthetic_batch(cfg, B)  # noqa: E731
        print('[data] synthetic batches (dataloader benchmark skipped)\n')
    else:
        dm = hydra_instantiate(
            cfg.datamodule, _convert_='all',
            train_dataset_path=(
                cfg.pretrain_dataset_path if args.phase == 'pretrain' else cfg.train_dataset_path
            ),
        )
        dm.setup('fit')
        loader = dm.train_dataloader()
        it = iter(loader)
        t0 = time.perf_counter()
        first = next(it)
        cold = (time.perf_counter() - t0) * 1e3
        fetch_ms = []
        for _ in range(args.data_batches):
            t0 = time.perf_counter()
            batch = next(it)
            fetch_ms.append((time.perf_counter() - t0) * 1e3)
        data_report = {
            'cold_first_batch_ms': cold,
            'steady_mean_ms': statistics.mean(fetch_ms),
            'steady_stdev_ms': statistics.stdev(fetch_ms) if len(fetch_ms) > 1 else 0.0,
            'steady_max_ms': max(fetch_ms),
            'batches_per_s': 1e3 / statistics.mean(fetch_ms),
        }
        print(f'[data] cold first batch : {cold:9.1f} ms   (worker startup)')
        print(f'[data] steady fetch     : {data_report["steady_mean_ms"]:9.1f} ms/batch '
              f'± {data_report["steady_stdev_ms"]:.1f}  (max {data_report["steady_max_ms"]:.1f}, '
              f'{data_report["batches_per_s"]:.1f} batches/s)\n')
        cached = [first, batch]
        batch_source = lambda: cached[torch.randint(0, len(cached), (1,)).item()]  # noqa: E731

    # ── Training-step timing ─────────────────────────────────────────────
    timer = _Timer(device)
    autocast = (
        torch.autocast('cuda', dtype=torch.bfloat16)
        if device.startswith('cuda') and str(cfg.trainer.get('precision', '')).startswith('bf16')
        else contextlib.nullcontext()
    )
    clip_val = cfg.trainer.get('gradient_clip_val', None)

    def one_step(record: ModuleTimers | None, segs: dict | None) -> None:
        marks = {}

        t = time.perf_counter()
        batch = batch_source()
        marks['data_fetch'] = (time.perf_counter() - t) * 1e3

        s = timer.start(); batch = to_device(batch, device); marks['h2d_copy'] = timer.stop(s)
        s = timer.start()
        with autocast:
            out = module._compute_loss(batch)
            loss = out[0]
        marks['forward+loss'] = timer.stop(s)
        s = timer.start(); loss.backward(); marks['backward'] = timer.stop(s)
        if clip_val:
            s = timer.start()
            torch.nn.utils.clip_grad_norm_(params, clip_val)
            marks['grad_clip'] = timer.stop(s)
        s = timer.start(); optimizer.step(); marks['optimizer_step'] = timer.stop(s)
        optimizer.zero_grad(set_to_none=True)

        if device.startswith('cuda'):
            torch.cuda.synchronize()
        if record is not None:
            record.flush_step()
        if segs is not None:
            for k, v in marks.items():
                segs[k].append(v if isinstance(v, float) else _Timer.elapsed_ms(v))

    print(f'[step] warmup x{args.warmup} ...')
    for _ in range(args.warmup):
        one_step(None, None)

    print(f'[step] measuring x{args.steps} ...\n')
    mt = ModuleTimers(module, device)
    segs: dict[str, list[float]] = defaultdict(list)
    for _ in range(args.steps):
        one_step(mt, segs)
    mt.remove()

    # ── Report ───────────────────────────────────────────────────────────
    seg_mean = {k: statistics.mean(v) for k, v in segs.items()}
    total = sum(seg_mean.values())
    print(f'── Step segments (mean over {args.steps} steps; total {total:.1f} ms/step) ' + '─' * 20)
    for k, v in sorted(seg_mean.items(), key=lambda kv: -kv[1]):
        sd = statistics.stdev(segs[k]) if len(segs[k]) > 1 else 0.0
        print(f'  {k:16s} {v:9.2f} ms  ± {sd:7.2f}   {100 * v / total:5.1f} %')

    fwd_bwd = seg_mean.get('forward+loss', 0.0) + seg_mean.get('backward', 0.0)
    rows = []
    for name in mt.classes:
        f = statistics.mean(mt.fwd_ms[name]) if mt.fwd_ms.get(name) else 0.0
        b = statistics.mean(mt.bwd_ms[name]) if mt.bwd_ms.get(name) else 0.0
        if f + b < 0.005:
            continue
        rows.append((name, mt.classes[name], mt.calls[name] // args.steps, f, b))
    rows.sort(key=lambda r: -(r[3] + r[4]))

    print('\n── Per-module time (ms/step; nested modules overlap their parents) ' + '─' * 10)
    print(f'  {"module":52s} {"class":22s} {"calls":>5s} {"fwd":>9s} {"bwd":>9s} {"f+b":>9s} {"%":>6s}')
    max_rows = 35
    for name, cls, calls, f, b in rows[:max_rows]:
        print(f'  {name:52s} {cls:22s} {calls:5d} {f:9.2f} {b:9.2f} {f + b:9.2f} '
              f'{100 * (f + b) / fwd_bwd if fwd_bwd else 0:5.1f} %')
    if len(rows) > max_rows:
        print(f'  ... {len(rows) - max_rows} smaller modules omitted (all in the JSON report)')

    print('\nNotes: single process — DDP all-reduce and cross-rank SupCon gather are NOT '
          'included; nested rows (deep module names) are subsets of their parents.')

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        report = {
            'phase': args.phase, 'overrides': args.overrides, 'batch_size': B,
            'params_M': n_params, 'dataloader': data_report,
            'segments_ms': seg_mean,
            'modules': [
                {'name': n, 'class': c, 'calls_per_step': k, 'fwd_ms': f, 'bwd_ms': b}
                for n, c, k, f, b in rows
            ],
        }
        path = out / f'profile_{args.phase}.json'
        path.write_text(json.dumps(report, indent=2))
        print(f'\nreport saved: {path}')


if __name__ == '__main__':
    main()
