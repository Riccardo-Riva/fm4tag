"""Plot uniformity (and effective rank) vs training epoch for a pretrain run.

Given a Lightning ``version_N`` directory, this script:

1. Finds all per-epoch checkpoints (``epoch=NNN-*.ckpt``, excluding ``last.ckpt``).
2. Auto-discovers the Hydra config saved alongside the run (searches for
   ``**/.hydra/config.yaml`` relative to the run root).
3. Calls :func:`fm4tag.eval_encoder.evaluate` for every checkpoint.
4. Plots the metrics vs epoch and saves the figure.

Usage::

    # Minimal — point at a version directory:
    python scripts/plot_uniformity.py \\
        slurm/pretraining/run_20260305_121539/outputs/model_0_colrow/version_0

    # Explicit config (useful when auto-discovery fails):
    python scripts/plot_uniformity.py \\
        slurm/pretraining/run_20260305_121539/outputs/model_0_colrow/version_0 \\
        --config slurm/pretraining/run_20260305_121539/outputs/2026-03-05/12-17-44/.hydra/config.yaml

    # Custom output path and evaluation file:
    python scripts/plot_uniformity.py <version_dir> \\
        --eval-file /data/val.h5 \\
        --output my_plot.png

    # Only evaluate every N-th checkpoint (faster):
    python scripts/plot_uniformity.py <version_dir> --stride 5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

# Allow running without installing the package (add src/ to path).
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / 'src'))

from fm4tag.eval_encoder import evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_config(version_dir: Path) -> Path | None:
    """Search for .hydra/config.yaml starting from the run root.

    The run root is assumed to be the grandparent of the experiment directory
    (i.e. ``version_dir/../../../``).  Walks up until it finds any
    ``.hydra/config.yaml``.
    """
    # Walk up from version_dir looking for a .hydra/config.yaml sibling.
    candidate = version_dir
    for _ in range(6):
        candidate = candidate.parent
        matches = sorted(candidate.glob('**/.hydra/config.yaml'))
        if matches:
            return matches[0]
    return None


def _parse_epoch(ckpt_path: Path) -> int | None:
    """Extract the epoch number from a checkpoint filename."""
    m = re.search(r'epoch=(\d+)', ckpt_path.stem)
    return int(m.group(1)) if m else None


def _parse_val_loss(ckpt_path: Path) -> float | None:
    """Extract the val_loss from a checkpoint filename, if present."""
    m = re.search(r'val_loss=([\d.]+)', ckpt_path.stem)
    return float(m.group(1)) if m else None


def _sorted_epoch_checkpoints(ckpts_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(epoch, path)`` pairs sorted by epoch, excluding last.ckpt."""
    result = []
    for p in ckpts_dir.glob('epoch=*.ckpt'):
        epoch = _parse_epoch(p)
        if epoch is not None:
            result.append((epoch, p))
    return sorted(result, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot(
    epochs: list[int],
    metrics_per_epoch: list[dict],
    val_losses: list[float | None],
    out_path: Path,
) -> None:
    """Render and save the uniformity / effective-rank figure."""
    if not metrics_per_epoch:
        print('No metrics to plot.')
        return

    # Collect all (obj_name, metric_key) combinations present in the data.
    obj_metric_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in metrics_per_epoch:
        for obj, vals in m.items():
            for k in vals:
                if k.startswith('n_'):
                    continue
                if (obj, k) not in seen:
                    obj_metric_pairs.append((obj, k))
                    seen.add((obj, k))

    has_val_loss = any(v is not None for v in val_losses)
    n_panels = len(obj_metric_pairs) + (1 if has_val_loss else 0)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    ax_flat = [ax for row in axes for ax in row]

    for idx, (obj, key) in enumerate(obj_metric_pairs):
        ax = ax_flat[idx]
        values = [
            m[obj][key] for m in metrics_per_epoch if obj in m and key in m[obj]
        ]
        ep = [
            epochs[i]
            for i, m in enumerate(metrics_per_epoch)
            if obj in m and key in m[obj]
        ]
        ax.plot(ep, values, marker='o', markersize=4, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_title(f'{obj} — {key}')
        ax.grid(True, alpha=0.3)

        # Annotate best value.
        if 'uniformity' in key:
            best_idx = int(min(range(len(values)), key=lambda i: values[i]))
            label = 'most uniform'
        else:
            best_idx = int(max(range(len(values)), key=lambda i: values[i]))
            label = 'best'
        ax.axvline(ep[best_idx], color='red', linestyle='--', alpha=0.5,
                   label=f'{label}: epoch {ep[best_idx]} ({values[best_idx]:.3f})')
        ax.legend(fontsize=8)

    if has_val_loss:
        ax = ax_flat[len(obj_metric_pairs)]
        vl = [(e, v) for e, v in zip(epochs, val_losses) if v is not None]
        ax.plot([x[0] for x in vl], [x[1] for x in vl],
                marker='o', markersize=4, linewidth=1.5, color='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_title('val_loss')
        ax.grid(True, alpha=0.3)

    # Hide unused panels.
    for ax in ax_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(str(out_path.parent), fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot uniformity vs epoch for a pretrain run.'
    )
    parser.add_argument(
        'version_dir',
        type=Path,
        help='Path to the Lightning version directory '
             '(contains checkpoints/ and metrics.csv).',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to the Hydra config.yaml. Auto-discovered if not provided.',
    )
    parser.add_argument(
        '--eval-file',
        type=str,
        default=None,
        help='HDF5 file to evaluate on. Falls back to pretrain_val_file / '
             'test_file from config.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output figure path. Default: <version_dir>/uniformity_evolution.png',
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Evaluate every N-th checkpoint (default: 1 = all).',
    )
    parser.add_argument(
        '--max-track-samples',
        type=int,
        default=20_000,
        help='Max track embeddings per checkpoint (default: 20000).',
    )
    parser.add_argument(
        '--max-jet-samples',
        type=int,
        default=5_000,
        help='Max jet embeddings per checkpoint (default: 5000).',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Torch device (e.g. cuda, cpu). Auto-detected if not set.',
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Save / reuse per-checkpoint JSON results to skip recomputation.',
    )
    args = parser.parse_args()

    version_dir: Path = args.version_dir.resolve()
    ckpts_dir = version_dir / 'checkpoints'
    if not ckpts_dir.is_dir():
        sys.exit(f'Error: no checkpoints/ directory found in {version_dir}')

    out_path: Path = args.output or (version_dir / 'uniformity_evolution.png')

    # ── Config ────────────────────────────────────────────────────────────────
    config_path: Path | None = args.config
    if config_path is None:
        config_path = _find_config(version_dir)
    if config_path is None:
        sys.exit(
            'Error: could not auto-discover config.yaml. '
            'Pass --config explicitly.'
        )
    print(f'Using config: {config_path}')
    cfg = OmegaConf.load(config_path)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    all_ckpts = _sorted_epoch_checkpoints(ckpts_dir)
    if not all_ckpts:
        sys.exit(f'Error: no epoch checkpoints found in {ckpts_dir}')

    selected = all_ckpts[:: args.stride]
    print(f'Found {len(all_ckpts)} checkpoints, evaluating {len(selected)}.')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    epochs: list[int] = []
    metrics_per_epoch: list[dict] = []
    val_losses: list[float | None] = []

    cache_path = version_dir / 'uniformity_cache.yaml'
    cache: dict[str, dict] = {}
    if args.cache and cache_path.exists():
        cache = OmegaConf.to_container(OmegaConf.load(cache_path), resolve=True)
        print(f'Loaded cache from {cache_path}')

    for epoch, ckpt_path in selected:
        cache_key = ckpt_path.name
        if args.cache and cache_key in cache:
            results = cache[cache_key]
            print(f'  epoch {epoch:3d} — (from cache)')
        else:
            print(f'  epoch {epoch:3d} — {ckpt_path.name}', end=' ... ', flush=True)
            results = evaluate(
                cfg,
                ckpt_path=str(ckpt_path),
                eval_file=args.eval_file,
                max_track_samples=args.max_track_samples,
                max_jet_samples=args.max_jet_samples,
                device=args.device,
            )
            print('done')
            if args.cache:
                cache[cache_key] = results
                OmegaConf.save(OmegaConf.create(cache), cache_path)

        epochs.append(epoch)
        metrics_per_epoch.append(results)
        val_losses.append(_parse_val_loss(ckpt_path))

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(epochs, metrics_per_epoch, val_losses, out_path)

    # Save final YAML summary.
    summary_path = out_path.with_suffix('.yaml')
    summary = {str(e): m for e, m in zip(epochs, metrics_per_epoch)}
    OmegaConf.save(OmegaConf.create(summary), summary_path)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
