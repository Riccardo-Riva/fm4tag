"""Plot uniformity/effective-rank evolution from a pre-computed YAML summary.

The YAML is expected to have epoch numbers as top-level keys, each containing
``jets`` and ``tracks`` sub-dicts as produced by ``plot_uniformity.py``.

Four plots are saved:
  - jets_uniformity.png           — jets.uniformity vs epoch
  - jets_effective_rank.png       — jets.effective_rank vs epoch
  - tracks_uniformity.png         — tracks.track_uniformity & tracks.jet_uniformity vs epoch
  - tracks_effective_rank.png     — tracks.track_effective_rank & tracks.jet_effective_rank vs epoch

Usage::

    python scripts/plot_uniformity_yaml.py \\
        --file slurm/pretraining/run_20260305_121539/outputs/model_0_colrow/version_0/uniformity_evolution.yaml \\
        --outdir plots/uniformity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(yaml_path: Path) -> tuple[list[int], dict[str, list[float]]]:
    """Load YAML and return sorted epochs + metric time-series.

    Returns
    -------
    epochs : list[int]
    series : dict mapping metric_key -> list of float values (one per epoch)
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    sorted_epochs = sorted(data.keys(), key=int)

    # Flatten all leaf metrics into a single dict of lists.
    series: dict[str, list[float]] = {}
    for epoch_key in sorted_epochs:
        entry = data[epoch_key]
        for obj, metrics in entry.items():
            for k, v in metrics.items():
                if k.startswith('n_'):
                    continue
                full_key = f'{obj}.{k}'
                series.setdefault(full_key, []).append(float(v))

    return [int(e) for e in sorted_epochs], series


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------


def _single_plot(
    epochs: list[int],
    values: list[float],
    title: str,
    ylabel: str,
    color: str = 'tab:blue',
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, values, marker='o', markersize=4, linewidth=1.5, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if 'uniformity' in ylabel.lower():
        best_idx = int(min(range(len(values)), key=lambda i: values[i]))
        label = 'most uniform'
    else:
        best_idx = int(max(range(len(values)), key=lambda i: values[i]))
        label = 'best'
    ax.axvline(
        epochs[best_idx],
        color='red',
        linestyle='--',
        alpha=0.6,
        label=f'{label}: epoch {epochs[best_idx]} ({values[best_idx]:.3f})',
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _paired_plot(
    epochs: list[int],
    values_a: list[float],
    label_a: str,
    values_b: list[float],
    label_b: str,
    title: str,
    ylabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        epochs,
        values_a,
        marker='o',
        markersize=4,
        linewidth=1.5,
        color='tab:blue',
        label=label_a,
    )
    ax.plot(
        epochs,
        values_b,
        marker='s',
        markersize=4,
        linewidth=1.5,
        color='tab:orange',
        label=label_b,
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot uniformity/effective-rank from a pre-computed YAML summary.'
    )
    parser.add_argument('--file', type=Path, required=True, help='Path to uniformity_evolution.yaml')
    parser.add_argument('--outdir', type=Path, required=True, help='Directory to store the output plots')
    args = parser.parse_args()

    yaml_file: Path = args.file.resolve()
    out_dir: Path = args.outdir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs, series = _load(yaml_file)

    # ── Jets uniformity ───────────────────────────────────────────────────────
    if 'jets.uniformity' in series:
        fig = _single_plot(
            epochs,
            series['jets.uniformity'],
            title='Jets — uniformity',
            ylabel='Uniformity',
            color='tab:blue',
        )
        _save(fig, out_dir, 'jets_uniformity.png')

    # ── Jets effective rank ───────────────────────────────────────────────────
    if 'jets.effective_rank' in series:
        fig = _single_plot(
            epochs,
            series['jets.effective_rank'],
            title='Jets — effective rank',
            ylabel='Effective rank',
            color='tab:green',
        )
        _save(fig, out_dir, 'jets_effective_rank.png')

    # ── Tracks uniformity (paired: track-level + jet-level) ───────────────────
    if 'tracks.track_uniformity' in series and 'tracks.jet_uniformity' in series:
        fig = _paired_plot(
            epochs,
            series['tracks.track_uniformity'],
            label_a='track uniformity',
            values_b=series['tracks.jet_uniformity'],
            label_b='track uniformity (mean over jet)',
            title='Tracks — uniformity',
            ylabel='Uniformity',
        )
        _save(fig, out_dir, 'tracks_uniformity.png')

    # ── Tracks effective rank (paired) ────────────────────────────────────────
    if 'tracks.track_effective_rank' in series and 'tracks.jet_effective_rank' in series:
        fig = _paired_plot(
            epochs,
            series['tracks.track_effective_rank'],
            label_a='track effective rank',
            values_b=series['tracks.jet_effective_rank'],
            label_b='track effective rank (mean over jet)',
            title='Tracks — effective rank',
            ylabel='Effective rank',
        )
        _save(fig, out_dir, 'tracks_effective_rank.png')


if __name__ == '__main__':
    main()
