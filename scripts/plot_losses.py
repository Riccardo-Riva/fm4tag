"""Plot train and val loss vs epoch from a Lightning metrics.csv file.

Usage::

    python scripts/plot_losses.py \\
        --file slurm/pretraining/run_20260305_121539/outputs/model_0_colrow/version_0/metrics.csv \\
        --outdir plots/losses
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot train/val loss vs epoch from a Lightning metrics.csv.'
    )
    parser.add_argument('--file', type=Path, required=True, help='Path to metrics.csv')
    parser.add_argument('--outdir', type=Path, required=True, help='Directory to store the output plot')
    args = parser.parse_args()

    out_dir: Path = args.outdir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.file)

    train = df[df['train_loss_epoch'].notna()][['epoch', 'train_loss_epoch']].copy()
    val = df[df['val_loss'].notna()][['epoch', 'val_loss']].copy()

    train = train.groupby('epoch', as_index=False)['train_loss_epoch'].mean()
    val = val.groupby('epoch', as_index=False)['val_loss'].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train['epoch'], train['train_loss_epoch'],
            marker='o', markersize=3, linewidth=1.5, label='train loss')
    ax.plot(val['epoch'], val['val_loss'],
            marker='s', markersize=3, linewidth=1.5, label='val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train / Val loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / 'losses.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
