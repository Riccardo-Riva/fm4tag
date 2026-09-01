"""Permutation feature importance for trained fm4tag classifiers.

Breiman's MDA, adapted to a padded track *set*: for every feature (or group of
features) the values are shuffled among jets and the resulting degradation of
the classification loss is the importance.  Nothing is retrained — this
measures what each *already trained* model relies on.

Designed for cross-model comparison (e.g. one architecture trained at several
jet-contrastive weights λ), which drives three deliberate choices:

**Metric is ``val/head/loss_ce``, never ``val/loss``.**  The training objective
is ``CE + λ·contrastive``, whose scale varies with λ (0.78 → 2.79 over the
λ scan), so ΔLoss would measure λ rather than feature reliance.  The bare CE
head loss is identically defined for every model.  Importance is reported both
as raw ΔCE and normalised to sum 1 per model, so the *shape* of the profile is
comparable even when baselines differ.

**Permutation is pooled over VALID tracks only.**  Constituents arrive as a
``(B, C, F)`` grid with a ``valid`` mask; on the val file 81.8 % of that grid
is padding (mean 7.3 valid tracks per jet out of C = 40) and padded slots hold
NaN.  Shuffling the raw grid along the batch axis would therefore inject NaN
into real slots and dilute every effect ~5.5×.  Instead all valid tracks in the
batch are gathered into one pool (~7.5 k tracks at B = 1024), the feature is
permuted within that pool, and the values are scattered back — which preserves
the feature's marginal distribution and the valid mask exactly, while
destroying both the jet↔track association and the within-jet correlation with
the other features.

**Common random numbers across models.**  The permutation seed depends only on
``(base_seed, repeat, batch_index)``, not on the model or the variant.  Every
model therefore sees *the same* shuffles, so model-to-model differences are
paired and the shared permutation noise cancels — essential here, because the
CE spread across the λ grid is only ~0.003.

Single-feature importance systematically under-states correlated blocks (the
model recovers ``d0`` from ``lifetimeSignedD0Significance`` and friends), so
the config also takes ``groups``: named feature sets permuted jointly with a
*shared* permutation, which keeps intra-group correlation intact and breaks
only the group↔rest association.

Outputs, under ``--output``:
  - ``importance_raw.csv``     — one row per (model, variant, repeat)
  - ``importance_summary.csv`` — mean ± std ΔCE per (model, variant), normalised
  - ``heatmap.png``            — variant × model grid of normalised importance
  - ``profile.png``            — top-N features vs the model axis
  - ``drift.png``              — Spearman correlation of each model's profile
                                 with the first model's

Usage::

    python scripts/permutation_importance.py --config scripts/configs/permutation_importance_lambda_scan.yaml
    python scripts/permutation_importance.py --config ... --max-jets 20000 --repeats 2   # smoke test
    python scripts/permutation_importance.py --config ... --output plots/my_run
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import roc_comparison as rc  # noqa: E402  -- checkpoint resolution / model rebuild reuse

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from fm4tag.datasets.datasets import DatasetCatCon  # noqa: E402
from fm4tag.datasets.loader import make_batch_dataloader  # noqa: E402

BASELINE = '__baseline__'


# ---------------------------------------------------------------------------
# Variants: what gets permuted together
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    """One permutation experiment.

    ``con``/``cat`` are column indices into the constituent object's continuous
    and categorical tensors; ``glob`` indexes the global (jet) feature tensor.
    A variant holding several indices permutes them with a *shared*
    permutation, which is what keeps intra-group correlations alive.
    """

    name: str
    kind: str  # 'single' | 'group' | 'global'
    con: list[int] = field(default_factory=list)
    cat: list[int] = field(default_factory=list)
    glob: list[int] = field(default_factory=list)
    members: list[str] = field(default_factory=list)


def build_variants(
    run_cfg: DictConfig, obj: str, groups: dict[str, list[str]], include_global: bool
) -> list[Variant]:
    """Single-feature variants for every track feature, plus the named groups."""
    inputs = run_cfg.variables[obj].inputs
    con_names = list(inputs.continuous)
    cat_names = list(inputs.categorical)
    con_at = {n: i for i, n in enumerate(con_names)}
    cat_at = {n: i for i, n in enumerate(cat_names)}

    variants = [Variant(n, 'single', con=[con_at[n]], members=[n]) for n in con_names]
    variants += [Variant(n, 'single', cat=[cat_at[n]], members=[n]) for n in cat_names]

    for group_name, members in (groups or {}).items():
        unknown = [m for m in members if m not in con_at and m not in cat_at]
        if unknown:
            raise KeyError(
                f'group {group_name!r} lists unknown {obj} features: {unknown}'
            )
        variants.append(
            Variant(
                group_name,
                'group',
                con=[con_at[m] for m in members if m in con_at],
                cat=[cat_at[m] for m in members if m in cat_at],
                members=list(members),
            )
        )

    if include_global:
        g_obj = run_cfg.global_object
        for i, name in enumerate(run_cfg.variables[g_obj].inputs):
            variants.append(
                Variant(f'{g_obj}:{name}', 'global', glob=[i], members=[name])
            )

    return variants


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------


def batch_seed(base: int, repeat: int, batch_idx: int) -> int:
    """Seed that depends on (repeat, batch) but NOT on the model or variant.

    Sharing it across models makes the comparison paired; sharing it across
    variants additionally pairs feature-vs-feature within a model.  Neither
    couples the variants to each other — each is applied to a fresh copy of the
    pristine batch.
    """
    return (base + 1_000_003 * repeat + 10_007 * batch_idx) % (2**63 - 1)


def permuted_batch(
    batch: dict,
    variant: Variant,
    obj: str,
    track_perm: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    global_perm: torch.Tensor,
) -> dict:
    """Shallow-copy ``batch`` with ``variant``'s features permuted.

    Only the tensors a variant actually touches are cloned (≤3 MB at B = 1024),
    so the pristine batch is reused untouched by every other variant.
    """
    out = {'label': batch['label'], 'global': batch['global']}
    out['constituents'] = {
        name: dict(fields) for name, fields in batch['constituents'].items()
    }

    if variant.con or variant.cat:
        src_rows, src_cols = rows[track_perm], cols[track_perm]
        fields = out['constituents'][obj]
        for key, columns in (('continuous', variant.con), ('categorical', variant.cat)):
            if not columns:
                continue
            tensor = fields[key].clone()
            for f in columns:
                # RHS advanced indexing materialises a copy before the scatter,
                # so reading and writing the same tensor is safe here.
                tensor[rows, cols, f] = tensor[src_rows, src_cols, f]
            fields[key] = tensor

    if variant.glob:
        tensor = out['global'].clone()
        for f in variant.glob:
            tensor[:, f] = tensor[global_perm, f]
        out['global'] = tensor

    return out


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    module: torch.nn.Module,
    run_cfg: DictConfig,
    eval_file: Path,
    variants: list[Variant],
    obj: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    n_jets: int | None,
    repeats: int,
    base_seed: int,
    desc: str,
) -> tuple[dict[tuple[str, int], float], float, int]:
    """Mean CE per (variant, repeat), plus the unpermuted baseline CE.

    One pass over the data: every batch is read once and all
    ``1 + n_variants * repeats`` forward passes run on it, because the
    compressed HDF5 read — not the forward pass — is the expensive part.
    """
    dataset = DatasetCatCon(
        file_path=str(eval_file),
        variables=OmegaConf.to_container(run_cfg.variables, resolve=True),
        global_object=run_cfg.global_object,
        constituent_objects=list(run_cfg.constituent_objects),
        norm_dict=rc._load_yaml(run_cfg.get('norm_dict_path')),
        class_dict=rc._load_yaml(run_cfg.get('class_dict_path')),
    )
    loader = make_batch_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2,
        max_rows=n_jets,
    )

    module.to(device)
    weights = getattr(module, 'class_weights', None)

    ce_sum: dict[tuple[str, int], float] = {
        (v.name, r): 0.0 for v in variants for r in range(repeats)
    }
    base_sum = 0.0
    n_total = 0

    def cross_entropy(b: dict) -> torch.Tensor:
        logits = module.head(module(b)['aggregator'])
        return F.cross_entropy(logits, b['label'], weight=weights, reduction='sum')

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
            batch_dev = {
                'label': batch['label'].to(device, non_blocking=True),
                'global': batch['global'].to(device, non_blocking=True),
                'constituents': {
                    name: {
                        k: v.to(device, non_blocking=True) for k, v in fields.items()
                    }
                    for name, fields in batch['constituents'].items()
                },
            }
            n_batch = batch_dev['label'].shape[0]
            n_total += n_batch
            base_sum += cross_entropy(batch_dev).item()

            rows, cols = batch_dev['constituents'][obj]['valid'].nonzero(as_tuple=True)
            n_valid = rows.numel()

            for repeat in range(repeats):
                generator = torch.Generator(device=device)
                generator.manual_seed(batch_seed(base_seed, repeat, batch_idx))
                track_perm = torch.randperm(n_valid, generator=generator, device=device)
                global_perm = torch.randperm(
                    n_batch, generator=generator, device=device
                )

                for variant in variants:
                    permuted = permuted_batch(
                        batch_dev, variant, obj, track_perm, rows, cols, global_perm
                    )
                    ce_sum[(variant.name, repeat)] += cross_entropy(permuted).item()

    module.cpu()
    return (
        {key: total / n_total for key, total in ce_sum.items()},
        base_sum / n_total,
        n_total,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_heatmap(summary: pd.DataFrame, order: list[str], out: Path) -> None:
    """Normalised importance, variants (rows) × models (columns)."""
    singles = summary[summary['kind'] == 'single']
    table = singles.pivot(index='variant', columns='model', values='fraction')
    table = table[order]
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(1.05 * len(order) + 4.5, 0.34 * len(table) + 2.0))
    image = ax.imshow(table.values, aspect='auto', cmap='magma_r')
    ax.set_xticks(range(len(order)), order, rotation=45, ha='right')
    ax.set_yticks(range(len(table)), table.index, fontsize=8)
    ax.set_title('Permutation importance — fraction of total ΔCE')

    hi = np.nanmax(table.values)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f'{100 * value:.1f}',
                    ha='center',
                    va='center',
                    fontsize=6.5,
                    color='white' if value > 0.55 * hi else 'black',
                )
    fig.colorbar(image, ax=ax, label='fraction of summed single-feature ΔCE')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_profile(summary: pd.DataFrame, order: list[str], out: Path, top: int) -> None:
    """Top-N single features and every group, as lines across the model axis."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, kind, title in (
        (axes[0], 'single', f'Top-{top} single features'),
        (axes[1], 'group', 'Feature groups'),
    ):
        subset = summary[summary['kind'] == kind]
        if subset.empty:
            ax.set_visible(False)
            continue
        ranked = (
            subset.groupby('variant')['delta_ce_mean']
            .mean()
            .sort_values(ascending=False)
        )
        names = list(ranked.index[:top] if kind == 'single' else ranked.index)
        x = np.arange(len(order))
        for name in names:
            rows = subset[subset['variant'] == name].set_index('model').loc[order]
            ax.errorbar(
                x,
                rows['delta_ce_mean'],
                yerr=rows['delta_ce_std'],
                marker='o',
                ms=4,
                capsize=2,
                lw=1.4,
                label=name,
            )
        ax.set_xticks(x, order, rotation=45, ha='right')
        ax.set_ylabel('ΔCE (permuted − baseline)')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=1)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_drift(summary: pd.DataFrame, order: list[str], out: Path) -> pd.DataFrame:
    """Spearman correlation of each model's single-feature profile with the first."""
    singles = summary[summary['kind'] == 'single']
    table = singles.pivot(index='variant', columns='model', values='delta_ce_mean')[
        order
    ]
    reference = table[order[0]].values

    rho = [spearmanr(reference, table[m].values).statistic for m in order]
    drift = pd.DataFrame({'model': order, 'spearman_vs_first': rho})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(len(order)), rho, marker='o', color='#4477AA')
    ax.set_xticks(np.arange(len(order)), order, rotation=45, ha='right')
    ax.set_ylabel(f'Spearman ρ of importance ranking vs {order[0]}')
    ax.set_title('Does the feature-reliance profile drift?')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return drift


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument(
        '--output', type=Path, default=None, help='overrides plot.out_dir'
    )
    parser.add_argument('--max-jets', type=int, default=None, help='overrides n_jets')
    parser.add_argument('--repeats', type=int, default=None, help='overrides repeats')
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.output or Path(
        cfg.get('plot', {}).get('out_dir', 'plots/permutation_importance')
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n_jets = args.max_jets or cfg.get('n_jets')
    repeats = args.repeats or cfg.get('repeats', 5)
    base_seed = int(cfg.get('seed', 12345))
    batch_size = cfg.get('inference', {}).get('batch_size', 1024)
    num_workers = cfg.get('inference', {}).get('num_workers', 6)
    device = torch.device(args.device)

    records: list[dict] = []
    order: list[str] = []
    variants: list[Variant] | None = None
    obj: str | None = None

    for entry in cfg['models']:
        run_dir = Path(entry['dir'])
        label = entry.get('label', run_dir.name)
        order.append(label)
        print(f'\n=== {label}\n  run: {run_dir}')

        run_cfg = rc.find_run_config(run_dir)
        ckpt = rc.resolve_checkpoint(run_dir, run_cfg, entry.get('checkpoint', 'best'))
        module = rc.build_module(run_cfg, ckpt)

        this_obj = cfg.get('object', list(run_cfg.constituent_objects)[0])
        this_variants = build_variants(
            run_cfg, this_obj, cfg.get('groups', {}), cfg.get('include_global', True)
        )
        if variants is None:
            variants, obj = this_variants, this_obj
            print(
                f'  variants: {len(variants)} '
                f'({sum(v.kind == "single" for v in variants)} single, '
                f'{sum(v.kind == "group" for v in variants)} group, '
                f'{sum(v.kind == "global" for v in variants)} global)'
            )
        elif [v.name for v in this_variants] != [v.name for v in variants]:
            raise ValueError(
                f'{label} defines different features than the first model — the '
                'importance profiles would not be comparable'
            )

        eval_file = Path(cfg.get('eval_file') or run_cfg.get('val_dataset_path'))
        ce, baseline, n_used = evaluate_model(
            module,
            run_cfg,
            eval_file,
            variants,
            obj,
            device,
            batch_size,
            num_workers,
            n_jets,
            repeats,
            base_seed,
            desc=label,
        )
        print(f'  baseline CE = {baseline:.5f} on {n_used} jets from {eval_file.name}')

        for variant in variants:
            for repeat in range(repeats):
                records.append(
                    {
                        'model': label,
                        'lam': entry.get('lam'),
                        'variant': variant.name,
                        'kind': variant.kind,
                        'members': ' '.join(variant.members),
                        'repeat': repeat,
                        'checkpoint': ckpt.name,
                        'n_jets': n_used,
                        'baseline_ce': baseline,
                        'permuted_ce': ce[(variant.name, repeat)],
                        'delta_ce': ce[(variant.name, repeat)] - baseline,
                    }
                )
        del module

    raw = pd.DataFrame(records)
    raw.to_csv(out_dir / 'importance_raw.csv', index=False)

    summary = (
        raw.groupby(
            ['model', 'lam', 'variant', 'kind', 'members', 'baseline_ce'], dropna=False
        )['delta_ce']
        .agg(
            delta_ce_mean='mean',
            delta_ce_std='std',
            delta_ce_sem=lambda s: s.std() / np.sqrt(len(s)),
        )
        .reset_index()
    )
    # Normalise within (model, single features) so profile SHAPE is comparable
    # across models even when their baselines differ.
    singles = summary['kind'] == 'single'
    totals = summary[singles].groupby('model')['delta_ce_mean'].transform('sum')
    summary['fraction'] = np.nan
    summary.loc[singles, 'fraction'] = summary.loc[singles, 'delta_ce_mean'] / totals
    summary.to_csv(out_dir / 'importance_summary.csv', index=False)

    plot_heatmap(summary, order, out_dir / 'heatmap.png')
    plot_profile(summary, order, out_dir / 'profile.png', top=cfg.get('top_n', 8))
    drift = plot_drift(summary, order, out_dir / 'drift.png')
    drift.to_csv(out_dir / 'profile_drift.csv', index=False)

    print(f'\nWrote {out_dir}/')
    for name in (
        'importance_raw.csv',
        'importance_summary.csv',
        'profile_drift.csv',
        'heatmap.png',
        'profile.png',
        'drift.png',
    ):
        print(f'  {name}')

    ranking = (
        summary[singles]
        .groupby('variant')['delta_ce_mean']
        .mean()
        .sort_values(ascending=False)
    )
    print('\nMean ΔCE across models, top 10 single features:')
    for name, value in ranking.head(10).items():
        print(f'  {value:+.5f}  {name}')


if __name__ == '__main__':
    main()
