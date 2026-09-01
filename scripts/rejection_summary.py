"""Constrained fraction-optimised rejection summary at a fixed working point.

For every model entry (cached predictions from roc_comparison.py /
fraction_grid_search.py runs) and every reference tagger, this script answers
the campaign question directly:

    What is the best c-rejection at the working point, over the (f_c, f_tau)
    discriminant fractions, subject to every *constraint* background's
    rejection staying >= the reference tagger's value at its own nominal
    fractions?

i.e. the u-rejection surplus a model has over the reference can be spent on
c-rejection by raising f_c, and a tau-rejection deficit can be repaired by
raising f_tau — this script finds the best feasible trade for each model.

Inputs are prediction caches (``probs_*.npz`` written by roc_comparison.py),
so no GPU is needed; an entry may instead give ``dir:`` (a training-run
directory) to run inference like roc_comparison does.

Outputs (under ``plot.out_dir``):

  - ``summary.csv``            — one row per tagger: rejections at nominal
                                 fractions, the constrained-best point and its
                                 rejections, binomial errors
  - ``crej_vs_<x>.png``        — constrained-best c-rej vs a chosen hyper-
                                 parameter column (config ``plot.x``), with
                                 the reference bar (optional)
  - ``heatmap_<label>.png``    — c-rej over the (f_c, f_tau) grid with the
                                 infeasible region masked, per tagger

The working-point machinery matches ftag.utils.calculate_rejection: the cut
is the (1 - wp) percentile of the signal discriminant, rejection is 1/eff.
The discriminant here is the likelihood *ratio* p_sig / sum(f_b p_b) rather
than its log (roc_comparison.py) — the log is monotone, so working points,
efficiencies and rejections are identical, and skipping it makes the ~700-
point grid affordable on CPU.

Usage::

    python scripts/rejection_summary.py --config scripts/configs/rejection_summary_phase0.yaml
    python scripts/rejection_summary.py --config ... --output plots/my_summary
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import roc_comparison as rc  # noqa: E402  -- cuts/flavour conventions/inference reuse

# ---------------------------------------------------------------------------
# Rejection at a fixed working point (ratio-scale discriminant)
# ---------------------------------------------------------------------------


def rejections_at_wp(
    ratio_by_flavour: dict[str, np.ndarray],
    signal: str,
    wp: float,
) -> dict[str, float]:
    """Background rejections at signal efficiency ``wp``.

    Same semantics as ftag.utils.calculate_rejection: cut at the (1 - wp)
    percentile of the signal discriminant, rejection = 1 / passing fraction.
    """
    cut = np.percentile(ratio_by_flavour[signal], 100.0 * (1.0 - wp))
    out = {}
    for flavour, ratio in ratio_by_flavour.items():
        if flavour == signal:
            continue
        eff = np.count_nonzero(ratio > cut) / len(ratio)
        out[flavour] = 1.0 / eff if eff > 0 else np.inf
    return out


def rejection_error(rej: float, n_bkg: int) -> float:
    """Binomial error on a rejection measured from ``n_bkg`` jets."""
    if not np.isfinite(rej) or rej <= 1.0:
        return np.nan
    return rej * np.sqrt((rej - 1.0) / n_bkg)


# ---------------------------------------------------------------------------
# Per-tagger grid scan
# ---------------------------------------------------------------------------


def split_by_flavour(
    probs: np.ndarray, truth: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """For each flavour subset, the per-class probability columns it needs.

    Returns ``{flavour: {'sig': p_b, 'b': p_b, 'c': p_c, ...}}`` restricted to
    that flavour's jets, so the grid scan never re-applies boolean masks.
    """
    out = {}
    for flavour in rc.FLAVOURS:
        mask = truth == rc.TRUTH_ID[flavour]
        out[flavour] = {
            f: np.ascontiguousarray(probs[:, i][mask])
            for i, f in enumerate(rc.FLAVOURS)
        }
    return out


def scan_grid(
    by_flavour: dict[str, dict[str, np.ndarray]],
    signal: str,
    wp: float,
    fc_vals: np.ndarray,
    ftau_vals: np.ndarray,
    chunk: int = 16,
) -> pd.DataFrame:
    """Rejections at ``wp`` for every (f_c, f_tau) point with f_u > 0.

    Vectorised over grid points: denominators are one sgemm per flavour
    subset, the WP cut is the k-th order statistic per grid column (instead
    of np.percentile's interpolated value — a sub-single-jet difference).
    """
    bkg = [f for f in rc.FLAVOURS if f != signal]
    pts = [(fc, ftau) for ftau in ftau_vals for fc in fc_vals if fc + ftau < 1.0]
    # Weight matrix (n_bkg, n_points); row order must match ``bkg``.
    w_all = np.empty((len(bkg), len(pts)), dtype=np.float32)
    for col, (fc, ftau) in enumerate(pts):
        weights = {'c': fc, 'tau': ftau, 'u': 1.0 - fc - ftau}
        w_all[:, col] = [weights[b] for b in bkg]

    sig = np.ascontiguousarray(by_flavour[signal][signal], dtype=np.float32)
    sig_bkg = np.stack([by_flavour[signal][b] for b in bkg], axis=1).astype(np.float32)
    bkg_cols = {
        f: (
            np.ascontiguousarray(by_flavour[f][signal], dtype=np.float32),
            np.stack([by_flavour[f][b] for b in bkg], axis=1).astype(np.float32),
        )
        for f in bkg
    }

    k_cut = int(np.floor((len(sig) - 1) * (1.0 - wp)))
    rej = {b: np.empty(len(pts)) for b in bkg}
    with np.errstate(divide='ignore', invalid='ignore'):
        for start in range(0, len(pts), chunk):
            w = w_all[:, start : start + chunk]
            ratio_sig = sig[:, None] / (sig_bkg @ w)
            cuts = np.partition(ratio_sig, k_cut, axis=0)[k_cut]
            for b, (p_sig, p_bkg) in bkg_cols.items():
                # p_sig/denom > cut  <=>  p_sig > cut * denom  (denom > 0)
                passed = (p_sig[:, None] > (p_bkg @ w) * cuts[None, :]).sum(axis=0)
                rej[b][start : start + chunk] = np.where(
                    passed > 0, len(p_sig) / np.maximum(passed, 1), np.inf
                )

    return pd.DataFrame(
        {
            'f_c': [p[0] for p in pts],
            'f_tau': [p[1] for p in pts],
            **{f'rej_{b}': rej[b] for b in bkg},
        }
    )


def best_feasible(
    grid: pd.DataFrame, target: str, bars: dict[str, float]
) -> pd.Series | None:
    """Grid row maximising ``rej_<target>`` among rows meeting every bar."""
    feasible = grid
    for bkg, bar in bars.items():
        feasible = feasible[feasible[f'rej_{bkg}'] >= bar]
    if feasible.empty:
        return None
    return feasible.loc[feasible[f'rej_{target}'].idxmax()]


# ---------------------------------------------------------------------------
# Entry loading
# ---------------------------------------------------------------------------


def meta_from_hydra(run_dir: Path, entry: dict) -> dict:
    """Ground-truth hyper-parameters from the run's own .hydra/config.yaml.

    Run/log NAMES are not trusted (they have gone stale before) — the saved
    hydra config is authoritative.  Values found there override the entry's
    YAML metadata; a mismatch is loudly flagged.
    """
    from omegaconf import OmegaConf

    cfg = rc.find_run_config(run_dir)
    layer = cfg.encoders.constituents[cfg.constituent_objects[0]].layers[0]
    jc = float(OmegaConf.select(cfg, 'finetune.loss_weights.jet_contrastive') or 0.0)

    lam = cfg.get('lambda')
    if lam is None:  # older configs: lam lives only on the view augmentations
        lam = OmegaConf.select(cfg, 'views.0.augmentations.0.lam')

    meta = {
        'arch': layer.get('row_mode', layer.get('type')),
        # lambda/temp only ever reach the model when the contrastive term is on.
        'lambda': float(lam) if jc > 0 and lam is not None else None,
        'jc': jc,
        'temp': (
            float(OmegaConf.select(cfg, 'finetune.jet_contrastive_loss.temperature'))
            if jc > 0
            else None
        ),
        'lr': float(cfg.optimizer.lr),
        'seed': int(cfg.get('seed', 0)),
    }
    for key, value in meta.items():
        claimed = entry.get(key)
        try:  # YAML reads e.g. `3e-4` as a string — compare as numbers
            claimed, value_cmp = float(claimed), float(value)
        except (TypeError, ValueError):
            value_cmp = value
        if claimed is not None and value is not None and claimed != value_cmp:
            print(
                f'  WARNING: entry says {key}={claimed} but the hydra config '
                f'says {value} — using the hydra value'
            )
    return meta


def load_entry_probs(
    entry: dict, n_jets: int, device_arg: str | None
) -> tuple[np.ndarray, Path]:
    """Class probabilities + run dir for one entry: cached npz, or inference."""
    if entry.get('cache'):
        cached = np.load(entry['cache'], allow_pickle=False)
        probs = cached['probs']
        if probs.shape[0] < n_jets:
            raise ValueError(
                f'{entry["label"]}: cache has {probs.shape[0]:,} jets, '
                f'need {n_jets:,} — rerun inference'
            )
        print(f'  cache: {entry["cache"]}\n  ckpt:  {cached["ckpt_path"]}')
        # .../<run_dir>/<experiment_name>/version_N/checkpoints/<epoch>.ckpt
        return probs[:n_jets], Path(str(cached['ckpt_path'])).parents[3]

    import torch  # deferred: cache-only runs never need it

    run_dir = Path(entry['dir'])
    run_cfg = rc.find_run_config(run_dir)
    ckpt = rc.resolve_checkpoint(run_dir, run_cfg, entry.get('checkpoint', 'best'))
    device = torch.device(
        device_arg or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    probs = rc.run_inference(
        rc.build_module(run_cfg, ckpt),
        run_cfg,
        Path(str(run_cfg.test_dataset_path)),
        device,
        entry.get('batch_size', 1024),
        entry.get('num_workers', 4),
        n_jets,
        desc=entry['label'],
    )
    return probs, run_dir


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_heatmap(
    grid: pd.DataFrame,
    label: str,
    target: str,
    signal: str,
    wp: float,
    bars: dict[str, float],
    best: pd.Series | None,
    out_dir: Path,
) -> None:
    pivot = grid.pivot(index='f_tau', columns='f_c', values=f'rej_{target}')
    feasible = np.ones_like(pivot.values, dtype=bool)
    for bkg, bar in bars.items():
        feasible &= (
            grid.pivot(index='f_tau', columns='f_c', values=f'rej_{bkg}').values >= bar
        )

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    mesh = ax.pcolormesh(
        pivot.columns,
        pivot.index,
        np.where(feasible, pivot.values, np.nan),
        shading='auto',
        cmap='viridis',
    )
    ax.pcolormesh(
        pivot.columns,
        pivot.index,
        np.where(feasible, np.nan, pivot.values),
        shading='auto',
        cmap='Greys',
        alpha=0.35,
    )
    fig.colorbar(mesh, ax=ax, label=f'{target}-rejection @ {signal}-eff = {wp:g}')
    if best is not None:
        ax.scatter(
            [best.f_c],
            [best.f_tau],
            marker='*',
            s=250,
            color='red',
            edgecolor='white',
            linewidth=0.8,
            zorder=5,
            label=f'best feasible: f_c={best.f_c:.3f}, f_tau={best.f_tau:.3f}',
        )
        ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('$f_c$')
    ax.set_ylabel('$f_\\tau$')
    constraint = ', '.join(f'{b}-rej ≥ {v:.1f}' for b, v in bars.items())
    ax.set_title(f'{label}\nfeasible: {constraint}', fontsize=10)
    fig.tight_layout()
    slug = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()
    fig.savefig(out_dir / f'heatmap_{slug}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_crej_vs(
    table: pd.DataFrame,
    x: str,
    target: str,
    wp: float,
    bars: dict[str, float],
    ref_row: pd.Series | None,
    out_dir: Path,
) -> None:
    models = table[~table.is_reference & table[x].notna()].sort_values(x)
    if models.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    # One series per combination of the meta columns not on the x axis, so
    # e.g. the per_token lr x jc points never join the concat lambda-scan line.
    group_cols = [
        c for c in META_COLUMNS if c != x and models[c].nunique(dropna=False) > 1
    ]
    for key, group in (
        models.groupby(group_cols, dropna=False) if group_cols else [((), models)]
    ):
        key = key if isinstance(key, tuple) else (key,)
        name = ', '.join(f'{c}={v}' for c, v in zip(group_cols, key) if pd.notna(v))
        style = '-' if len(group) > 1 else 'none'
        ax.errorbar(
            group[x],
            group[f'best_rej_{target}'],
            yerr=group[f'best_rej_{target}_err'],
            marker='o',
            linestyle=style,
            capsize=3,
            label=f'constrained-best  {name}'.strip(),
        )
        ax.errorbar(
            group[x],
            group[f'nominal_rej_{target}'],
            yerr=group[f'nominal_rej_{target}_err'],
            marker='s',
            linestyle='--' if len(group) > 1 else 'none',
            capsize=3,
            alpha=0.5,
            label=f'nominal fractions  {name}'.strip(),
        )
    if ref_row is not None:
        ax.axhline(
            ref_row[f'nominal_rej_{target}'],
            color='crimson',
            linestyle=':',
            label=f'{ref_row["label"]} (nominal)',
        )
    ax.set_xlabel(x)
    ax.set_ylabel(f'{target}-rejection @ {wp:g} WP')
    constraint = ', '.join(f'{b} ≥ {v:.0f}' for b, v in bars.items())
    ax.set_title(f'Constrained-best {target}-rejection  ({constraint})', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f'crej_vs_{x}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

META_COLUMNS = ('arch', 'lambda', 'jc', 'temp', 'lr', 'seed')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Constrained fraction-optimised rejections at a fixed WP.'
    )
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--device', default=None, help='for entries without a cache')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    signal = cfg.get('signal', 'b')
    target = cfg.get('target_background', 'c')
    wp = float(cfg.get('working_point', 0.77))
    cuts = cfg.get('cuts', [])
    constraint_bkgs = list(cfg.get('constraints', ['u', 'tau']))

    grid_cfg = cfg['grid']
    fc_vals = np.linspace(
        grid_cfg['fc']['min'], grid_cfg['fc']['max'], int(grid_cfg['fc']['num'])
    )
    ftau_vals = np.linspace(
        grid_cfg['ftau']['min'], grid_cfg['ftau']['max'], int(grid_cfg['ftau']['num'])
    )

    out_dir = args.output or Path(
        cfg.get('plot', {}).get('out_dir', 'plots/rejection_summary')
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {out_dir}')

    # ── Reference tagger: defines the constraint bars ──────────────────────
    ref_cfg = cfg['reference']
    ref_fractions = ref_cfg.get('fractions', {'c': 0.2, 'tau': 0.01})
    ref_jets = rc.read_jets(Path(ref_cfg['file']), list(ref_cfg['probs'].values()))
    ref_mask = rc.cut_mask(ref_jets, cuts)
    ref_truth = ref_jets['HadronConeExclTruthLabelID'][ref_mask]
    ref_probs = np.stack(
        [ref_jets[ref_cfg['probs'][f]][ref_mask] for f in rc.FLAVOURS], axis=1
    )
    ref_by_flavour = split_by_flavour(ref_probs, ref_truth)
    n_bkg_ref = {f: len(ref_by_flavour[f][signal]) for f in rc.FLAVOURS}

    bkg = [f for f in rc.FLAVOURS if f != signal]
    weights = dict(ref_fractions)
    weights['u'] = 1.0 - sum(ref_fractions.values())
    ref_ratio = {
        f: cols[signal] / sum(weights[b] * cols[b] for b in bkg)
        for f, cols in ref_by_flavour.items()
    }
    bars_all = rejections_at_wp(ref_ratio, signal, wp)
    bars = {b: bars_all[b] for b in constraint_bkgs}
    print(f'\nReference {ref_cfg["label"]} at nominal fractions {ref_fractions}:')
    for b in bkg:
        err = rejection_error(bars_all[b], n_bkg_ref[b])
        print(f'  {b}-rej @ {wp:g} WP = {bars_all[b]:.2f} ± {err:.2f}')
    print(f'Constraint bars: {", ".join(f"{b} ≥ {v:.2f}" for b, v in bars.items())}')

    # ── Test file / jets for the model entries ─────────────────────────────
    test_file = Path(cfg['test_file'])
    n_jets = cfg.get('n_jets')
    jets = rc.read_jets(test_file, [], n_jets)
    n_jets = len(jets['pt'])
    mask = rc.cut_mask(jets, cuts)
    truth = jets['HadronConeExclTruthLabelID'][mask]
    print(f'\nTest file: {test_file} ({n_jets:,} jets, {mask.sum():,} after cuts)')

    # ── Scan every tagger ──────────────────────────────────────────────────
    rows = []
    grid_dir = out_dir / 'grids'
    grid_dir.mkdir(exist_ok=True)

    def scan_tagger(
        label: str,
        by_flavour: dict[str, dict[str, np.ndarray]],
        meta: dict,
        is_reference: bool,
    ) -> None:
        n_bkg = {f: len(by_flavour[f][signal]) for f in rc.FLAVOURS}
        nominal = ratio_discriminant_all(by_flavour, signal, ref_fractions)
        nominal_rejs = rejections_at_wp(nominal, signal, wp)

        grid = scan_grid(by_flavour, signal, wp, fc_vals, ftau_vals)
        slug = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()
        grid.to_csv(grid_dir / f'grid_{slug}.csv', index=False)
        best = best_feasible(grid, target, bars)

        row = {'label': label, 'is_reference': is_reference, **meta}
        for b in bkg:
            row[f'nominal_rej_{b}'] = nominal_rejs[b]
            row[f'nominal_rej_{b}_err'] = rejection_error(nominal_rejs[b], n_bkg[b])
        row['feasible'] = best is not None
        if best is not None:
            row['best_f_c'] = best.f_c
            row['best_f_tau'] = best.f_tau
            for b in bkg:
                row[f'best_rej_{b}'] = best[f'rej_{b}']
                row[f'best_rej_{b}_err'] = rejection_error(best[f'rej_{b}'], n_bkg[b])
        rows.append(row)
        plot_heatmap(grid, label, target, signal, wp, bars, best, out_dir)

        state = (
            f'best feasible {target}-rej = {best[f"rej_{target}"]:.2f} at '
            f'f_c={best.f_c:.3f}, f_tau={best.f_tau:.3f}'
            if best is not None
            else 'NO feasible point'
        )
        print(f'  nominal {target}-rej = {nominal_rejs[target]:.2f};  {state}')

    def ratio_discriminant_all(by_flavour, signal, fractions):
        weights = dict(fractions)
        weights['u'] = 1.0 - sum(fractions.values())
        return {
            f: cols[signal] / sum(weights[b] * cols[b] for b in bkg)
            for f, cols in by_flavour.items()
        }

    for entry in cfg.get('entries', []):
        print(f'\nModel: {entry["label"]}')
        probs, run_dir = load_entry_probs(entry, n_jets, args.device)
        by_flavour = split_by_flavour(probs[mask], truth)
        meta = {k: entry.get(k) for k in META_COLUMNS}
        meta.update(meta_from_hydra(run_dir, entry))
        scan_tagger(entry['label'], by_flavour, meta, is_reference=False)

    print(f'\nReference: {ref_cfg["label"]}')
    scan_tagger(
        ref_cfg['label'],
        ref_by_flavour,
        {k: None for k in META_COLUMNS},
        is_reference=True,
    )
    for extra in cfg.get('extra_references', []):
        print(f'\nReference: {extra["label"]}')
        ex_jets = rc.read_jets(Path(extra['file']), list(extra['probs'].values()))
        ex_mask = rc.cut_mask(ex_jets, cuts)
        ex_truth = ex_jets['HadronConeExclTruthLabelID'][ex_mask]
        ex_probs = np.stack(
            [ex_jets[extra['probs'][f]][ex_mask] for f in rc.FLAVOURS], axis=1
        )
        scan_tagger(
            extra['label'],
            split_by_flavour(ex_probs, ex_truth),
            {k: None for k in META_COLUMNS},
            is_reference=True,
        )

    # ── Outputs ────────────────────────────────────────────────────────────
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / 'summary.csv', index=False)
    print(f'\nSaved {out_dir / "summary.csv"}')

    ref_row = table[table.label == ref_cfg['label']].iloc[0]
    for x in cfg.get('plot', {}).get('x', []):
        plot_crej_vs(table, x, target, wp, bars, ref_row, out_dir)

    cols = ['label', f'nominal_rej_{target}', 'feasible', 'best_f_c', 'best_f_tau'] + [
        f'best_rej_{b}' for b in bkg
    ]
    cols = [c for c in cols if c in table.columns]
    print(
        f'\nSummary ({target}-rej @ {wp:g} WP, constraints: '
        f'{", ".join(f"{b} ≥ {v:.1f}" for b, v in bars.items())}):'
    )
    print(table[cols].to_string(index=False, float_format=lambda v: f'{v:.2f}'))


if __name__ == '__main__':
    main()
