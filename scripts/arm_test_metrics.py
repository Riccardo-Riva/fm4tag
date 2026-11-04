"""Test-set rejections of every run of the compared training methods (arms), with their spread over seeds.

The ROC plots show one model per method: the seed with the best validation CE.  This script scores ALL runs of
each method, selected and put in arms from their .hydra configs by ``plot_arm_epochs.select_runs``, each at its
best epoch (lowest ``val/head/loss_ce``):

  nominal   light / c / tau rejection on test at every working point, all models at the same fractions
  optimal   each run's own (f_c, f_tau), chosen on the VALIDATION split: maximal c-rejection at the search working
            point with light and tau rejection >= floor_frac x the reference tagger's there, then applied to test

Two steps, so that the GPU part can be split over cards:

  python scripts/arm_test_metrics.py --config scripts/configs/arm_test_metrics.yaml --infer --part 0/2
  python scripts/arm_test_metrics.py --config scripts/configs/arm_test_metrics.yaml --metrics --workers 16

``--infer`` caches val and test class probabilities under <out_dir>/cache (roc_comparison's cache: keyed by
checkpoint, file and precision).  ``--metrics`` reads only those caches and writes per_run.csv (one row per run,
plus the reference tagger) and per_arm.csv (mean, sample std, min, max per arm) to <out_dir>.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from ftag.utils import calculate_rejection
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
import fraction_constrained_search as fcs  # noqa: E402 -- (f_c, f_tau) grid sweep
import plot_arm_epochs as pae  # noqa: E402 -- run selection from .hydra configs
import roc_comparison as rc  # noqa: E402 -- model rebuild, inference, cache, discriminant

BACKGROUNDS = ('u', 'c', 'tau')
CLASS_INDEX = {rc.TRUTH_ID[f]: i for i, f in enumerate(rc.FLAVOURS)}


def n_rows(path: Path) -> int:
    with h5py.File(path, 'r') as f:
        return f['jets'].shape[0]


def cached_split_probs(
    run_dir: Path, split: str, path: Path, ac: dict, device: torch.device | None
) -> np.ndarray:
    """Class probabilities of the run's best checkpoint on one split; inference only when ``device`` is set."""
    run_cfg = rc.find_run_config(run_dir)
    ckpt = rc.resolve_checkpoint(run_dir, run_cfg, 'best')
    inf, n = ac['inference'], n_rows(path)
    slug = (
        re.sub(r'[^A-Za-z0-9]+', '_', '/'.join(run_dir.parts[-3:])).strip('_').lower()
    )

    def compute() -> np.ndarray:
        if device is None:
            raise FileNotFoundError(
                f'no cached {split} predictions for {run_dir}: run --infer first'
            )
        module = rc.build_module(run_cfg, ckpt)
        return rc.run_inference(
            module,
            run_cfg,
            path,
            device,
            inf['batch_size'],
            inf['num_workers'],
            n,
            desc=f'{slug} {split}',
            precision=inf['precision'],
        )

    return rc.cached_probs(
        Path(ac['out_dir']) / 'cache',
        f'{slug}_{split}',
        ckpt,
        path,
        n,
        False,
        compute,
        precision=inf['precision'],
    )


class Split:
    """Truth, cut mask and flavour masks of one file, plus the reference tagger's scores."""

    def __init__(self, path: Path, ac: dict):
        ref = ac['reference']
        self.jets = rc.read_jets(path, list(ref['probs'].values()))
        self.mask = rc.cut_mask(self.jets, ac['cuts'])
        truth = self.jets['HadronConeExclTruthLabelID']
        self.label = np.array([CLASS_INDEX[t] for t in truth])
        self.masks = {f: truth[self.mask] == rc.TRUTH_ID[f] for f in rc.FLAVOURS}
        self.ref_probs = {
            f: self.jets[col][self.mask] for f, col in ref['probs'].items()
        }

    def flavour_probs(self, probs: np.ndarray) -> dict[str, np.ndarray]:
        return {f: probs[self.mask, i] for i, f in enumerate(rc.FLAVOURS)}

    def rejections(
        self, probs: dict, fractions: dict, wps: list[float]
    ) -> dict[str, float]:
        disc = rc.discriminant(probs, 'b', fractions)
        sig = disc[self.masks['b']]
        return {
            f'{bkg}_rej@{wp:g}': float(
                calculate_rejection(sig, disc[self.masks[bkg]], target_eff=wp)
            )
            for wp in wps
            for bkg in BACKGROUNDS
        }


def optimum(val: Split, probs: dict, floor: dict, sc: dict, workers: int) -> dict:
    """The run's (f_c, f_tau) on validation: maximal c-rejection with light and tau rejection at the floors."""
    step = float(sc['step'])
    fcs_ = np.round(np.arange(0.0, float(sc['fc_max']), step), 10)
    ftaus = np.round(np.arange(0.0, float(sc['ftau_max']), step), 10)
    points = [(fc, ft) for ft in ftaus for fc in fcs_ if fc + ft < 1.0]
    bkg_masks = {f: val.masks[f] for f in BACKGROUNDS}
    df = fcs.sweep_joint(
        probs,
        val.masks['b'],
        bkg_masks,
        'b',
        points,
        float(sc['working_point']),
        workers,
    )
    ok = df[(df.u_rejection >= floor['u']) & (df.tau_rejection >= floor['tau'])]
    if len(ok):
        best, feasible = ok.loc[ok.c_rejection.idxmax()], True
    else:  # closest approach to both floors
        score = np.minimum(df.u_rejection / floor['u'], df.tau_rejection / floor['tau'])
        best, feasible = df.loc[score.idxmax()], False
    edge = (
        best.f_c >= float(sc['fc_max']) - 2 * step
        or best.f_tau >= float(sc['ftau_max']) - 2 * step
    )
    return {
        'f_c': float(best.f_c),
        'f_tau': float(best.f_tau),
        'feasible': feasible,
        'grid_edge': edge,
    }


def infer(ac: dict, runs: pd.DataFrame, part: str) -> None:
    index, count = (int(x) for x in part.split('/'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    for run in list(runs.itertuples())[index::count]:
        for split in ('val', 'test'):
            cached_split_probs(
                Path(run.run_dir), split, Path(ac[f'{split}_file']), ac, device
            )


def metrics(ac: dict, runs: pd.DataFrame, workers: int) -> None:
    wps, sc = [float(w) for w in ac['working_points']], ac['search']
    val, test = Split(Path(ac['val_file']), ac), Split(Path(ac['test_file']), ac)
    ref = ac['reference']
    ref_val = val.rejections(
        val.ref_probs, ref['fractions'], [float(sc['working_point'])]
    )
    floor = {
        b: float(sc['floor_frac']) * ref_val[f'{b}_rej@{float(sc["working_point"]):g}']
        for b in ('u', 'tau')
    }
    print(
        f'floors on val at {sc["working_point"]}: light >= {floor["u"]:.1f}, tau >= {floor["tau"]:.2f}'
    )

    rows = [
        {
            'arm': ref['label'],
            'f_c': ref['fractions']['c'],
            'f_tau': ref['fractions']['tau'],
        }
        | {
            f'nominal_{k}': v
            for k, v in test.rejections(test.ref_probs, ref['fractions'], wps).items()
        }
        | {
            f'optimal_{k}': v
            for k, v in test.rejections(test.ref_probs, ref['fractions'], wps).items()
        }
    ]
    for run in runs.itertuples():
        run_dir = Path(run.run_dir)
        p_val = cached_split_probs(run_dir, 'val', Path(ac['val_file']), ac, None)
        p_test = cached_split_probs(run_dir, 'test', Path(ac['test_file']), ac, None)
        opt = optimum(val, val.flavour_probs(p_val), floor, sc, workers)
        fractions = {'c': opt['f_c'], 'tau': opt['f_tau']}
        nll = -np.log(
            np.clip(p_test[np.arange(len(test.label)), test.label], 1e-12, None)
        )
        row = {
            'arm': run.arm,
            'seed': run.seed,
            'pretraining_run': run.pretraining_run,
            'run_dir': run.run_dir,
            'best_val_ce': run.best_val_ce,
            'best_epoch': run.best_epoch,
            'test_ce': float(nll.mean()),
            **opt,
        }
        row |= {
            f'nominal_{k}': v
            for k, v in test.rejections(
                test.flavour_probs(p_test), ac['nominal_fractions'], wps
            ).items()
        }
        row |= {
            f'optimal_{k}': v
            for k, v in test.rejections(
                test.flavour_probs(p_test), fractions, wps
            ).items()
        }
        rows.append(row)
        print(
            f'{run.arm:8s} seed {run.seed:<8} f_c {opt["f_c"]:.3f} f_tau {opt["f_tau"]:.3f} feasible {opt["feasible"]}'
        )

    per_run = pd.DataFrame(rows)
    out_dir = Path(ac['out_dir'])
    per_run.to_csv(out_dir / 'per_run.csv', index=False)
    numeric = [
        c
        for c in per_run.columns
        if c.startswith(('nominal_', 'optimal_')) or c in ('best_val_ce', 'test_ce')
    ]
    runs_only = per_run[per_run.arm != ref['label']]
    per_arm = runs_only.groupby('arm', sort=False)[numeric].agg(
        ['mean', 'std', 'min', 'max']
    )
    per_arm.insert(0, ('n', ''), runs_only.groupby('arm', sort=False).size())
    per_arm.insert(
        1,
        ('infeasible', ''),
        runs_only.groupby('arm', sort=False)['feasible'].apply(
            lambda s: int((~s).sum())
        ),
    )
    per_arm.to_csv(out_dir / 'per_arm.csv')
    show = [
        c
        for c in numeric
        if c.endswith(('@0.77', '@0.8')) or c in ('best_val_ce', 'test_ce')
    ]
    table = per_run.groupby('arm', sort=False)[show].agg(
        lambda s: (
            f'{s.mean():.4g} ± {s.std(ddof=1):.2g}'
            if len(s) > 1
            else f'{s.iloc[0]:.4g}'
        )
    )
    print(table.T.to_string())
    print(f'Saved {out_dir / "per_run.csv"} and {out_dir / "per_arm.csv"}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config')
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument(
        '--infer', action='store_true', help='Cache val and test predictions (GPU)'
    )
    step.add_argument(
        '--metrics',
        action='store_true',
        help='Rejections from the cached predictions (CPU)',
    )
    parser.add_argument(
        '--part',
        default='0/1',
        help='With --infer: this process takes runs i, i+n, ... ("i/n")',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='With --metrics: processes of the grid sweep',
    )
    args = parser.parse_args()

    ac = OmegaConf.to_container(
        OmegaConf.load(args.config)['arm_test_metrics'], resolve=True
    )
    Path(ac['out_dir'], 'cache').mkdir(parents=True, exist_ok=True)
    runs = pae.select_runs(ac)
    if args.infer:
        infer(ac, runs, args.part)
    else:
        metrics(ac, runs, args.workers)


if __name__ == '__main__':
    main()
