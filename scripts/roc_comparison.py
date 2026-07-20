"""ROC comparison of trained fm4tag classifiers (+ optional reference taggers).

Given a YAML config listing training-run directories (as produced by the
``slurm/classify_from_scratch`` pipeline), this script

1. resolves each run's best checkpoint — the saved epoch with the best
   monitored metric in ``metrics.csv`` (monitor/mode are read from the run's
   own ModelCheckpoint config),
2. rebuilds the ``FinetuneModule`` from the run's ``.hydra/config.yaml``
   exactly as ``fm4tag.runner.run`` does,
3. runs test-set inference — class probabilities are cached as ``.npz`` under
   ``<out_dir>/cache/``, so re-plotting does not re-run inference,
4. draws puma (ATLAS-style) comparison plots:

   - ``roc.png``              — ROC curves, one ratio panel per background
   - ``disc_<label>.png``     — per-flavour discriminant distributions, one per tagger
   - ``disc_signal.png``      — signal-jet discriminant, all taggers overlaid
   - ``rej_vs_pt_<bkg>.png``  — background rejection vs pT at a fixed WP (optional)
   - ``sig_eff_vs_pt.png``    — signal efficiency vs pT at a fixed WP (optional)
   - ``working_points.csv``   — rejection table at fixed working points

Reference taggers whose scores are stored as HDF5 jet fields (GN2v01, DL1dv01,
GN2 Open Data, GN2 only-JT, …) can be overlaid via the ``reference_taggers``
config block — either from the test file itself or from external (e.g. salt)
evaluation files.  External files are always read in full; ``--max-jets`` only
truncates the fm4tag test file.

The discriminant is the fraction-weighted log-likelihood ratio

    D_sig = log( p_sig / sum_bkg f_bkg * p_bkg )

where flavours not listed in ``fractions`` share the remainder equally: for
b-tagging with ``fractions: {c: 0.2, tau: 0.01}`` this is the GN2 D_b with
f_u = 0.79, and a DL1d-style tagger without a tau output gets f_u = 1 - f_c.

Usage::

    python scripts/roc_comparison.py --config scripts/configs/roc_comparison.yaml
    python scripts/roc_comparison.py --config ... --output plots/my_comparison
    python scripts/roc_comparison.py --config ... --recompute       # ignore cached predictions
    python scripts/roc_comparison.py --config ... --max-jets 50000  # quick partial run
    python scripts/roc_comparison.py --config ... --device cpu

``--output`` overrides ``plot.out_dir`` from the config.  Note that the
prediction cache lives under the output directory, so a fresh output directory
re-runs inference.
"""

from __future__ import annotations

import argparse
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ftag import Flavours
from ftag.utils import calculate_rejection
from puma import Histogram, HistogramPlot, Roc, RocPlot, VarVsEff, VarVsEffPlot

from fm4tag.datasets.datasets import DatasetCatCon, cat_con_collate_fn
from fm4tag.utils import build_aggregator, build_encoders, build_head

# ---------------------------------------------------------------------------
# Flavour conventions
# ---------------------------------------------------------------------------

# Column order of the fm4tag class probabilities (= flavour_label 0..3).
FLAVOURS = ('b', 'c', 'u', 'tau')
TRUTH_ID = {'b': 5, 'c': 4, 'u': 0, 'tau': 15}  # HadronConeExclTruthLabelID
PUMA_CLASS = {'b': 'bjets', 'c': 'cjets', 'u': 'ujets', 'tau': 'taujets'}

CUT_OPS = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
}

# Paul Tol's 'bright' qualitative palette (colour-blind safe); one colour per
# tagger, assigned in config order.  puma additionally encodes the rejection
# class as linestyle, so identity never relies on colour alone.
TAGGER_COLOURS = ('#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377')


@dataclass
class Tagger:
    """Everything needed to plot one tagger (model or reference)."""

    label: str
    colour: str | None
    disc: np.ndarray  # discriminant, after kinematic cuts
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # flavour -> bool
    pt: np.ndarray | None = None  # jet pT after cuts (for the pT profiles)
    is_reference: bool = False


# ---------------------------------------------------------------------------
# Run resolution: hydra config + best checkpoint
# ---------------------------------------------------------------------------


def find_run_config(run_dir: Path) -> DictConfig:
    """Load the newest ``.hydra/config.yaml`` under ``run_dir`` (one per rank)."""
    candidates = sorted(
        run_dir.glob('outputs/*/*/.hydra/config.yaml'),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f'No .hydra/config.yaml found under {run_dir}')
    return OmegaConf.load(candidates[-1])


def resolve_checkpoint(run_dir: Path, run_cfg: DictConfig, which: str) -> Path:
    """Resolve a checkpoint inside ``run_dir``.

    ``which`` is ``'best'`` (default: saved epoch with the best monitored
    metric from metrics.csv), ``'last'`` (highest epoch), or an explicit path.
    """
    if which not in ('best', 'last'):
        path = Path(which)
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint {path} not found')
        return path

    ckpt_dirs = sorted(
        run_dir.glob('*/version_*/checkpoints'), key=lambda p: p.stat().st_mtime
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f'No checkpoints directory found under {run_dir}')
    ckpt_dir = ckpt_dirs[-1]

    by_epoch: dict[int, Path] = {}
    for path in ckpt_dir.glob('*.ckpt'):
        match = re.search(r'epoch(\d+)', path.name)
        if match:
            by_epoch[int(match.group(1))] = path
    if not by_epoch:
        raise FileNotFoundError(f'No epochNNN-*.ckpt files in {ckpt_dir}')

    if which == 'last':
        epoch = max(by_epoch)
        print(f'  checkpoint: epoch {epoch} (last) — {by_epoch[epoch].name}')
        return by_epoch[epoch]

    # 'best': look the saved epochs up in the run's metrics.csv, using the
    # monitor/mode the run's own ModelCheckpoint used.
    monitor = OmegaConf.select(run_cfg, 'callbacks.model_checkpoint.monitor')
    mode = OmegaConf.select(run_cfg, 'callbacks.model_checkpoint.mode') or 'min'
    monitor = monitor or 'val/loss'

    metrics_csv = ckpt_dir.parent / 'metrics.csv'
    if not metrics_csv.is_file():
        epoch = max(by_epoch)
        print(f'  WARNING: {metrics_csv} missing — falling back to last epoch {epoch}')
        return by_epoch[epoch]

    metrics = pd.read_csv(metrics_csv)
    per_epoch = metrics.dropna(subset=[monitor]).groupby('epoch')[monitor].last()
    per_epoch.index = per_epoch.index.astype(int)

    worst = -np.inf if mode == 'max' else np.inf
    scores = {e: per_epoch.get(e, worst) for e in by_epoch}
    epoch = (max if mode == 'max' else min)(scores, key=scores.get)
    print(
        f'  checkpoint: epoch {epoch} ({monitor}={scores[epoch]:.4f}, {mode} '
        f'of epochs {sorted(by_epoch)}) — {by_epoch[epoch].name}'
    )
    return by_epoch[epoch]


# ---------------------------------------------------------------------------
# Model rebuild + inference
# ---------------------------------------------------------------------------


def build_module(run_cfg: DictConfig, ckpt_path: Path) -> torch.nn.Module:
    """Rebuild the FinetuneModule from config and load checkpoint weights.

    Mirrors the finetune branch of :func:`fm4tag.runner.run.run`.
    """
    encoders = build_encoders(run_cfg)
    aggregator = build_aggregator(run_cfg, encoders)
    head = build_head(run_cfg, aggregator)
    views = [hydra_instantiate(v) for v in run_cfg.get('views', [])]

    overrides: dict = {}
    if run_cfg.get('use_class_weights', False):
        with open(run_cfg.class_dict_path) as f:
            class_dict = yaml.safe_load(f)
        label_var = run_cfg.variables[run_cfg.global_object].label
        overrides['class_weights'] = class_dict[run_cfg.global_object][label_var]

    n_classes = len(run_cfg.variables[run_cfg.global_object].unique_labels)
    module = hydra_instantiate(
        run_cfg.finetune,
        encoders=encoders,
        aggregator=aggregator,
        head=head,
        views=views,
        n_classes=n_classes,
        _convert_='all',
        **overrides,
    )

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    module.on_load_checkpoint(ckpt)  # backwards-compat key handling
    module.load_state_dict(ckpt['state_dict'])
    module.eval()
    return module


def run_inference(
    module: torch.nn.Module,
    run_cfg: DictConfig,
    test_file: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    n_jets: int | None,
    desc: str,
) -> np.ndarray:
    """Class probabilities ``(n_jets, n_classes)`` on the (truncated) test set."""
    dataset = DatasetCatCon(
        file_path=str(test_file),
        variables=OmegaConf.to_container(run_cfg.variables, resolve=True),
        global_object=run_cfg.global_object,
        constituent_objects=list(run_cfg.constituent_objects),
        norm_dict=_load_yaml(run_cfg.get('norm_dict_path')),
        class_dict=_load_yaml(run_cfg.get('class_dict_path')),
    )
    if n_jets is not None and n_jets < len(dataset):
        dataset = Subset(dataset, range(n_jets))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cat_con_collate_fn,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2 if num_workers > 0 else None,
        # See CatConDataModule: avoids CUDA re-init crashes in fork children.
        multiprocessing_context='forkserver' if num_workers > 0 else None,
    )

    module.to(device)
    probs = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc):
            batch_dev = {
                'global': batch['global'].to(device, non_blocking=True),
                'constituents': {
                    obj: {k: v.to(device, non_blocking=True) for k, v in fields.items()}
                    for obj, fields in batch['constituents'].items()
                },
            }
            probs.append(module.predict_step(batch_dev, 0).float().cpu())
    module.cpu()
    return torch.cat(probs).numpy()


def cached_probs(
    cache_dir: Path,
    label: str,
    ckpt_path: Path,
    test_file: Path,
    n_jets: int,
    recompute: bool,
    compute,
) -> np.ndarray:
    """Load cached predictions if they match ckpt/test-file, else recompute."""
    slug = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()
    cache_file = cache_dir / f'probs_{slug}.npz'

    if cache_file.is_file() and not recompute:
        cached = np.load(cache_file, allow_pickle=False)
        if (
            str(cached['ckpt_path']) == str(ckpt_path)
            and str(cached['test_file']) == str(test_file)
            and cached['probs'].shape[0] >= n_jets
        ):
            print(f'  using cached predictions: {cache_file}')
            return cached['probs'][:n_jets]
        print('  cache stale (different checkpoint/test file/size) — recomputing')

    probs = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_file, probs=probs, ckpt_path=str(ckpt_path), test_file=str(test_file)
    )
    print(f'  cached predictions: {cache_file}')
    return probs


# ---------------------------------------------------------------------------
# Jets, cuts, discriminant
# ---------------------------------------------------------------------------


def _load_yaml(path: str | None) -> dict | None:
    if path is None:
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def read_jets(
    file_path: Path, columns: list[str], n_jets: int | None = None
) -> dict[str, np.ndarray]:
    """Read jet columns (kinematics + truth + requested scores) from HDF5."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(file_path, 'r') as f:
        jets = f['jets']
        stop = jets.shape[0] if n_jets is None else min(n_jets, jets.shape[0])
        for col in dict.fromkeys(['pt', 'eta', 'HadronConeExclTruthLabelID', *columns]):
            out[col] = jets[col][:stop]
    return out


def cut_mask(jets: dict[str, np.ndarray], cuts: list) -> np.ndarray:
    """Boolean mask from config cuts ``[[var, op, value], ...]``."""
    mask = np.ones(len(jets['pt']), dtype=bool)
    for var, op, value in cuts:
        mask &= CUT_OPS[op](jets[var], value)
    return mask


def discriminant(
    probs: dict[str, np.ndarray], signal: str, fractions: dict[str, float]
) -> np.ndarray:
    """Fraction-weighted log-likelihood-ratio discriminant.

    Background flavours present in ``probs`` but not in ``fractions`` share
    ``1 - sum(fractions)`` equally.  Flavours in ``fractions`` that the tagger
    has no output for are ignored (their weight is re-absorbed).
    """
    bkg = [f for f in probs if f != signal]
    explicit = {f: fractions[f] for f in bkg if f in fractions}
    rest = [f for f in bkg if f not in fractions]
    weights = dict(explicit)
    if rest:
        leftover = 1.0 - sum(explicit.values())
        weights.update({f: leftover / len(rest) for f in rest})

    denom = sum(weights[f] * probs[f] for f in bkg)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(probs[signal] / denom)


def make_tagger(
    label: str,
    colour: str | None,
    probs: dict[str, np.ndarray],
    jets: dict[str, np.ndarray],
    cuts: list,
    signal: str,
    fractions: dict[str, float],
    is_reference: bool,
) -> Tagger:
    """Apply kinematic cuts and package discriminant + flavour masks."""
    mask = cut_mask(jets, cuts)
    truth = jets['HadronConeExclTruthLabelID'][mask]
    return Tagger(
        label=label,
        colour=colour,
        disc=discriminant({f: p[mask] for f, p in probs.items()}, signal, fractions),
        masks={f: truth == TRUTH_ID[f] for f in FLAVOURS},
        pt=jets['pt'][mask],
        is_reference=is_reference,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_rocs(
    taggers: list[Tagger],
    signal: str,
    backgrounds: list[str],
    sig_eff: np.ndarray,
    plot_cfg: dict,
    out_dir: Path,
) -> None:
    roc_plot = RocPlot(
        n_ratio_panels=len(backgrounds),
        ylabel='Background rejection',
        xlabel=f'${signal}$-jet efficiency',
        figsize=tuple(plot_cfg.get('figsize', (7, 7))),
        y_scale=plot_cfg.get('y_scale', 1.3),
        atlas_brand=cfg.get('atlas_brand', 'ATLAS'),
        atlas_first_tag=plot_cfg.get('atlas_first_tag', 'Simulation Internal'),
        atlas_second_tag=plot_cfg.get('atlas_second_tag', ''),
        dpi=plot_cfg.get('dpi', 200),
    )
    for tagger in taggers:
        disc_sig = tagger.disc[tagger.masks[signal]]
        for bkg in backgrounds:
            disc_bkg = tagger.disc[tagger.masks[bkg]]
            roc_plot.add_roc(
                Roc(
                    sig_eff,
                    calculate_rejection(disc_sig, disc_bkg, sig_eff),
                    n_test=len(disc_bkg),
                    rej_class=PUMA_CLASS[bkg],
                    signal_class=PUMA_CLASS[signal],
                    label=tagger.label,
                    colour=tagger.colour,
                ),
                reference=tagger.is_reference,
            )
    for panel, bkg in enumerate(backgrounds, start=1):
        roc_plot.set_ratio_class(panel, Flavours[PUMA_CLASS[bkg]])
    roc_plot.draw()
    roc_plot.savefig(str(out_dir / 'roc.png'), transparent=False)
    print(f'Saved {out_dir / "roc.png"}')


def plot_disc_distributions(
    taggers: list[Tagger],
    signal: str,
    plot_cfg: dict,
    out_dir: Path,
) -> None:
    disc_cfg = plot_cfg.get('disc_plot', {})
    lo, hi = disc_cfg.get('range', (-10, 20))
    bins = np.linspace(lo, hi, disc_cfg.get('bins', 50) + 1)
    second_tag = plot_cfg.get('atlas_second_tag', '')
    common = dict(
        xlabel=f'$D_{{{signal}}}$',
        ylabel='Normalised entries',
        logy=disc_cfg.get('logy', False),
        atlas_first_tag=plot_cfg.get('atlas_first_tag', 'Simulation Internal'),
        dpi=plot_cfg.get('dpi', 200),
    )

    # Per-flavour distributions, one figure per tagger.
    for tagger in taggers:
        hist_plot = HistogramPlot(
            atlas_second_tag=f'{second_tag}\n{tagger.label}'.strip('\n'), **common
        )
        for flavour in FLAVOURS:
            puma_flavour = Flavours[PUMA_CLASS[flavour]]
            hist_plot.add(
                Histogram(
                    tagger.disc[tagger.masks[flavour]],
                    bins=bins,
                    flavour=puma_flavour,
                    label=puma_flavour.label,
                )
            )
        hist_plot.draw()
        slug = re.sub(r'[^A-Za-z0-9]+', '_', tagger.label).strip('_').lower()
        hist_plot.savefig(str(out_dir / f'disc_{slug}.png'), transparent=False)
        print(f'Saved {out_dir / f"disc_{slug}.png"}')

    # Signal-jet discriminant, all taggers overlaid (ratio wrt the reference).
    hist_plot = HistogramPlot(
        n_ratio_panels=1, atlas_second_tag=f'{second_tag}\n${signal}$-jets', **common
    )
    for tagger in taggers:
        hist_plot.add(
            Histogram(
                tagger.disc[tagger.masks[signal]],
                bins=bins,
                label=tagger.label,
                colour=tagger.colour,
            ),
            reference=tagger.is_reference,
        )
    hist_plot.draw()
    hist_plot.savefig(str(out_dir / 'disc_signal.png'), transparent=False)
    print(f'Saved {out_dir / "disc_signal.png"}')


def plot_eff_vs_pt(
    taggers: list[Tagger],
    signal: str,
    backgrounds: list[str],
    eff_cfg: dict,
    plot_cfg: dict,
    out_dir: Path,
) -> None:
    """Signal efficiency / background rejection vs jet pT at a fixed WP."""
    wp = eff_cfg.get('working_point', 0.77)
    bins = np.asarray(eff_cfg.get('bins_gev', [20, 30, 40, 60, 85, 110, 140, 175, 250]))
    flat_per_bin = eff_cfg.get('flat_per_bin', False)
    wp_kind = 'flat' if flat_per_bin else 'inclusive'
    second_tag = plot_cfg.get('atlas_second_tag', '')
    common = dict(
        xlabel=r'$p_{T}$ [GeV]',
        n_ratio_panels=1,
        figsize=tuple(plot_cfg.get('figsize', (7, 7))),
        atlas_first_tag=plot_cfg.get('atlas_first_tag', 'Simulation Internal'),
        atlas_second_tag=(
            f'{second_tag}\n{wp_kind} $\\varepsilon_{{{signal}}}$ = {wp:.0%} WP'
        ).strip('\n'),
        dpi=plot_cfg.get('dpi', 200),
    )

    def curves(bkg: str) -> list[tuple[VarVsEff, bool]]:
        out = []
        for tagger in taggers:
            sig_mask, bkg_mask = tagger.masks[signal], tagger.masks[bkg]
            out.append(
                (
                    VarVsEff(
                        x_var_sig=tagger.pt[sig_mask] / 1_000,
                        disc_sig=tagger.disc[sig_mask],
                        x_var_bkg=tagger.pt[bkg_mask] / 1_000,
                        disc_bkg=tagger.disc[bkg_mask],
                        bins=bins,
                        working_point=wp,
                        flat_per_bin=flat_per_bin,
                        label=tagger.label,
                        colour=tagger.colour,
                    ),
                    tagger.is_reference,
                )
            )
        return out

    for bkg in backgrounds:
        eff_plot = VarVsEffPlot(
            mode='bkg_rej', ylabel=f'${bkg}$-jet rejection', logy=True, **common
        )
        for curve, is_reference in curves(bkg):
            eff_plot.add(curve, reference=is_reference)
        eff_plot.draw()
        eff_plot.savefig(str(out_dir / f'rej_vs_pt_{bkg}.png'), transparent=False)
        print(f'Saved {out_dir / f"rej_vs_pt_{bkg}.png"}')

    # Per-bin signal efficiency of the inclusive/flat cut (bkg choice is moot).
    eff_plot = VarVsEffPlot(
        mode='sig_eff', ylabel=f'${signal}$-jet efficiency', **common
    )
    for curve, is_reference in curves(backgrounds[0]):
        eff_plot.add(curve, reference=is_reference)
    eff_plot.draw()
    eff_plot.savefig(str(out_dir / 'sig_eff_vs_pt.png'), transparent=False)
    print(f'Saved {out_dir / "sig_eff_vs_pt.png"}')


def working_point_table(
    taggers: list[Tagger],
    signal: str,
    backgrounds: list[str],
    working_points: list[float],
    out_dir: Path,
) -> None:
    rows = []
    for tagger in taggers:
        disc_sig = tagger.disc[tagger.masks[signal]]
        for wp in working_points:
            row: dict = {'tagger': tagger.label, f'eff_{signal}': f'{wp:.0%}'}
            for bkg in backgrounds:
                rej = calculate_rejection(
                    disc_sig, tagger.disc[tagger.masks[bkg]], np.array([wp])
                )
                row[f'rej_{bkg}'] = float(np.atleast_1d(rej)[0])
            rows.append(row)
    table = pd.DataFrame(rows)

    out_file = out_dir / 'working_points.csv'
    table.to_csv(out_file, index=False)
    print(f'\nBackground rejections at fixed {signal}-efficiency working points:')
    print(table.to_string(index=False, float_format=lambda x: f'{x:.1f}'))
    print(f'Saved {out_file}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='ROC comparison of fm4tag classifiers (puma / ATLAS style).'
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config file')
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Plot output directory (overrides plot.out_dir from the config)',
    )
    parser.add_argument(
        '--recompute', action='store_true', help='Ignore cached predictions'
    )
    parser.add_argument(
        '--max-jets', type=int, default=None, help='Truncate the test set (smoke test)'
    )
    parser.add_argument(
        '--device', default=None, help="Inference device (default: cuda if available)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    signal = cfg.get('signal', 'b')
    backgrounds = list(cfg.get('backgrounds', ['u', 'c', 'tau']))
    fractions = cfg.get('fractions', {'c': 0.2, 'tau': 0.01})
    cuts = cfg.get('cuts', [])
    eff_cfg = cfg.get('sig_eff', {})
    sig_eff = np.linspace(
        eff_cfg.get('min', 0.6), eff_cfg.get('max', 1.0), eff_cfg.get('num', 100)
    )
    plot_cfg = cfg.get('plot', {})
    inference_cfg = cfg.get('inference', {})

    out_dir = args.output or Path(plot_cfg.get('out_dir', 'plots/roc_comparison'))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {out_dir}')

    device = torch.device(
        args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    if device.type == 'cpu':
        print('WARNING: running inference on CPU — use a GPU node for the full test set')
    torch.set_float32_matmul_precision('high')

    # ── Resolve runs and the shared test file ─────────────────────────────
    model_entries = cfg['models']
    run_cfgs, ckpts = [], []
    for entry in model_entries:
        run_dir = Path(entry['dir'])
        print(f'\nModel: {entry["label"]} ({run_dir.name})')
        run_cfg = find_run_config(run_dir)
        run_cfgs.append(run_cfg)
        ckpts.append(
            resolve_checkpoint(run_dir, run_cfg, entry.get('checkpoint', 'best'))
        )

    test_files = {str(c.test_dataset_path) for c in run_cfgs}
    if cfg.get('test_file'):
        test_file = Path(cfg['test_file'])
    else:
        if len(test_files) > 1:
            raise ValueError(
                f'Runs used different test files {test_files}; set test_file explicitly'
            )
        test_file = Path(test_files.pop())
    print(f'\nTest file: {test_file}')

    with h5py.File(test_file, 'r') as f:
        n_total = f['jets'].shape[0]
    n_jets = n_total if args.max_jets is None else min(args.max_jets, n_total)
    if n_jets < n_total:
        print(f'Truncating test set to the first {n_jets:,} of {n_total:,} jets')

    # ── fm4tag model predictions (cached) ─────────────────────────────────
    ref_entries = cfg.get('reference_taggers') or []
    palette = iter(TAGGER_COLOURS)

    def next_colour(entry: dict) -> str | None:
        if 'colour' in entry:
            return entry['colour']
        return next(palette, None)  # palette exhausted → puma default cycle

    jets_main = read_jets(
        test_file,
        [c for e in ref_entries if not e.get('file') for c in e['probs'].values()],
        n_jets,
    )

    reference_label = cfg.get('reference', model_entries[0]['label'])
    taggers: list[Tagger] = []
    for entry, run_cfg, ckpt_path in zip(model_entries, run_cfgs, ckpts):
        probs = cached_probs(
            out_dir / 'cache',
            entry['label'],
            ckpt_path,
            test_file,
            n_jets,
            args.recompute,
            compute=lambda: run_inference(
                build_module(run_cfg, ckpt_path),
                run_cfg,
                test_file,
                device,
                inference_cfg.get('batch_size', 1024),
                inference_cfg.get('num_workers', 4),
                n_jets,
                desc=entry['label'],
            ),
        )
        taggers.append(
            make_tagger(
                entry['label'],
                next_colour(entry),
                {f: probs[:, i] for i, f in enumerate(FLAVOURS)},
                jets_main,
                cuts,
                signal,
                fractions,
                is_reference=(entry['label'] == reference_label),
            )
        )

    # ── Reference taggers from HDF5 score columns ─────────────────────────
    for entry in ref_entries:
        if entry.get('file'):
            jets = read_jets(Path(entry['file']), list(entry['probs'].values()))
        else:
            jets = jets_main
        taggers.append(
            make_tagger(
                entry['label'],
                next_colour(entry),
                {f: jets[col] for f, col in entry['probs'].items()},
                jets,
                cuts,
                signal,
                entry.get('fractions', fractions),
                is_reference=(entry['label'] == reference_label),
            )
        )

    if not any(t.is_reference for t in taggers):
        raise ValueError(f'reference label {reference_label!r} matches no tagger')

    for tagger in taggers:
        counts = '  '.join(f'{f}: {tagger.masks[f].sum():,}' for f in FLAVOURS)
        print(f'{tagger.label:>24} — jets after cuts: {counts}')

    # ── Plots + tables ─────────────────────────────────────────────────────
    plot_rocs(taggers, signal, backgrounds, sig_eff, plot_cfg, out_dir)
    plot_disc_distributions(taggers, signal, plot_cfg, out_dir)
    if cfg.get('eff_vs_pt'):
        plot_eff_vs_pt(taggers, signal, backgrounds, cfg['eff_vs_pt'], plot_cfg, out_dir)
    working_point_table(
        taggers, signal, backgrounds, cfg.get('working_points', [0.70, 0.77]), out_dir
    )


if __name__ == '__main__':
    main()
