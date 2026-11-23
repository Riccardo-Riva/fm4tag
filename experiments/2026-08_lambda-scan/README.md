# λ (noise) scan and the "beat GN2" analysis (2026-08-18 → 2026-10-12)

**Status:** closed (code: tag `exp/lambda-scan-final`).

## Question
Does CE + JC at augmentation noise level λ beat CE alone and the GN2 tagger in c-rejection at fixed b-efficiency?

## Data
`data_parallel_preprocess_251112_112022`: train 52.0 M / val 5.0 M / test 5.0 M jets (ttbar, 20 < pT < 250 GeV, |eta| < 2.5).

## Model
**Model A (`encoders=rowcol_v0`, 2.2 M parameters).** Per-feature tokens of width 16 for the jet (2 features) and
for each track (19 features); one RowCol stack of depth 6 per object type: column attention across the feature tokens
of one object (2 heads x 8) and **row (intersample) attention through ISAB** (16 learned inducing points per feature
token, `row_mode: per_token`) over the rank-local batch; projectors to 32 (jet) and 128 (tracks).
`TransformerAggregator`: a 6-layer transformer over the valid tracks of each jet (4 heads x 64, GEGLU x4), mean pool;
jet vector z_jet = [z_g, pooled z_t] (160 dims) -> MLP head (hidden 256, dropout 0.02) -> 4 classes (b, c, light, tau).

## Training methods
From scratch, CE + 0.3 x JC (T 0.1), views CutMix(λ) + Mixup(λ) vs identity, λ from 0.05 to 0.9; DDP 2 x 1024 (and a 1-GPU 2048 ladder), AdamW 3e-4, cosine, early stopping (patience 15) on val CE; unweighted CE; no-JC and no-JC + dropout 0.1 baselines; several seeds per λ.

## Stages and configs
Launcher `slurm/classify_param_scan.sh` → `outputs/classify_param_scan/run_<TS>/`; `slurm/submit_lambda_seed_fraction_roc.sh`; plot / ROC / fraction-search configs in `configs/` (run with the generic `scripts/submit_*.sh` tools); figure list `plot_paths.txt`; fact sheet `paper/factsheets/fm4tag_lambda_scan.md`.

## Results
JC beats no-JC on val CE (λ 0.25–0.35: 0.6015–0.6024 vs 0.6035, n = 4) and passes GN2v01 c-rejection with optimised fractions (+0.5 % at λ 0.45; no-JC −2.2 %). Caveats (see `docs/repo_inspection_2026-11-23.md` § 8): no-JC + dropout 0.1 reaches the same CE (0.6016); lr-schedule confound; **fractions optimised on the test set**; unweighted CE vs GN2's class-weighted CE.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | cells files (one per stage) and the YAML configs of the plot / analysis scripts | yes |
| `slurm/` | launchers specific to this experiment (generic ones are in the repo's `slurm/` and `scripts/`) | yes |
| `notes/`, `diagnostics/` | design notes, reports, one-off diagnostic scripts (where present) | yes |
| `outputs/` | run folders (`<STAGE>_<timestamp>/<cell>/seed_<s>/`: `.hydra/`, `metrics.csv`, checkpoints, `run_info.yaml`) and logs | no |
| `plots/` | every figure and metric table of the experiment | no |

## Related experiments
Grew out of `2026-07_ce-jc-grids`; questioned by `2026-11_four-way-jc-dropout`; salt port of the same scan in `/storage3/DSIP/rriva/research/salt-ririva/slurm/scratch_lambda_scan/output/`.

`plots/` also holds the outputs of the earlier ROC / fraction tools (`run_<TS>_*`, `roc_gallery_*`, `rejection_summary_phase0`, `permutation_importance_*`); `outputs/script_logs/` = their logs.
