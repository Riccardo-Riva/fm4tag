# Fact sheet: fm4tag noise-parameter (lambda) scan, multi-task CE + jet-contrastive (JC)

Area key: `fm4tag_lambda_scan`. Compiled 2026-11-17, read-only. Provenance tags: [files] = verified in run dirs / csv / configs / code;
[memory] = memory note only; [chat] = chat transcript d2e47d07; [inferred] = my computation or reasoning.
Scratch scripts and recomputed tables: `/tmp/claude-1138/-storage3-DSIP-rriva-research-fm4tag/653cf529-153e-437a-bcf0-9bb91ed33820/scratchpad/fm4tag_lambda_scan/`
(`recompute.py` -> `per_seed.csv`; `lr_at_best.py` -> `lr_at_best.csv`).

## 1. Setup of the final ("definitive") pool [files]

Pool definition: `/storage3/DSIP/rriva/research/fm4tag/scripts/configs/val_loss_ce_seed_curves_cmap.yaml` (committed a14c0e7, 2026-10-12;
header "DEFINITIVE pool, regenerated after every fm4tag run finished (2026-09-30)"). 67 trainings, all `status: completed` in run_info.yaml.
Run dirs under `/storage3/DSIP/rriva/research/fm4tag/slurm/classify_param_scan/`.

Common to all 67 (checked in each `.hydra/config.yaml`):
- Encoders `rowcol_v0`: global (jet) encoder dim 16, proj_out 32; track encoder dim 16, proj_out 128; rowcol layer depth 6, col_heads 2,
  row_heads 2, dim_head 8, dim_row_head 8, ff_mult 1, num_inds 16 (ISAB inducing points), row_mode per_token.
- Aggregator TransformerAggregator depth 6, heads 4, dim_head 64, ff_mult 4. Aggregator output z_jet = 32 + 128 = 160 dims.
- Head MultiStreamClassifierHead mlp_hidden 256, mlp_dropout 0.02. 2.2 M trainable params (rank_0.err ModelSummary).
- AdamW lr 3e-4, weight decay 5.96e-5; per-step cosine schedule over `max_epochs` (finetune warmup_frac 0.0 -> 1 warm-up step);
  bf16-mixed; no class weights.
- Early stopping AND ModelCheckpoint monitor `val/head/loss_ce` (callbacks_finetune), patience 15 (patience lowered to 12 only in
  commit 2a1f74f on 2026-09-28, after all pool runs started).
- Loss = 1.0 * CE + w_JC * L_JC; w_JC = 0.3 for every lambda point ("alpha_Agg = 0.3" in plot titles), 0 for baselines.
- L_JC = `MultiViewSupConLoss` (used label-free, i.e. InfoNCE/NT-Xent), temperature 0.1, loss_type out, include_pos_in_denom true,
  include_same_view_negatives false, on z_jet; embeddings all-gathered across DDP ranks (negatives = global batch).
- Two views: view 1 = Compose[CutMix(lam), Mixup(lam)], view 2 = Identity (reuses the clean pass). CutMix: each raw feature cell
  replaced with prob lam by the same feature of a random other row of the batch; Mixup (embedding stage):
  x' = (1 - lam) x + lam x_perm. So lambda >= 0.5 makes the augmented view closer to the partner than to itself in Mixup.
- CE is always computed on the CLEAN (un-augmented) forward pass, for train and val.
- Data: train 51,999,998 jets / val 4,999,997 / test 5,000,005 (`.../transforming-jet-flavor/data_parallel_preprocess_251112_112022`),
  up to 40 tracks per jet, inputs: jet pt_btagJes, eta_btagJes + 10 continuous and 9 categorical track features.

Three batch regimes (datamodule.batch_size is PER RANK; row attention and the contrastive negatives depend on it):
| regime | per-rank bs x GPUs | gradient batch / JC negatives pool | run dirs | max_epochs |
|---|---|---|---|---|
| A | 1024 x 2 | 2048 | run_20260819_134452, _135557, _135709 (50 ep); run_20260828_153214, _163630 (100 ep) | 50 or 100 |
| B | 2048 x 1 | 2048 | run_20260903_155735 | 100 |
| C | 2048 x 2 | 4096 | run_20260915_133955, run_20260915_134011 (dropout baseline), run_20260922_094553, _095140 | 100 |

Seeds: canonical list 42 424 4242 42424 424242 4242424 42424242 424242424, run in order until the 7-day walltime runs out.

Exclusions (as defined by the author in the config):
- Stuck rule (author, 2026-10-01): val CE never <= 0.610 within the first 20 epochs, applied to scan AND baselines. 6 runs:
  no-JC seeds 424, 42424, 424242 (run_20260819_134452; plateau ~0.614, best 0.61263/0.61310/0.61258);
  lambda 0.4 run_20260828_153214 seed_42 (best 0.62015 @ep98); lambda 0.7 run_20260922_095140 seed_424 (0.60725 @ep98, first <=0.610 at ep37);
  lambda 0.8 run_20260915_133955 seed_424 (0.60804 @ep72, first <=0.610 at ep36).
- Baseline outlier cut (author, 2026-10-01): baseline seeds with best > 0.606 dropped from the jc=0 points only:
  no-JC seed 42424242 (0.60613), dropout seed 42424 (0.60745). Lambda points are not cut.
- 50-epoch-capped seeds (author, 2026-09-18): run_20260819_135557 lambda 0.4 seed_42 (best 0.60057 @ep48) and run_20260819_135709
  lambda 0.5 seed_42 (0.60087 @ep49) are NOT in the pool.
- Architecture variant F2@d6 (run_20260915_134020, run_20260922_095153; 5.5 M params): 3 seeds 0.61066 / 0.60334 / 0.60768, mean 0.60723 +- 0.00368.
- Failed runs run_20260915_110720 (GPU-contention OOM incident, rc 134), unfinished pilots, and run_20260818_172348 seed_42.
- Focus plot (overlay `val_loss_ce_best_vs_lambda_no_dropout_min2.yaml`, author 2026-10-11): also hides the dropout baseline and
  lambdas with < 2 usable seeds (0.64, 0.7, 0.75, 0.8) -> 49 trainings (4 no-JC + 45 CL).

## 2. Validation results (metric = per-seed minimum over epochs of val/head/loss_ce, unweighted CE on the 5M-jet val set) [files]

Recomputed from every metrics.csv; matches `/storage3/DSIP/rriva/research/fm4tag/plots/val_loss_ce/val_loss_ce_best_vs_lambda_no_dropout_min2_summary.csv` to all printed digits.

| config | regime(s) | n used (stuck/outlier removed) | mean best val CE +- std | min | best epochs | val-train gap at last epoch |
|---|---|---|---|---|---|---|
| no JC (CE only) | A50 | 4 (3 stuck, 1 outlier) | 0.60345 +- 0.00049 | 0.60298 | 5,7,7,10 | +0.045 |
| no JC + dropout 0.1 (not in focus plot) | C100 | 6 (1 outlier) | 0.60162 +- 0.00096 | 0.59977 | 10-35 | +0.021 |
| lambda 0.05 | A50 | 6 | 0.60742 +- 0.00047 | 0.60676 | 7-11 | +0.043 |
| 0.1 | A50 | 5 | 0.60620 +- 0.00035 | 0.60581 | 9-14 | +0.039 |
| 0.2 | A100, B | 3 | 0.60461 +- 0.00019 | 0.60443 | 17,17,17 | +0.029 |
| 0.25 | A50 | 4 | 0.60243 +- 0.00023 | 0.60210 | 17-20 | +0.028 |
| 0.3 | A50, B | 5 | 0.60206 +- 0.00038 | 0.60165 | 20-27 | +0.026 |
| 0.35 | A50, A100, B | 5 | 0.60207 +- 0.00059 | 0.60136 | 22-32 | +0.021 |
| 0.4 | B, C | 3 (1 stuck) | 0.60179 +- 0.00047 | 0.60124 | 27,48,39 | +0.016 |
| 0.45 | B, C | 2 | 0.60145 +- 0.00041 | 0.60116 | 48,72 | +0.019 |
| 0.5 | A100, B, C | 5 | 0.60231 +- 0.00032 | 0.60197 | 36-46 | +0.017 |
| 0.55 | B, C | 2 | 0.59987 +- 0.00081 | 0.59929 | 62,98 | +0.016 |
| 0.6 | A100, B, C | 3 | 0.59993 +- 0.00078 | 0.59905 | 48,62,95 | +0.015 |
| 0.64 | C | 1 | 0.59921 | - | 99 | +0.014 |
| 0.65 | B, C | 2 | 0.59973 +- 0.00053 | 0.59936 | 62,71 | +0.016 |
| 0.7 | C | 1 (1 stuck) | 0.59953 | - | 99 | +0.015 |
| 0.75 | C | 1 | 0.59972 | - | 73 | +0.016 |
| 0.8 | C | 1 (1 stuck) | 0.60012 | - | 61 | +0.018 |

- Val-train CE gap at the BEST epoch is +0.010..+0.016 for every config; the difference is what happens after: CE-only val CE rises by
  +0.0093 on average by the early stop (train 0.568 vs val 0.613 at ep22 for the best no-JC seed), lambda>=0.55 rises by <= 0.0004.
  Best epoch moves from ~8 (no JC, lambda 0.05) to 60-99 (lambda >= 0.45): augmentation strength delays overfitting.
- Regime-pure comparison (A, 50 epochs, same schedule): no JC 0.60345 +- 0.00049 (n=4) vs lambda 0.25 0.60243 +- 0.00023 (n=4),
  lambda 0.3 (A only) 0.60192 +- 0.00026 (n=4), lambda 0.35 (A50) 0.60153 (n=2); lambda 0.05 / 0.1 are WORSE than no JC.
- Stuck rate: 3/8 no-JC seeds vs 3/52 CL seeds (1/29 within regime A); one-sided Fisher p ~ 0.026 within regime A [inferred].
- JC-weight check at lambda 0.1 (regime A, run_20260819_134452): jc 0 = 0.60345 +- 0.00049 (n=4), jc 0.3 = 0.60620 +- 0.00035 (n=5),
  jc 1.0 = 0.60515 +- 0.00024 (n=4, not in the pool): non-monotonic in jc; at lambda 0.1 any JC hurts val CE.
- Learning-rate confound [files+inferred]: LR at the best epoch is ~2.7-2.9e-4 (still at peak) for no-JC and lambda <= 0.1, but ~0
  (<1.2e-6) for the best seeds at lambda 0.55/0.6/0.64/0.7. Spearman(LR at best epoch, best CE) = +0.91 over 51 healthy CL runs.
  At FIXED lambda the two excluded 50-epoch-capped seeds (cosine fully annealed, LR 3e-7 / 5e-16) reach 0.60057 (lambda 0.4) and
  0.60087 (lambda 0.5), 0.0007-0.0019 below all their 100-epoch, early-stopped siblings (0.60124-0.60274).

## 3. Test-set comparison with GN2 [files]

- Reference: "GN2 only-JT", trained by the author in salt (`/storage3/DSIP/rriva/packages/salt-pt/salt/logs/GN2_only_jt_20260817-T134649/`)
  on the SAME train/val files, jet-classification task only (no track-origin / vertexing aux tasks; "GN2 open data" = with aux tasks,
  used only in Phase 0), 4-layer transformer embed 256, batch 3000 x 2 GPUs, max 50 epochs, best epoch 15, class-weighted CE
  (weights b 1.5, c 1.5, u 1.0, tau 4.0) -> its val_loss 0.59814 is NOT comparable to fm4tag's unweighted val CE.
  Eval file `.../ckpts/epoch=015-val_loss=0.59814__test_pp_output_test.h5` on the same test h5.
- Selection: one model per lambda = the seed with the lowest best val CE, at its pinned best-CE checkpoint
  (`/storage3/DSIP/rriva/research/fm4tag/scripts/configs/fraction_constrained_lambda_scan_final.yaml`, committed 3ab272f).
- Cuts 20 < pT < 250 GeV, |eta| < 2.5: 4,998,732 test jets (b 1,290,091; c 1,289,789; u 1,934,992; tau 483,860).
- D_b = log[p_b / (f_c p_c + f_tau p_tau + (1 - f_c - f_tau) p_u)]. Final metric: c-rejection at 80% b-efficiency, maximised over
  (f_c, f_tau) on a grid f_c in [0, 0.598], f_tau in [0, 0.148], step 0.002 (22,500 points) subject to u-rej >= 1.0 x GN2 (171.117) and
  tau-rej >= 1.0 x GN2 (37.908); GN2 at nominal f_c = 0.2, f_tau = 0.01 (c-rej 10.348). Floors 1.0/1.0 set by the author 2026-10-01
  (an earlier 0.95/0.90 run was superseded). WP 0.80 (the Aug/Sep campaign used 0.77).
- Fractions were optimised ON THE TEST SET (fraction_constrained_search.py reads the runs' test_dataset_path; test_file: null) and
  the ROC is evaluated on the same jets. Re-derived from the stored grids for no JC, 0.05, 0.4, 0.55, 0.6, 0.64: all match.
- Binomial stat. error on the test set at 80% WP [inferred]: c-rej +-0.27% (+-0.028), u-rej +-0.94% (+-1.6), tau-rej +-0.87% (+-0.33).

| model | best val CE | f_c | f_tau | c-rej (opt) | vs GN2 | c-rej (nominal 0.2/0.01) | u-rej nominal | tau-rej nominal |
|---|---|---|---|---|---|---|---|---|
| GN2 only-JT | (0.59814 weighted) | 0.2 | 0.01 | 10.348 | - | 10.348 | 171.1 | 37.91 |
| no JC | 0.60298 | 0.298 | 0.064 | 10.122 | -2.18% | 9.654 | 212.2 | 25.13 |
| 0.05 | 0.60676 | 0.260 | 0.070 | 9.770 | -5.59% | 9.462 | 199.8 | 23.28 |
| 0.1 | 0.60581 | 0.242 | 0.062 | 9.979 | -3.57% | 9.764 | 192.1 | 23.72 |
| 0.2 | 0.60443 | 0.256 | 0.046 | 10.035 | -3.02% | 9.801 | 195.9 | 26.07 |
| 0.25 | 0.60210 | 0.256 | 0.046 | 10.215 | -1.28% | 9.954 | 196.3 | 26.70 |
| 0.3 | 0.60165 | 0.302 | 0.050 | 10.385 | +0.36% | 9.936 | 211.5 | 25.74 |
| 0.35 | 0.60136 | 0.262 | 0.052 | 10.393 | +0.43% | 10.106 | 199.0 | 26.69 |
| 0.4 | 0.60124 | 0.310 | 0.032 | 10.350 | +0.02% | 9.883 | 211.0 | 30.96 |
| 0.45 | 0.60116 | 0.268 | 0.050 | 10.405 | +0.55% | 10.077 | 201.8 | 26.57 |
| 0.5 | 0.60197 | 0.294 | 0.032 | 10.279 | -0.67% | 9.887 | 206.6 | 30.30 |
| 0.55 | 0.59929 | 0.306 | 0.046 | 10.580 | +2.24% | 10.151 | 213.6 | 27.27 |
| 0.6 | 0.59905 | 0.306 | 0.044 | 10.561 | +2.06% | 10.138 | 210.7 | 27.15 |
| 0.64 | 0.59921 | 0.310 | 0.046 | 10.610 | +2.53% | 10.155 | 214.2 | 27.44 |
| 0.65 | 0.59936 | 0.316 | 0.048 | 10.566 | +2.11% | 10.092 | 216.3 | 27.47 |
| 0.7 | 0.59953 | 0.304 | 0.046 | 10.544 | +1.90% | 10.114 | 210.2 | 27.84 |
| 0.75 | 0.59972 | 0.318 | 0.048 | 10.524 | +1.70% | 10.016 | 215.4 | 26.66 |
| 0.8 | 0.60012 | 0.272 | 0.036 | 10.481 | +1.29% | 10.184 | 200.5 | 30.49 |

- Spearman(best val CE of the picked seed, optimised c-rej) = -0.978 over 17 models [inferred]: val CE is an excellent proxy.
- At nominal fractions no fm4tag model beats GN2 in c-rej; all have far higher light rejection (192-216 vs 171) and far lower
  tau rejection (23-31 vs 38). The "beat GN2" statement relies on re-tuning f_c, f_tau (f_c 0.24-0.32, f_tau 0.03-0.07).
- At 77% (fractions tuned at 80%): lambda 0.55 c-rej 15.461 vs GN2 15.096 (+2.4%), no JC 14.684 (-2.7%); some models then miss the
  GN2 light floor (e.g. lambda 0.2 u-rej 280.2 < 284.5).
- Phase 0 (2026-08-18, single seeds, WP 0.77, coarse grid, test set): per_token no-JC lr 3e-4 15.25 vs GN2 15.10;
  JC 0.3/1.0/3.0 at lambda 0.1: 15.02/14.95/14.63; concat (jc 1.0) best 14.65 @lambda 0.25; concat lambda 0.75 / 0.9 diverged
  (c-rej 1.99 / 2.43; best val CE 1.131 / 1.052). (`/storage3/DSIP/rriva/research/fm4tag/scripts/plots/rejection_summary_phase0/summary.csv`)

## 4. Embedding quality [files]

Metrics (`src/fm4tag/metrics/`): uniformity = log mean exp(-2 ||z_i - z_j||^2) of L2-normalised z (Wang & Isola; <= 4096 sampled;
lower = more uniform, about -4 is the floor for near-orthogonal vectors); effective rank = exp(entropy of normalised singular values)
of the mean-centred z (Roy & Vetterli; range 1..160). Logged by the EmbeddingMetrics callback on <= 8192 VALIDATION jets per epoch,
for z_jet = aggregator output (column val/aggregator/*).

At each seed's best epoch: no JC uniformity -1.63 (-2.06..-1.29), eff. rank 96.7 (93.2..98.8); dropout baseline -1.95, 101.2;
every CL run -3.94..-3.89, eff. rank 121-148, decreasing with lambda (0.05: 145.9 -> 0.3: 128.6 -> 0.6: 124.5 -> 0.8: 121.0),
anomaly lambda 0.5: 143.7 in all 5 seeds and 3 regimes. Among CL runs Spearman(eff. rank, best CE) = +0.88 (+0.95 without lambda 0.5):
lower rank and slightly less uniform go with better CE; JC itself raises both far above CE-only.
Per-epoch (plot_ce_and_embedding): CE-only best seed uniformity -2.43 (ep0) -> -1.20 (ep22) and rank 80.6 -> 106.4 while val CE
overfits after ep7; CL lambda 0.6 seed 42 stays at -3.85..-3.91, rank 115.5..126.2; ep25 event (val CE 0.6048 -> 0.6093 -> 0.6037,
uniformity -3.878, rank dip to 115.5) is real data.

## 5. Permutation importance [files] (old architecture)

`/storage3/DSIP/rriva/research/fm4tag/scripts/plots/permutation_importance_20260820_081655/` on the OLD jc=1.0 lambda scan
(run_20260811_105149: concat row mode, num_inds 128/256, ff_mult 2, single seed 42), lambda 0.15-0.5, 200k val jets, 5 repeats,
metric val CE, pooled over valid tracks, paired shuffles. Ranking stable (Spearman vs lambda 0.15 >= 0.977); dphi > deta > d0 ~
lifetimeSignedD0Significance > qOverP, z0; hit-pattern group dCE 0.019-0.032. Profile flattens with lambda: dphi share 0.330 -> 0.181,
measurement-uncertainty group dCE 0.071 -> 0.228, momentum group 0.047 -> 0.196 (lambda 0.15 -> 0.5); impact-parameter group ~0.78-0.87;
jet pT 0.13-0.29, jet eta ~0. Never repeated on the final per_token jc=0.3 ladder.

## 6. Paper-ready figures (all exist; commands in /storage3/DSIP/rriva/research/fm4tag/plot_paths.txt)

Base config B = scripts/configs/val_loss_ce_seed_curves_cmap.yaml; overlay O1 = val_loss_ce_best_vs_lambda_no_dropout_min2.yaml;
O2 = val_loss_ce_best_vs_lambda_categorical_colours.yaml; O3 = ce_and_embedding_curves.yaml (all in scripts/configs/).
- Mean best val CE vs lambda (focus plot, categorical): /storage3/DSIP/rriva/research/fm4tag/plots/val_loss_ce/val_loss_ce_best_vs_lambda_no_dropout_min2_categorical.png (B+O1+O2; plot_best_vs_lambda.py) + _summary.csv
- Same, gradient colours: /storage3/DSIP/rriva/research/fm4tag/plots/val_loss_ce/val_loss_ce_best_vs_lambda_no_dropout_min2.png (B+O1)
- All 16 lambdas + both baselines: /storage3/DSIP/rriva/research/fm4tag/plots/val_loss_ce/val_loss_ce_best_vs_lambda_all.png (B)
- Per-seed val CE curves with stars, focus pool (49 runs): /storage3/DSIP/rriva/research/fm4tag/plots/val_loss_ce/val_loss_ce_best_vs_lambda_no_dropout_min2_categorical_curves.png (B+O1+O2); gradient: ..._no_dropout_min2_curves.png
- Full-pool curves / markers: /storage3/DSIP/rriva/research/fm4tag/plots/val_loss_ce/val_loss_ce_seed_curves_cmap.png and _cmap_markers.png (plot_seed_curves.py, B); categorical val_loss_ce_seed_curves.png (val_loss_ce_seed_curves.yaml); xlim80/ variants
- ROC, 13 models + GN2 (categorical, c/light/tau panels, own optimal fractions): /storage3/DSIP/rriva/research/fm4tag/scripts/plots/run_20261001_110047_lambda_scan_final_wp080_u100_tau100/categorical_no_dropout_min2/roc_no_dropout_min2.png (its roc_comparison_config.yaml)
- ROC reduced, latest (no JC black, GN2 grey, lambda 0.6 blue, 0.55 green, 0.05 red; WP line 0.8; 600 dpi; no "Simulation Internal"; 2026-11-10): /storage3/DSIP/rriva/research/fm4tag/scripts/plots/run_20261001_110047_lambda_scan_final_wp080_u100_tau100/categorical_no_dropout_min2/ce_only_gn2_best_worst_valce_600dpi/roc_ce_only_gn2_best_second_worst_valce_colours_600dpi.png
- ROC 17 models, optimal fractions (older layout): .../run_20261001_110047_lambda_scan_final_wp080_u100_tau100/roc.png (+ optimal_fractions.csv, working_points.csv)
- ROC 17 models at nominal fractions: /storage3/DSIP/rriva/research/fm4tag/scripts/plots/run_20260930_152803_lambda_scan_final_wp080/nominal/roc.png (+ working_points.csv)
- CE + embedding curves: /storage3/DSIP/rriva/research/fm4tag/plots/ce_vs_embedding/{ce_only,cl}_{uniformity,effective_rank}.png (B+O1+O3; plot_ce_and_embedding.py) + run_selection.csv
- Permutation importance (old arch): /storage3/DSIP/rriva/research/fm4tag/scripts/plots/permutation_importance_20260820_081655/{profile,heatmap,drift}.png
- Phase 0: /storage3/DSIP/rriva/research/fm4tag/scripts/plots/rejection_summary_phase0/{crej_vs_jc,crej_vs_lambda}.png

## 7. Inconsistencies / referee risks
1. Lambda is confounded with batch regime (A/B/C: gradient batch 2048 vs 4096, row-attention context 1024 vs 2048) and with
   max_epochs (50 vs 100): lambda <= 0.25 is A-only, lambda >= 0.64 is C-only; at lambda 0.55-0.65 C is 0.0007-0.0012 below B.
2. LR-schedule confound: cosine LR over max_epochs + early stopping means low-lambda runs stop at peak LR, high-lambda best seeds at
   LR ~ 0; at fixed lambda 0.4/0.5 the annealed (excluded) seeds are 0.0007-0.0019 better. Part of the lambda gain may be annealing.
3. The two excluded 50-epoch-capped seeds were excluded as "not converged", but their schedules were complete (LR -> 0) and they
   are the best seeds at their lambda; excluding them makes lambda 0.4/0.5 look worse.
4. No-JC (no dropout) baseline exists only in regime A/50 epochs; the regime-C baseline is no-JC + dropout 0.1 (0.60162 +- 0.00096),
   which matches CL at lambda 0.3-0.5; only lambda >= 0.55 beats it. Dropout baseline was never evaluated on the test set.
5. Fractions optimised on the test set and evaluated on the same set; GN2 kept at nominal fractions (its own constrained optimum at
   80% not checked; at 77% Phase 0 found nominal = optimum on a coarse grid). One (best-of-n) seed per lambda in the ROC: selection
   bias depends on n (6 seeds at 0.05, 1-2 at >= 0.45); seed spread of c-rej never measured.
6. Baseline outlier cut (> 0.606) applied to baselines only (conservative against JC, stated as such).
7. Memory note "lambda >= 0.75 diverges" holds only for the old concat jc=1.0 setup; in the final per_token jc=0.3 ladder lambda 0.75
   and 0.8 train (0.59972, 0.60012) but with more stuck seeds.
8. Pool config gives the reason for excluding run_20260818_172348 seed_42 as "same config AND seed" as run_20260819_134452 seed_42,
   but that run used the drifted architecture (num_inds 128, ff_mult 2): exclusion right, stated reason wrong.
9. Plot tag says sqrt(s) = 13 TeV; the poster abstract says the JetSet sample is 13.6 TeV.
10. Campaign memory used WP 0.77 throughout August/September; the final ROC and fraction search use WP 0.80.
11. GN2's val_loss 0.59814 is class-weighted; never quote it against fm4tag's unweighted 0.599.
12. Eff. rank anomaly at lambda = 0.5 (143.7 vs ~126 at neighbours) in all seeds/regimes: unexplained.

## 8. Open items
- Same-regime, same-schedule comparison (no JC and several lambdas at one batch regime, fixed-length schedule or LR annealed before stop).
- Seed spread of the test c-rejection (rerun the fraction search for all pool seeds; caches exist for best seeds only).
- Fractions re-derived on the validation split, then applied to test; GN2 optimised under the same constraints.
- More seeds at lambda >= 0.45 (n = 1-2) and runs beyond 100 epochs (seeds at 0.55/0.6/0.64/0.7 still improving at ep 95-99).
- Permutation importance on the final per_token jc=0.3 ladder; test-set evaluation of the dropout baseline.
