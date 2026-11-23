# Experiments index

Where every experiment's outputs live, and what each folder is. All folders are git-ignored and live in this worktree
(`/storage3/DSIP/rriva/research/fm4tag`). Since 2026-11-23 they are gathered here; the SSL-campaign results came from
the former `fm4tag-chunk` worktree. `/storage3/DSIP/rriva/research/fm4tag-chunk` is now a **symlink to this
directory**, so the old absolute paths recorded inside run folders (`.hydra/config.yaml`, caches, wandb) keep
resolving. Code states: tags `exp/lambda-scan-final` and `exp/ssl-wsd-lr-search`, and `archive/*` for the deleted
branches. History and criticalities: `docs/repo_inspection_2026-11-23.md`.

## Campaigns

| Campaign | Dates | Question | Code | Outputs | Design / notes |
|---|---|---|---|---|---|
| First tests | 2026-03 → 06 | first pretrain / scratch / fine-tune / frozen runs of fm4tag | before `exp/lambda-scan-final` | `outputs/`, `slurm/old_tests/` | git log of that period |
| CE + JC grids | 2026-07 → 08 | lr × JC weight, concat JC × λ, JC = 1 λ sweep | `exp/lambda-scan-final` | `slurm/classify_from_scratch/run_<ts>/` | `docs/param_scan_guide.md` |
| λ (noise) scan, "beat GN2" | 2026-08-18 → 10-12 | does CE + JC at noise level λ beat no-JC and GN2? (52M jets, `251112_112022`) | `exp/lambda-scan-final` | `slurm/classify_param_scan/run_<ts>/`; plots `scripts/plots/lambda_seed_bestval_*`, `scripts/plots/run_<ts>_*`, `plots/{val_loss_ce,losses,uniformity,ce_vs_embedding*}` | `paper/factsheets/fm4tag_lambda_scan.md`, `plot_paths.txt` |
| Kernel profiling | 2026-07 → 10 | attention kernel speed / memory, S0 benchmark | various | `slurm/profiling/` (`s0_*` = S0 benchmark, tag `archive/s0-snapshot`) | memory note on kernel economics |
| SSL campaign | 2026-10-16 → 11-23 | does contrastive pretraining (P0/P1/P2) help scratch / fine-tuned / frozen training? (18.6M jets, `251112_152051`) | `exp/ssl-wsd-lr-search` | `slurm/ssl_campaign/<STAGE>_<ts>/<cell>/seed_<s>/`; plots `plots/{arm_comparison,arm_comparison_p1,p0_vs_p1,label_scan,pretrain_curves,dropout,wsd_lr_search}`; inference logs `slurm/arm_roc/` | `docs/ssl_campaign.md`, header of each `scripts/configs/campaign/<STAGE>.cells`, `paper/factsheets/ssl_campaign.md` |
| GN2 on the same 18.6M slice | 2026-11-06 | same-data GN2 reference (salt, CE only, 8 seeds) | salt-ririva `6e43f64` | `/storage3/DSIP/rriva/research/salt-ririva/slurm/scratch_lambda_scan/output/run_20261106_092332/` (not copied) | inspection report § 5.2 |

## SSL-campaign stages (`slurm/ssl_campaign/`)

The stage name is the first part of each folder name; the cells file of the same name holds the design in its header.

| Stage prefix | What |
|---|---|
| `smoke_*` | smoke tests of the launchers |
| `S1_arch_*` | per_token vs concat row attention × chunk 128/512 (from scratch) |
| `S2_*` | pretraining recipe grid (T × λ) judged by frozen probes; `S2_qk` = with QK-norm |
| `P_full_*` | the full 30-epoch pretrainings P0, P1, P2 (seeds 42, 424) used downstream |
| `S3_lr_*`, `S3_epochs_*`, `S3_lowlabel_*` | lr per downstream arm, 15-epoch cosine, 10 % labels (`*_ABORTED_*` = stopped attempts) |
| `F_final_*` | final runs of the 8 downstream arms (4 runs each) |
| `C_gap_*` | why the val CE differs from the λ scan (dataset size) |
| `F_scjc_clip_*`, `F_scjc_stab_*`, `F_p1_nojc_*` | from scratch + JC stabilisation; FT/FR from P1 with CE only |
| `L_lr_*`, `L_reg_*`, `L_scan_*` | label-fraction study (lr check, regularisation check, 1–30 % scan) |
| `D_drop_*` | the six P0 vs P1 cases with dropout 0.2, 80 epochs (+ `D_drop_sweep_*`, `D_drop_smoke_*`) |
| `P_drop_*` | P0/P1/P2 pretrained again with dropout 0.1 (`CHANGES.txt` = manual interventions) |
| `W_measure_*`, `W_share_measure_*` | memory / speed at batch 12800, alone and sharing a card |
| `W_lr_full_*`, `W_lr_shared_*` | WSD learning-rate search, 66 runs, batch 12800 |

Each stage folder holds `sweep_info.txt` (git hash, cells, seeds), `tasks.tsv`, `pack_*.out`, and per run
`run_info.yaml`, `.hydra/config.yaml`, `metrics.csv` and checkpoints.

## Adding a campaign

Add a row to the campaign table (question, dates, code tag, output folders, design doc) when the first stage of a new
campaign is launched, and a row to the stage table for every new stage prefix.
