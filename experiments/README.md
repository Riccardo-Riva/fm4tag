# Experiments

Every experiment has its own folder `experiments/<yyyy-mm>_<name>/` with everything that belongs to it: `README.md`
(question, data, **model**, **training methods**, stages, results, related experiments), `configs/` (cells files and
analysis configs), `slurm/` (its own launchers), `notes/`, and the git-ignored `outputs/` (run folders, logs) and
`plots/`. Generic tools stay outside: `slurm/ssl_pack.sh` (+ `pack_worker.sh`, `pack_gpu.sh`, `gpu_preflight.sh`) runs
a cells file and puts the stage into the `outputs/` of the experiment whose `configs/` holds it; `scripts/` holds the
Python tools and the config-driven `scripts/submit_*.sh`, which write to the experiment of their config;
`slurm/plot_job.sh` renders plots on a CPU node. To start a new experiment, copy `_template/`.

| Experiment | Dates | Question | Code | Status |
|---|---|---|---|---|
| [`2026-03_early-runs`](2026-03_early-runs/README.md) | 2026-03 → 07 | first pretrain / fine-tune / scratch tests | before `exp/lambda-scan-final` | closed |
| [`2026-07_ce-jc-grids`](2026-07_ce-jc-grids/README.md) | 2026-07 → 08 | lr x JC weight, concat vs per_token, first λ sweeps (from scratch) | `exp/lambda-scan-final` | closed |
| [`2026-07_kernel-profiling`](2026-07_kernel-profiling/README.md) | 2026-07 → 10 | attention-kernel speed / memory, S0 benchmark | various, `archive/s0-snapshot` | closed |
| [`2026-08_lambda-scan`](2026-08_lambda-scan/README.md) | 2026-08-18 → 10-12 | CE + JC at noise λ vs no-JC and GN2 (52 M jets) | `exp/lambda-scan-final` | closed |
| [`2026-10_ssl-campaign`](2026-10_ssl-campaign/README.md) | 2026-10-16 → 11-12 | does contrastive pretraining (P0 / P1 / P2) help SC / FT / FR? (18.6 M jets) | `exp/ssl-wsd-lr-search` | closed |
| [`2026-11_dropout`](2026-11_dropout/README.md) | 2026-11-12 → 11-22 | dropout downstream (0.2) and in pretraining (0.1) | `exp/ssl-wsd-lr-search` | closed |
| [`2026-11_wsd-lr-search`](2026-11_wsd-lr-search/README.md) | 2026-11-16 → 11-23 | one batch size, WSD, no early stopping: lr per case (protocol v1) | `exp/ssl-wsd-lr-search` | closed |
| [`2026-11_four-way-jc-dropout`](2026-11_four-way-jc-dropout/README.md) | 2026-11-23 → | CE / CE+JC / CE+dropout / CE+JC+dropout, from scratch, class-weighted CE | `main` | running |
| [`2026-11_pretrain-v1`](2026-11_pretrain-v1/README.md) | 2026-11-23 → 11-25 | P0 / P1 pretrained again with the protocol-v1 settings (batch 12800, WSD 80 epochs, no early stopping) | `main` | cancelled (model B replaced by a small model) |
| [`2026-11_pretrain-saint-small`](2026-11_pretrain-saint-small/README.md) | 2026-11-25 → | P0 / P1 / P2 of small_RC (`saint_small_RC`, 3.5 M), small_C (`saint_small_C`, 2.1 M) and normal_C (`saint_normal_C`, 2.2 M), P_v1 recipe over 40 epochs, 3 seeds for now (5 in the end) | `main` | preparing: GPU memory estimate on CPU |
| GN2 on the same 18.6 M slice | 2026-11-06 | same-data GN2 reference (salt, CE only, 8 seeds) | salt-ririva `6e43f64` | done; outputs in `/storage3/DSIP/rriva/research/salt-ririva/slurm/scratch_lambda_scan/output/run_20261106_092332/` |

## Old paths

The reorganisation of 2026-11-23 is recorded in `docs/repo_inspection_2026-11-23.md` (§ 13) and
`scripts/maintenance/restructure_experiments_2026-11-23.py`. Run folders written before it record absolute paths of
the old layout in `.hydra/config.yaml` (for example the `encoder_ckpt` of fine-tuned runs). To keep them working,
`slurm/ssl_campaign/<stage>` is a symlink to the stage's new folder and `/storage3/DSIP/rriva/research/fm4tag-chunk`
is a symlink to this repository. Do not write new runs there.
