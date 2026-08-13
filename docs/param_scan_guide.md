# Parameter scans with seed repetitions

How to run [`slurm/classify_param_scan.sh`](../slurm/classify_param_scan.sh) and
plot its results with [`scripts/plot_val_curves.py`](../scripts/plot_val_curves.py)
(script) or [`notebooks/val_loss_curves.ipynb`](../notebooks/val_loss_curves.ipynb)
(interactive).

This is the seed-repeated sibling of
[`slurm/classify_from_scratch.sh`](../slurm/classify_from_scratch.sh): same
experiment (`phase=finetune`, `encoder_ckpt=null` — backbone + aggregator +
head all train jointly from random init), but instead of one job per
hyper-parameter combination, one job per combination trains **several seeds
back-to-back**, so a later plot can show mean ± spread instead of a single
sample that may be a lucky (or unlucky) seed.

## 1. What one job does

Each grid cell = one Slurm job = 2 L40S GPUs for up to 7 days. Inside the job,
a bash loop trains the *same* configuration once per seed, sequentially, over
both GPUs via DDP:

```
job (2 GPUs, 7 days)
└─ seed 42 [DDP-2] → seed 43 [DDP-2] → seed 44 [DDP-2] → … (until time runs out)
```

Why sequential DDP-2 rather than two independent single-GPU trainings on the
two GPUs: the dataloader is the bottleneck here, so a single-GPU training costs
almost as much GPU-time as a DDP-2 one but takes ~2x the wall-clock. Sequential
DDP-2 finishes a seed sooner, wastes less of the walltime on an unfinished
tail, and keeps the gradient batch (`GPU_NUM × batch_size` = 2048) identical to
the earlier `classify_from_scratch.sh` sweeps, so results are comparable.

**Seeding.** `seed=<n>` is a single top-level Hydra key. The runner passes it
to `L.seed_everything(workers=True)` (model init, augmentation RNG, dataloader
worker seeding) and it is interpolated into `datamodule.seed` (batch
composition/shuffling), so one override reseeds the entire training —
see [`src/fm4tag/runner/run.py`](../src/fm4tag/runner/run.py).

**Time budgeting.** The loop only starts a seed if the time left exceeds an
estimate of one training's duration (`EST_SECONDS_PER_EPOCH × MAX_EPOCHS +
startup`, seeded from a measured 50-epoch DDP-2 run and then only ever raised,
never lowered, by what earlier seeds of the *same* cell actually took). Each
training also gets `+trainer.max_time` set to the remaining budget, so the
last one ends gracefully — checkpoint written, `metrics.csv` flushed, wandb
synced — instead of being killed mid-epoch by the Slurm walltime.

## 2. Configuring a scan

Open the script and edit the block under `── Grid definitions ──`:

```bash
# Fixed hyper-parameters
TRANSFORMER_TYPE=rowcol_concat
BATCH_SIZE=1024
LR=3e-4
CROSS_ENTROPY=1.0
JET_CONTRASTIVE=1.0

# Scan axis A — any Hydra key, crossed with axis B
SCAN_PARAM="lambda"          # augmentation strength (CutMix/Mixup lam)
SCAN_LABEL="lam"
SCAN_VALUES=(0.1 0.2 0.25 0.3 0.35 0.4)

# Scan axis B — kept to one value unless you deliberately want a bigger grid
SCAN2_PARAM="finetune.jet_contrastive_loss.temperature"
SCAN2_LABEL="temp"
SCAN2_VALUES=(0.1)

SEEDS=(42 43 44 45 46 47 48 49)   # consumed in order until the walltime runs out
```

Number of jobs submitted = `|SCAN_VALUES| × |SCAN2_VALUES|`. Each additional
value on either axis is one more 2-GPU, 7-day job — mind the partition's
footprint (`gpu-L40S-open` has 16 GPUs total) before adding a second real axis.

To scan something else instead of `lambda` (e.g. learning rate), just repoint
axis A:

```bash
SCAN_PARAM="optimizer.lr"; SCAN_LABEL="lr"; SCAN_VALUES=(1e-4 3e-4 1e-3)
```

`EXTRA_OVERRIDES=(...)` appends verbatim Hydra overrides to every training,
e.g. `EXTRA_OVERRIDES=("loggers.wandb.log_model=false")` to stop pushing
checkpoints to wandb on a big sweep.

## 3. Submitting

```bash
cd /storage3/DSIP/rriva/research/fm4tag
DRY_RUN=1 bash slurm/classify_param_scan.sh    # write job scripts, submit nothing — inspect first
bash slurm/classify_param_scan.sh              # submit for real
```

This creates `slurm/classify_param_scan/run_<TIMESTAMP>/`:

```
run_<TS>/
  sweep_info.txt                 settings + cell list
  <cell_name>/                   one Slurm job (one grid cell)
    cell_job.sh                  the generated job script
    job_out.txt, job_err.txt     the seed loop's own log
    seed_<seed>/                 ONE training, self-contained
      run_info.yaml              status / elapsed / exit code / hyper-params
      rank_*.out, rank_*.err
      outputs/<date>/<time>/.hydra/config.yaml
      <experiment_name>/version_0/metrics.csv
```

`seed_<seed>/` has exactly the shape `plot_val_curves.py` expects from a run
directory, so it can be listed (or, better, its parent cell can be listed)
directly in a plotting config.

## 4. Monitoring and resuming

```bash
# progress across the whole sweep
grep -H '^status:' slurm/classify_param_scan/run_<TS>/*/seed_*/run_info.yaml

# a specific cell's own log (budget decisions, per-seed timing)
tail -f slurm/classify_param_scan/run_<TS>/<cell_name>/job_out.txt
```

`run_info.yaml` per seed:

| field | meaning |
|---|---|
| `status` | `completed` \| `truncated` (hit `max_time`) \| `failed` (non-zero exit) |
| `exit_code`, `elapsed_seconds`, `budget_seconds` | what happened and how long it had |
| `seed`, `cell`, the scanned params, `lr`, `batch_size`, `max_epochs` | for cross-checking against `.hydra/config.yaml` |

**Resubmitting a cell** (after a node failure, or just to add more seeds once
the walltime allows) reuses whatever is already there: seeds with
`status: completed` in `run_info.yaml` are skipped, so `sbatch cell_job.sh`
from inside the cell's directory picks up where it left off. `truncated` and
`failed` seeds are retried.

A training that fails in under 10 minutes (`MIN_SANE_SECONDS`) aborts the
whole cell rather than burning the remaining seeds on a repeated
configuration error.

## 5. Plotting

`plot_val_curves.py` overlays a per-epoch validation metric across several
**configurations**. Each entry in the YAML config can now be backed by more
than one run:

```yaml
models:
  - dir: /path/to/a/single/run                 # one run — old behaviour, no band
  - cell: /path/to/run_<TS>/<cell_name>         # every <cell>/seed_*/ under it
  - dirs: [/path/a, /path/b]                    # an explicit list of runs
```

For a `cell:` (or `dirs:`) entry with more than one run, the script:

- reads each seed's `metrics.csv`,
- averages them into one curve, restricted to the epochs **every** seed
  reached (so the mean is never built from a shrinking set of runs — that
  would put a kink in it exactly where a seed early-stops),
- draws a ±spread band around that mean,
- marks the star at the **mean over seeds of the best value each seed
  reached** (not the minimum of the mean curve — that would be biased low by
  whichever seed happens to dip lowest at a given epoch), with the spread as
  an error bar,
- writes `<plot>_summary.csv` with the numbers behind every star: `best_mean`,
  the spread, `best_min`/`best_max`, `best_epoch_mean`, and `best_per_seed`.

Config-level knobs:

```yaml
error: std          # std (sample stdev) | sem (std / sqrt(n)) | minmax (full range)
seed_status: [completed]   # only seeds with this run_info.yaml status are included

plot:
  band: true         # false → mean curves only, error bars kept
  band_alpha: 0.15
  show_n: true        # append "(n = k)" to the legend for aggregated entries
```

`error: minmax` is the more honest choice at the n = 2–4 seeds a single 7-day
job typically produces — a sample standard deviation from that few points is
itself noisy. Switch to `std`/`sem` once more seeds have landed.

`seed_status` defaults to `[completed]`, so a `truncated` seed (cut short by
the job's walltime, biased toward a worse best value) is silently excluded —
loosen it deliberately if you want to inspect truncated runs.

### Running it

```bash
# one-shot PNG + summary CSV
python scripts/plot_val_curves.py --config scripts/configs/val_loss_lambda_seed_scan.yaml

# override the output directory
python scripts/plot_val_curves.py --config ... --output plots/my_scan
```

[`scripts/configs/val_loss_lambda_seed_scan.yaml`](../scripts/configs/val_loss_lambda_seed_scan.yaml)
is a ready template for `classify_param_scan.sh`'s default λ scan — set
`paths.root` to your `run_<TS>` directory and drop any cell whose seeds
haven't landed yet (an entry with zero completed seeds is an error, not an
empty curve).

### Interactively

`notebooks/val_loss_curves.ipynb` imports `Entry`, `load_entry` and
`summarise` straight from `scripts/plot_val_curves.py` — only the legend
labelling and the actual `matplotlib` drawing stay in the notebook, so the two
cannot drift apart the way they used to. Edit `CONFIG_PATH` / `SCALE` in the
setup cell, run through, then tweak `ax` in place before saving.

## 6. Gotchas

- **Storage.** Each training checkpoints its top-3 + last (`ModelCheckpoint`),
  ~120 MB each — a full 6-cell × ~6-seed sweep is tens of GB on disk, and the
  same again pushed to wandb (`log_model: true`). Add
  `EXTRA_OVERRIDES=("loggers.wandb.log_model=false")` for a big sweep if you
  don't need the checkpoints in wandb.
- **Partition footprint.** Each cell holds 2 of the 16 GPUs in
  `gpu-L40S-open` for up to 7 days. A 6-cell sweep is 12/16 GPUs — check
  `squeue` for other users' pending jobs before submitting a wide scan.
- **Comparability.** Keep `GPU_NUM` (hence the gradient batch) and the
  partition fixed across an entire scan you intend to compare — DDP-2 vs.
  single-GPU, or L40S vs. A40, both change the numbers as well as the timing.
