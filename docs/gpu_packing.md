# Running several trainings on one GPU (shared-card packs)

## Why

Two cluster rules pull in opposite directions:

- A user may have at most **10 GPU jobs**, running plus pending. A job counts as one whether it asks for 4 shards or 2
  whole GPUs.
- A job that holds a whole GPU is expected to **fill its memory** (44.4 GiB on an A40 or L40S).

Trainings that must all use the same batch size cannot fill the card by raising the batch. At batch 12800 a
cross-entropy-only training needs roughly half a card and a frozen-encoder training much less.

The answer used here: the job still asks for **whole GPUs** (2 per job, as always), and **several independent trainings
run at the same time on each card**, as many as fit its memory. The card is full, the job limit is spent well, and no
other user shares the card (which is the risk with shards: they enforce no memory limit).

## Where the code is

| File | Role |
|---|---|
| `slurm/ssl_pack.sh` | The launcher. Reads the cell file, writes the task list and the job script, submits the pack jobs. `SLOTS_PER_GPU` and `GPU_BUDGET_GIB` switch the sharing on. |
| `slurm/pack_gpu.sh` | Runs once per GPU of a job. Starts `SLOTS_PER_GPU` workers on that card and gives them a shared ledger. |
| `slurm/pack_worker.sh` | One worker = one training at a time. Takes the next task from the stage's queue, checks the ledger, runs the training, repeats. |
| `slurm/gpu_preflight.sh` | Unchanged. Before any training starts, waits until the allocated cards are really empty. |

With `SLOTS_PER_GPU=1` (the default) nothing changes: one worker per card, no ledger, the same job script as before.

## How it works

```
pack job (sbatch, --gres=gpu:2)
├── srun step, GPU A ── pack_gpu.sh ──┬── worker w0s0 ── training, 19 GiB ┐
│                                     ├── worker w0s1 ── training, 19 GiB ├─ ledger of GPU A: 38 of 40 GiB
│                                     └── worker w0s2 ── waits (card full)┘
└── srun step, GPU B ── pack_gpu.sh ──┬── worker w1s0 ── training, 38 GiB ── ledger of GPU B: 38 of 40 GiB
                                      └── ...
```

1. **Each cell declares its GPU memory.** In the cell file, `@mem=19` says a training of that cell peaks at about
   19 GiB. The launcher writes this per task to `<stage dir>/task_mem.tsv`. A cell without `@mem` is given the whole
   budget, so it runs alone on its card.
2. **Slurm assigns the cards.** The job asks for `--gres=gpu:2`; each card gets its own `srun --gres=gpu:1` step. The
   scripts never choose a GPU index themselves.
3. **Each card gets several workers.** `pack_gpu.sh` starts `SLOTS_PER_GPU` workers inside the step. They all see the
   same card. The step is given the CPUs of all of them (each training keeps its own data-loader workers).
4. **The ledger keeps the card within budget.** The workers of one card share a directory on the node
   (`/tmp/fm4tag_pack_<job id>_<gpu>/`). A running training has one file there holding its `@mem`. Before a worker
   claims a task it takes a lock, sums the files, and starts only if `sum + @mem <= GPU_BUDGET_GIB` (default 40).
   Otherwise it tries the next task in the list, and if none fits it waits and looks again every 2 minutes.
   When the training ends, for any reason, its file is removed.
5. **A task alone on a card always starts**, even if its `@mem` is above the budget. This is how a full-card training
   (cross entropy + jet contrastive, about 38 GiB) runs in the same stage as small ones.
6. **The task queue is unchanged.** Tasks are still claimed with an atomic `mkdir` under `<stage dir>/.queue/`, so two
   workers, on the same card or on different nodes, never run the same task. A claim whose job died is released.

The ledger is per job and per card: the two cards of a job do not count each other's trainings, and two jobs never
share a card.

## How to use it

```bash
# cell file: one line per cell, with the two estimates measured beforehand
#   sc_wsd   arm=scratch ... datamodule.batch_size=12800 ... @est=1900 @mem=19
#   fr_p0    arm=frozen  ... datamodule.batch_size=12800 ... @est=900  @mem=5

SLOTS_PER_GPU=3 GPU_BUDGET_GIB=40 GPU_NODE=gpu-L40S-open MEM_PER_GPU_GB=60 \
CELLS=scripts/configs/campaign/<stage>.cells SEEDS="42" PACKS=4 \
MAX_EPOCHS=40 EST_SECONDS_PER_EPOCH=2000 bash slurm/ssl_pack.sh
```

- `SLOTS_PER_GPU`: the most trainings a card runs at once. The memory budget decides how many actually run.
- `GPU_BUDGET_GIB`: the sum of `@mem` allowed on a card. 40 leaves about 4 GiB of margin on a 44.4 GiB card.
- `MEM_PER_GPU_GB`: host RAM per card, now shared by all its trainings, so raise it with the slots.
- `DRY_RUN=1` writes the stage directory and submits nothing; the task list shows each task's GiB.

Each worker logs what it reserved when it starts a training:

```
[w0s1] sc_wsd__s42: start (max_time ..., 19 GiB of 38 reserved on the card, budget 40)
```

## What to keep in mind

- **Sharing a card does not make trainings faster.** There is no MPS on this cluster, so the trainings on a card take
  turns on its compute. Two side by side each run slower than one alone; the total throughput of the card is about the
  same. `@est` (seconds per epoch) must therefore be the time measured **while sharing**, or the walltime check is wrong.
- **`@mem` is an estimate, and nothing enforces it.** If a training uses more than its `@mem`, the card can run out of
  memory and every training on it can crash. Measure the peak (`mem/gpu_peak_MiB` in `metrics.csv`) with a short run at
  the real batch size, and round up.
- **CPUs limit the slots.** A training needs its loader workers plus about 2 CPUs (8 CPUs at the default 6 workers).
  The L40S nodes have 24 CPUs per GPU, enough for 3 trainings per card. node82 (A40) has 10 per GPU: use
  `GPU_NODE=gpu-L40S-open` for shared-card stages, or lower `NUM_WORKERS`.
- **A large task waits for an empty card.** Small tasks can keep a card busy, so a full-card task starts only when its
  card has emptied. Putting the large cells in their own stage (`CELL_FILTER`) avoids the wait.
- **Order in the cell file is the priority**, as before: longest cells first.

## How it was checked

- A local test with a fake training command (no GPU): 8 tasks of 5 to 45 GiB on one card with 3 workers and a budget
  of 40. Every task ran once, the reserved sum of trainings sharing the card never passed 40 (largest 38), and the
  45 GiB task ran alone.
- The launcher's dry run with `SLOTS_PER_GPU=1` writes the same job script as the stage that was running before the
  change.
- **Not yet run on a real GPU** as of 2026-11-16. The first real use is the memory and speed measurement at batch
  12800, which also provides the `@mem` and `@est` values.
