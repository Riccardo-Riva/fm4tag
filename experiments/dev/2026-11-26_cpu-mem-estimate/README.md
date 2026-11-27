# CPU estimate of the GPU memory of the small_RC / small_C / normal_C pretrainings (2026-11-26)

## Question
How much GPU memory does each of the nine pretrainings of `experiments/2026-11_pretrain-saint-small/configs/`
(`P_small_RC.cells`, `P_small_C.cells`, `P_normal_C.cells`: P0 / P1 / P2 of each model) need at batch 12800? The
answer is the `@mem` of each cell, which decides how many runs share a card (their sum must stay within 40 GiB of the
44.4 GiB). None of these models has a GPU measurement yet.

## How
`mem_estimate_cpu.py`, one process per cell:
- composes the cell with its base config and builds the pretraining module as the runner does;
- takes the first validation batch of the real data (the same jets for every cell);
- runs one forward and backward pass of the training loss on CPU in float32;
- records the highest resident memory during the step, minus the level just before it (sampled every 20 ms).

The optimiser step is left out: AdamW's state for these models is under 0.2 GiB.

A CPU number is not a GPU number. The CPU uses float32 instead of bf16-mixed, and other attention kernels. So it is used
as a ratio against model B, whose GPU peaks were measured in P_full (`mem/gpu_peak_MiB`, batch 12288):
P0 32.3 GiB, P1 40.8 GiB, P2 34.1 GiB (stages P_full_20261023_085735 and P_full_20261026_094445).

    GPU estimate (cell, batch 12800) = GPU peak (model B, same loss) × CPU step (cell) / CPU step (model B) × 12800 / 12288

with both CPU steps measured at the same batch (2560). The method holds if the three model B ratios GPU / CPU agree.
Two cells are also measured at batch 1280, to check that the memory grows linearly with the batch.

Earlier submissions, cancelled (logs kept in `outputs/` with a `_CANCELLED` suffix):
- 7477911, after 20 s and before any result: it also measured an aggregator with `ff_mult` 2, which is not wanted;
- 7477912, after model B's P0 (34.1 GiB on CPU at batch 2560): the base configs were being renamed
  (`saint_reduced` → `saint_small_RC`, new `saint_small_C` and `saint_normal_C`), and the remaining cells read them.

Run 7477983 measures all of them under the new names.

    mkdir -p experiments/dev/2026-11-26_cpu-mem-estimate/outputs
    sbatch experiments/dev/2026-11-26_cpu-mem-estimate/mem_estimate_cpu.sh

## Result
**The estimate was wrong: 1.5–1.9× too low.** small_RC's real GPU peaks (stage `P_small_RC_20261126_123135`, first
epochs, `mem/gpu_peak_MiB`) are P0 16.6, P1 22.5, P2 15.3 GiB against 8.6 / 13.6 / 10.2 estimated. Two P1 runs on one
card ran out of memory. Do not reuse this method between models of different depth.

Why: on GPU, small_RC's P0 needs 51 % of normal_RC's; on CPU, 26 %. A large part of the GPU memory does not shrink with
the encoder's depth (embeddings, projectors, contrastive losses, the views), while on CPU each attention layer also
stores the full attention matrices that the GPU's flash kernels never build, which inflates the part that does shrink.
The two-model calibration (normal_RC) therefore overstated how much a shallower model saves. With two measured models
an affine fit (GPU = a + b · CPU) is possible, but it still assumes the CPU-only part scales alike in models with and
without row steps; the real numbers come from each stage's first epoch.

What follows is the original result, kept for the record.

Job 7477983 (commit 6e28201 configs, node1, 16 threads), `outputs/results_7477983.jsonl`. The calibration cells are
named `p*_B`: B = normal_RC (model B).

**CPU memory of one step** (GiB, batch 2560, 18 143 valid tracks):

| Model | P0 | P1 | P2 |
|---|---|---|---|
| normal_RC | 34.15 | 37.65 | 36.37 |
| small_RC | 8.71 | 10.71 | 9.34 |
| small_C | 3.51 | 5.45 | 4.13 |
| normal_C | 6.10 | 9.62 | 8.30 |

**Checks.**
- normal_RC's ratios GPU peak (batch 12288) / CPU step (batch 2560): P0 0.95, P1 1.08, P2 0.94. They agree within 15 %.
- Memory grows linearly with the batch: from 1280 to 2560 jets, ×2.13 (normal_RC P1) and ×2.07 (small_RC P1).
- The aggregator costs relatively more on GPU than on CPU: normal_RC's P1 − P0 is 8.5 GiB on GPU and 3.5 GiB on CPU,
  a ratio of 2.43, against 0.95 for the rest. So a second estimate scales the P0 part by 0.95 and what the cell adds to
  its P0 by 2.43. On normal_RC's P2 it gives 37.7 GiB against the measured 34.1 (10 % high), so it errs on the safe side.

**GPU estimate at batch 12800** (GiB): the larger of the two estimates; `@mem` = estimate × 1.25 + 1 GiB (allocator
slack and each process's CUDA context), rounded up.

| Cell | same-loss ratio | two-part | `@mem` |
|---|---|---|---|
| p0_small_RC | 8.6 | 8.6 | 12 |
| p1_small_RC | 12.1 | 13.6 | 19 |
| p2_small_RC | 9.1 | 10.2 | 14 |
| p0_small_C | 3.5 | 3.5 | 6 |
| p1_small_C | 6.2 | 8.4 | 12 |
| p2_small_C | 4.0 | 5.0 | 8 |
| p0_normal_C | 6.0 | 6.0 | 9 |
| p1_normal_C | 10.9 | 14.9 | 20 |
| p2_normal_C | 8.1 | 11.6 | 16 |

Within the 40 GiB budget of a card: any 2 runs of small_RC or normal_C fit, a third only without P1; any 3 runs of
small_C fit, 4–5 when mixed. These are estimates; each run logs its real peak (`mem/gpu_peak_MiB`) in its first epoch.

Not measured here: the time per epoch. The CPU times are no guide (the aggregator runs one jet at a time on CPU).
