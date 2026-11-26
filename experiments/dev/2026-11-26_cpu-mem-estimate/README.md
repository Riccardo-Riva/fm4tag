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
Pending.
