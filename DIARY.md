# fm4tag diary

Plan, decision log and progress. One dated entry per session, newest last:
Done / Decisions / Numbers / Open questions / Next.

## 2026-11-25 — config refactor, step 1: one home for the optimiser and lr schedule

**Plan (config refactor, agreed today).**
1. One `optimizer` block for lr, weight decay and the lr schedule; the runner passes it to the module. *(this entry)*
2. Step 1b: one shared warmup / cosine / wsd function for both modules (code only, same lr curves).
3. Step 2: a `phase` group (`pretrain` | `finetune`) carries its module block, `monitor` and gradient clipping;
   `arm` (`scratch` | `finetune` | `frozen`) becomes a plain key of the finetune phase, checked by the runner, which
   derives `freeze_encoders` from it; `callbacks_finetune` goes away (`monitor: ${monitor}`).
4. Delete `ContrastiveDenoisingModule` (+ `modules/losses/`): not reachable from the runner or any config.

**Done.**
- `optimizer:` in both bases (`default.yaml`, `saint_reduced.yaml`) now holds `lr`, `weight_decay`, `schedule`,
  `warmup_frac`, `warmup_epochs`, `decay_frac`. The `pretrain` / `finetune` blocks no longer write them (before:
  lr / wd linked with `${}`, the schedule keys written twice and unlinked, so `finetune.schedule=wsd` was silently
  ignored by a pretraining).
- `runner/run.py`: passes `cfg.optimizer` to whichever module runs. Module signatures unchanged, so run configs
  written before today (lr inside the module block) still rebuild in `roc_comparison.py` & co.
- `optimizer.backbone_lr` (written by every arm) → `finetune.backbone_lr: null` in the base. The arms' values were
  redundant: FT / FR set `${optimizer.lr}`, which is what `null` already means in `FinetuneModule`; scratch and
  pretrain do not read it.
- Scripts: `report_deck.py` reads warmup / backbone lr from both the new and the old run-config layout;
  `make_wsd_cells.py` and `dropout_batch_sweep.py` write the new keys. `docs/usage.md` and the `hpo.py` docstring
  updated.
- Tests: `test_the_lr_schedule_is_written_only_in_optimizer` (layout rule + every optimizer key is a parameter of
  both modules); `test_run_builds_the_module_with_the_optimizer_block` (calls `run()` with `Trainer.fit` patched,
  pretrain + scratch; checked to fail when the runner does not pass the block); the HPO interpolation test now
  uses `lam: ${lambda}`.

**Decisions.**
- Default schedule: **wsd** (was cosine), for both pretraining and fine-tuning (rriva).
- Keep the name `optimizer` (229 cell overrides, the HPO search space and the report scripts use `optimizer.lr`).
- Fine-tuning gradient clipping default stays `null`; protocol-v1 cells set 1.0 (rriva, for step 2).
- Keep the name `arm` for the SC / FT / FR switch (rriva, for step 2).

**Numbers.** `uv run pytest` on 41cdcb5 + this working tree: 347 passed, 14 skipped, 1 failed
(`test_campaign_uses_the_disjoint_split`: `batch_size` 12800 is not a multiple of 8 · `chunk_size` = 4096; the
12800 is an edit in the working tree that predates this step).

**Open questions.**
- `unfreeze_at_epoch` after step 2: FT uses 2, FR 0, and step 2 leaves one default. Which arm sets it in its cells?
  (Only matters for FR from P1, whose pretrained aggregator would sit at lr 0 for 2 epochs with a default of 2.)
- `batch_size: 12800`: keep it and relax the test + base comment (multiple of `chunk_size` only), or go back?
- `optimizer.lr = 3e-4` was tuned under cosine (S3); with wsd as the default it is no longer a tuned value.
- Old cells in `experiments/` that use `finetune.schedule` / `optimizer.backbone_lr` no longer compose (records only).

**Check of `slurm/` and `experiments/` after step 1 (committed as be345a8).**
- `slurm/`: no script writes or reads a moved key. `pack_worker.sh` parses `arm=` only for the wandb `job_type`
  (to revisit in step 2, when pretraining becomes `phase=pretrain`). `ssl_pack.sh` refuses to launch with
  uncommitted config changes and composes every cell at launch, so an old-key cell fails at submission.
- T4way (running, 24/24 tasks claimed or done): the pack worker starts `fm4tag` from this checkout only when it
  claims a task, so the running trainings are not affected. Composed on be345a8, every queued T4way override line
  fails, first at `encoders=` (renamed `encoder` in a4577c4) and then at `finetune.schedule`. A task re-claimed after
  a pack job dies would fail at compose and be marked `.failed`. Both renames together compose all 24 lines and
  both P_v1 lines (checked with Hydra `compose`).
- Closed experiments: their cells / slurm scripts belong to the branch they ran on (README "Code" column); left
  unchanged.
- P_v1: cancelled 2026-11-25 (inspection log §14), but its README still says "running" and `experiments/README.md`
  says "planned".

Renames an old cell needs to run on main: `encoders` → `encoder` (a4577c4); `pretrain.|finetune.{schedule,
warmup_frac,warmup_epochs,decay_frac}` → `optimizer.*`; `optimizer.backbone_lr` → `finetune.backbone_lr` (be345a8).

**Next.** Step 1b (shared schedule function), after review.

## 2026-11-25 (evening) — SAINT-small: GN2-like aggregator and head, pretraining prepared, t4 stopped

**Done.**
- **SAINT-small sizes** (owner):
  - encoder `rowcol_concat_v1_small`: depth 2 (both encoders), 4 track row heads, `ff_mult` 2, track projector
    304 → 256 → 256;
  - aggregator `transformer_v1_small`: width 256 (= the track projector's output), 4 layers, 8 heads × 32,
    `ff_mult` 1, per-track projection 256 → 128 + LayerNorm;
  - head `mlp_v1_small`: 128 → 64 → 32, no dropout;
  - 3.56 M parameters with the head (GN2: 2.24 M).
- **Code:**
  - `TransformerAggregator` gets `proj_dim` and `final_norm` (off by default: model B's state_dict is unchanged).
    They are applied per track after the transformer, in GN2's order: salt's `out_proj`, then `out_norm`.
  - `MultiStreamClassifierHead.mlp_hidden` takes an int or a list.
  - `scripts/dropout_batch_sweep.py` gets `--config-name`.
  - Tests: `tests/test_aggregator.py` (+ the projection and norm), new `tests/test_heads.py`.
- **GN2 rebuilt** from salt's `config.yaml` (`run_20261106_092332` s42; the epoch-12 checkpoint loads with no missing
  keys). Depth-5 module summary in the inspection log §14; script in the session scratchpad (`gn2_summary.py`).
- **New experiment `experiments/2026-11_pretrain-saint-small/`** (README, `configs/P_small.cells`: P0 and P1 with the
  P_v1 recipe, `CONFIG=saint_reduced`).
  - Measurement job 7470396 (peak GPU memory and s/step of both cells at batch 12800) submitted. Pending at 21:30.
- **Four-way t4 (CE + JC + dropout) stopped.** The 4 A40 seeds (jobs 7385105/106) were cancelled at epoch 35, with
  their tasks marked `.failed` + `.cancelled`. The 2 L40S seeds had finished: best val CE 0.737 and 0.786, last 1.2.
- **Docs:**
  - P_v1 marked cancelled (its README and `experiments/README.md`);
  - README layout and docs list (`slurm/` is generic tools now; `experiments/`, `gpu_packing.md`, `DIARY.md` added);
  - `gpu_packing.md` launch example (cells path, `CONFIG=`);
  - `usage.md` group table;
  - inspection log §14 (SAINT-small, GN2 comparison) and §15 ("changed since then").

**Decisions** (owner):
- the SAINT-small sizes above;
- GEGLU kept (not ReLU), no register token, mean pool kept, jet pT/η not attached to the tracks;
- t4 cancelled.

**Numbers.**
- Parameters (fine-tune phase): track encoder 1.60 M, jet encoder 0.04 M, aggregator 1.88 M, head 0.04 M.
- Aggregator layer at width 256: 8 × 32 attention 262 K; GEGLU feed-forward `ff_mult` 2 394 K (GN2's ReLU 263 K).
- Tests: `pytest -m "not ddp"` 349 passed, 1 failed (`test_campaign_uses_the_disjoint_split`, the batch 12800 open
  question above). The DDP suite was not run.
- Four-way best val CE so far (20:43): t1 CE 0.620–0.633, t2 CE + JC 0.624–0.630, t3 CE + dropout 0.622–0.629 (two
  seeds 0.653 and 0.663). GN2: 0.6137–0.6154.

**Open questions.**
- Pretraining epochs (P_v1's 80, or 40) and seeds (2 seeds per pack if a P0 and a P1 fit on one card).
- Batch size, lr and `chunk_size` of `saint_reduced` are still model B's.
- `docs/param_scan_guide.md` describes a closed experiment's launcher (`2026-08_lambda-scan`). Should it move to that
  experiment's `notes/`?

**Next.**
- Code, configs and the experiment folder are committed (bc107ea, 5d0ba97, 29b0235). The doc edits of this entry are
  not yet committed.
- Read the measurement, set `@est` / `@mem` in `P_small.cells`, launch `P_small`.
