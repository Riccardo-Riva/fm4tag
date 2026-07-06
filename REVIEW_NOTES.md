# Framework review — working notes (Claude session handoff)

**Branch:** `review/claude-deep-dive` (started from `review/all` @ 5d45d75).
**Goal:** full framework review (focus: the two LightningModules), find bugs,
add clarifying comments, write usage docs, add Optuna HPO.
**Test baseline:** 119 non-DDP tests passed before any change; 123 pass now.

## Done (committed)

1. **b2821de — BUG FIX: `TransformerAggregator` masked mean-pool leaked padding.**
   Invalid slots were zeroed only *before* the cross-constituent transformer;
   after it they are non-zero again (LayerNorm beta, padded queries attending
   valid keys, FFN bias), and the pool summed over ALL slots / n_valid.
   Verified O(1) output shifts vs. amount of padding. Fixed by re-masking
   after the transformer; added `tests/test_aggregator.py` (padding-amount /
   padding-value invariance, all-invalid jet → zero vector).
   ⚠ Numerics change vs. checkpoints trained before the fix.

## Confirmed findings, NOT yet fixed (next steps, in order)

2. **BUG (latent): `RowAttention` mask broadcast is wrong** —
   `attention.py:81` `attn_mask = mask[:, None, None]` is `(B,1,1)` but q/k/v
   are `(h, B, d)` so SDPA needs a key mask broadcastable to `(h, B, B)`:
   correct is `mask[None, None, :]`. Verified crash:
   `RowAttention(dim=16, heads=4)(x, mask)` → RuntimeError. Currently a dead
   path (row attention is always called with `mask=None` because constituents
   are pre-flattened by valid), but fix + test.

3. **DDP trap 1: `PretrainedFinetuning` + multi-GPU = silently unsynced grads.**
   Verified in installed Lightning: `BaseFinetuning.setup()` calls
   `freeze_before_training` during `_call_setup_hook`, which runs BEFORE
   `strategy.setup()` wraps DDP (trainer.py:1039 vs 1053). Params frozen at
   wrap time get no DDP reduce hooks → after `unfreeze_at_epoch` their grads
   are never all-reduced → each rank trains a divergent backbone.
   Plan: emit a loud warning in `PretrainedFinetuning.setup` when
   `trainer.world_size > 1` and unfreeze < max_epochs; document workarounds
   (single-GPU finetune, unfreeze_at_epoch >= max_epochs, or drop callback and
   rely on `backbone_lr`).

4. **DDP trap 2: unused parameters crash with default DDP strategy.**
   - Pretrain: `jet_contrastive: 0` → aggregator params get no grads;
     `denoising_*: 0` → reconstructor heads unused; both contrastive
     weights 0 → projectors unused.
   - Finetune: encoder reconstructor heads (`cat_reconstructor`,
     `con_reconstructor`, `reconstructor`) are NEVER used.
   Configs work around it with `ddp_find_unused_parameters_true` (commented
   out in `pretraining_test_260630.yaml`!). Plan: in both modules' `__init__`,
   set `requires_grad_(False)` on heads whose loss weight is 0 (finetune:
   always freeze reconstructors) — removes them from DDP hooks *and* avoids
   the find_unused overhead. Add tests.

5. **Empty-input NaN guards (cheap):**
   - `MultiViewSupConLoss`: if a rank's local slice is empty,
     `loss_per_anchor[start:end].mean()` = NaN → poisons DDP allreduce.
     Guard: return zero connected to graph (`z_all.sum() * 0`).
   - `denoising_cat_loss` / `denoising_con_loss` on 0 valid rows → NaN;
     guard with early zero return.

6. **Augmentation `setup()` is never called by anything** —
   `CategoricalShift` and `ContinuousFeatureDilation` are silent no-ops when
   used from config. Plan: one-time warning at first forward when not set up
   + mention in docs. (Grep confirmed: no caller of `.setup(` in src.)

7. **Efficiency notes (comment/document, don't rewrite):**
   - `sync_dist=True` on `on_step` train logging in both modules' `_step` →
     per-step cross-rank reduce per metric; recommend sync only for epoch.
   - Finetune with `jet_contrastive > 0` runs V+1 full encoder passes per
     step (1 clean for CE + V augmented); pretrain reuses views. Comment.
   - Per-object SupCon on flattened constituents: gathered sim matrix is
     (ΣN_r·V)² — with B=1024, ~20 tracks/jet, 2 views, 2 ranks that is
     ~(80k)² → memory hot spot; note chunking option in docs.
   - EmbeddingMetrics runs an extra fwd per batch until quota — fine (capped).
   - `Encoder.con_reconstructor` gets `torch.ones(...)` tensor as `categories`
     (works, but should be `[1]*num_continuous`) — cosmetic.
   - HDF5 per-index random reads in `DatasetCatCon.__getitem__` — known cost.

8. **Design warts to document (not change):**
   - Intersample (row) attention ⇒ **inference is batch-dependent** (a jet's
     embedding/prediction depends on the other jets in its batch; eval-mode
     ChunkedRowAttention uses contiguous chunks). Must go in docs.
   - EMBEDDING-stage augs act pre-transformer for constituents but
     post-encoder for the global object (`encode_global_view` applies them to
     encoder *output*) — asymmetric semantics, comment it.
   - `run.py` `use_class_weights` passes `class_dict[obj][label]` straight as
     CE weights — if the YAML holds raw counts this is inverted (should be
     ~1/freq). VERIFY with the actual class_dict file.
   - `DatasetCatCon.class_dict` is stored but never used (dead).
   - `FinetuneModule.on_load_checkpoint` silently replaces shape-mismatched
     keys with random init — documented in docstring, but risky.
   - Legacy: `modules/contrastive_denoising_module.py` (628 lines) +
     `modules/losses/` adapters + duplicate `metrics/metrics.py`
     (uniformity/effective_rank duplicated in registry files) — recommend
     deletion (matches stated refactor plan); eval_encoder.py imports the
     duplicates from `metrics.metrics`, so redirect those imports first.

## Remaining plan (after fixes above)

9. **Comments pass** on `pretrain_module.py`, `finetune_module.py`,
   `view_encoding.py`, `attention.py` (chunked perm/inverse), datamodule —
   only where non-obvious; code is already well docstring'd.
10. **docs/usage.md**: install (uv), data layout (HDF5 + variables/norm/class
    dicts), config anatomy, pretrain → finetune → test/predict workflows,
    log-key naming `<split>/<component>/<metric>`, checkpoint loading
    (encoder_ckpt vs ckpt_path), eval_encoder, caveats (batch-dependent
    inference, DDP traps above, augmentation setup()).
11. **Optuna HPO**: create `src/fm4tag/hpo.py` — config section `hpo:` ALREADY
    EXISTS in `pretraining_test_260630.yaml` (study_name/storage/n_trials/
    sampler TPE/median pruner/search_space per phase with
    {param,type,low,high,log,choices}); `run.py` already accepts
    `extra_callbacks` and its docstring references
    `fm4tag.hpo._OptunaMetricCallback`. Optuna >= 4.7 already in pyproject
    deps. Add: suggest-params from search_space via `OmegaConf.update`,
    `_OptunaMetricCallback` (report metric on validation end + prune),
    metric/direction keys (default `val/loss`/minimize; finetune could use
    `val/head/auroc`/maximize), per-trial `experiment_name-trial{N}`,
    `max_epochs_{pretrain,finetune}` overrides, `fm4tag-hpo` console script in
    pyproject, light unit tests (mock trial; no real training).
12. Final report to user listing all findings.

## Conventions (from user memory)
- Small reviewable commits; `git diff --cached` check before each commit.
- Explicit module args, loss_weights dicts, `split/component/metric` log naming.
- User wants pushback and best-practice flags.
