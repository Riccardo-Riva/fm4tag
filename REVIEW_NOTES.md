# Framework review — findings log

**Branch:** `review/claude-deep-dive`. Full review of the framework with
focus on the two LightningModules. Suite: 146 non-DDP + 12 DDP tests pass.

## Fixed (committed on this branch)

1. **b2821de — `TransformerAggregator` masked mean-pool leaked padding.**
   Invalid slots re-populated by the cross-constituent transformer (LayerNorm
   beta, padded queries attending valid keys, FFN bias) were included in the
   pooled sum. Verified O(1) output shifts vs. amount of padding. Re-masked
   after the transformer; regression tests in `tests/test_aggregator.py`.
   ⚠ Numerics change vs. checkpoints trained before the fix.
2. **f0b28b9 — `RowAttention` key-padding mask broadcast.** `(B,1,1)` cannot
   broadcast against the `(heads, B, B)` scores; crashes on use (or silently
   masks per-head when B == heads). Dead path today; fixed + tests.
3. **0ebf015 — NaN guards in losses.** Empty local anchor slice (SupCon under
   DDP), empty batches in denoising cat/con → NaN poisoning gradients via
   allreduce. Now graph-connected zeros.
4. **5785e12 — DDP without `find_unused_parameters`.** Heads that can never
   receive gradients are frozen at module construction (pretrain:
   weight-driven; finetune: reconstructors always); optimisers take only
   trainable params. `ddp_find_unused_parameters_true` can be dropped from
   configs. Also: `PretrainedFinetuning.setup` warns when world_size > 1 and
   unfreeze < max_epochs (frozen-at-wrap params get no DDP hooks → unfreezing
   silently desyncs ranks; verified hook order in installed Lightning).
5. **b293953 — augmentation `setup()` no-op trap.** `CategoricalShift` /
   `ContinuousFeatureDilation` now warn once when used without `setup()`
   (nothing in the framework calls it).
6. **b23db56 — short clarifying comments** (sync_dist cost, V+1 finetune
   passes, EMBEDDING-stage asymmetry, batch-dependent intersample attention).
7. **8579346 — Optuna HPO** (`fm4tag.hpo`, `fm4tag-hpo` entry point, tests);
   `run()` now returns the Trainer.
8. **70a6739 — docs/usage.md** (install, data, configs, workflows, logging
   conventions, HPO, caveats).

## Open recommendations (not implemented — need owner decision)

- **Delete legacy code:** `modules/contrastive_denoising_module.py` (628
  lines) + `modules/losses/` adapters + duplicate `metrics/metrics.py`
  (uniformity/effective_rank duplicated by the registry versions;
  `eval_encoder.py` imports the duplicates — redirect first).
- **Verify `use_class_weights` semantics:** `run.py` passes
  `class_dict[obj][label]` straight to CE `weight=`. If the YAML stores raw
  counts (not weights), this up-weights majority classes — check the actual
  class_dict file. Related: `DatasetCatCon.class_dict` is stored, never used.
- **`FinetuneModule.on_load_checkpoint`** silently replaces shape-mismatched
  keys with random init — consider a loud warning listing affected keys.
- **Per-object SupCon memory:** gathered similarity matrix is (ΣN_r·V)² —
  with B=1024, ~20 valid tracks/jet, 2 views, 2 ranks that is ~(80k)².
  Chunked/streaming logsumexp would cap it if it becomes a limit.
- **`sync_dist=True` on on_step train logs** costs a cross-rank reduce per
  metric per step (commented in code). Option: log step values unsynced and
  epoch values synced.
- **Structurally DDP-safe staged unfreezing** (replace requires_grad freeze
  with lr=0 param group until epoch E) if multi-GPU finetuning with
  unfreezing is ever needed.
- **Finetune CE uses a clean pass while jet-contrastive re-encodes V views**
  (V+1 passes). If cost matters, an Identity first view could be reused for CE.
- **EMBEDDING-stage asymmetry:** for the global object the stage applies to
  the encoder *output*; for constituents to embedded tokens *before* the
  transformer. Unify or document as intended.
