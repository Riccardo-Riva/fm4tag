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
9. **9d35661 — `col`/`row` transformer types crashed; encoder
   architecture moved to a Hydra config group.** `_build_layers` forwarded the
   whole layer dict to every block class, but the shared `layers:` block spoke
   `RowColTransformer`'s vocabulary (`col_heads`/`row_heads`/`dim_row_head`/
   `chunk_size`), so `type: col`/`row` raised `TypeError` on the first foreign
   kwarg — only `rowcol` ever ran (every `col` sweep run died in ~7 s). Fix:
   per-type config group `configs/encoders/{rowcol,col,row}_v0.yaml`, each
   listing only the kwargs its class accepts; selected via the `defaults:` list
   / `encoders=<type>_v0`, replacing the `transformer_type=` interpolation.
   Verified all three build and that group-swap + deep overrides
   (`encoders.constituents.tracks.layers.0.depth=7`) still apply. Also promoted
   the working config to `default.yaml`, deleted redundant `default_01.yaml` /
   `jets_only*.yaml` + the `run_jets_only_test` scripts, and updated the slurm
   generators + docs. ⚠ `OmegaConf.load` no longer yields a complete config —
   the `defaults:` list needs Hydra `compose`.
10. **4fb18ec — contrastive collapse: augmentation views destroyed
    sample identity.** Every SupCon term sat pinned at its ~log(N) max-entropy
    floor (pretrain `jets`/`aggregator` ≈ 7.12, `tracks` ≈ 9.60; finetune
    `jet_contrastive` ≈ 7.62), and enabling `jet_contrastive` at ANY weight
    collapsed finetune `val/head/auroc` from ~0.89 (ce-only) to ~0.53 (random),
    jc=0.3 as hard as jc=1.0. Cause: `lam` meant "fraction **kept**" (CutMix) /
    "weight on the **original**" (Mixup), and the config set both to 0.1, so
    view 0 kept ~1 % of the anchor and was ~99 % *other* jets — the SupCon
    "positive" was indistinguishable from a negative (measured
    `cos(view0,self)` = 0.01 vs `cos(view0,other)` = 0.01, 46 % closer-to-self).
    Fix: **inverted `lam` to a noise level** (0 = identity, 1 = full corruption)
    in `CutMix`/`Mixup` (defaults flipped 0.7→0.3, 0.8→0.2); the config's
    `lam=0.1` now means mild noise → `cos(view0,self)` = 0.89, 99 % closer-to-
    self. Tests updated (`test_augmentations.py`), 146 non-DDP pass.
11. **(pending commit) — DDP-safe staged unfreeze; `PretrainedFinetuning`
    callback removed.** The callback froze the pretrained parts via
    `requires_grad=False` *before* the DDP wrap, so those params got no
    gradient-sync hooks; unfreezing them mid-run (any epoch < max_epochs) left
    their gradients un-all-reduced and the ranks silently diverged — and
    `finetune.sh` runs on 2 GPUs. Replaced with a schedule-based staged unfreeze
    in `FinetuneModule.configure_optimizers`: all params are in the optimiser
    from step 0 (each gets a DDP hook), the pretrained group is held at `lr=0`
    by a per-group `LambdaLR` multiplier until `finetune.unfreeze_at_epoch`,
    then ramps onto the head's cosine curve (AdamW at lr=0 is a true freeze).
    Deleted the callback + `callbacks_finetune` config; `unfreeze_at_epoch` now
    lives in the finetune block (default 3; classify-from-scratch sets 0),
    `initial_ratio_lr` dropped (backbone group uses `backbone_lr` directly).
    Tests rewritten; 145 non-DDP pass. End-to-end: backbone group lr=0 at step 0,
    live after the unfreeze epoch.

## Open recommendations (not implemented — need owner decision)

- **Contrastive has no projection head.** Both pretrain and finetune apply the
  SupCon loss directly to the aggregator/encoder embedding the classifier also
  consumes, so the contrastive objective reshapes the classification embedding
  itself. Standard SimCLR/SupCon uses a separate projection head. Now that the
  views are fixed (finding 10) this is a refinement, not a blocker — worth an
  ablation. (Collapse seen in runs `run_20260707_103027` finetune,
  `run_20260707_103022` pretrain.)

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
- **Finetune CE uses a clean pass while jet-contrastive re-encodes V views**
  (V+1 passes). If cost matters, an Identity first view could be reused for CE.
- **EMBEDDING-stage asymmetry:** for the global object the stage applies to
  the encoder *output*; for constituents to embedded tokens *before* the
  transformer. Unify or document as intended.
