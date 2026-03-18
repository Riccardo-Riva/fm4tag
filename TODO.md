# fm4tag — Development TODO

---

## 1. Uniformity and effective-rank monitoring (online, during training)

### Existing infrastructure — post-hoc evaluation
The following already exists and works:

- `src/fm4tag/models/components/eval_metrics.py` — `uniformity()` function (Wang & Isola 2020).
- `src/fm4tag/eval_encoder.py` — `evaluate()` function: loads a checkpoint, runs
  `encoder.pt_mlp1` embeddings over a dataset, computes both uniformity and effective rank
  (effective rank is implemented inline as the private `_effective_rank()`).
- `scripts/plot_uniformity.py` — iterates over saved per-epoch checkpoints, calls
  `evaluate()` for each, plots metrics vs epoch.
- `scripts/plot_uniformity_yaml.py` — plots from a pre-computed YAML cache.

**The gap**: these tools only work post-hoc (loading saved checkpoints). There is no
online monitoring — i.e. metrics are not logged to TensorBoard/CSV during training, so
you cannot watch them in real time alongside the loss.

### Goal
Log uniformity and effective rank **at the end of each epoch** inside the Lightning
modules, so they appear in TensorBoard and the CSV log alongside the loss curves.

### What embeddings to use
Same as `eval_encoder.py`:
- Constituent encoders: `z = encoder.pt_mlp1(X.flatten(1, 2))` where `X` is the
  transformer output `(N_valid, F, dim)`.
- Global encoder: `z = encoder.pt_mlp1(X.flatten(1))` where `X` is `(B, F_g, dim)`.
- Jet-level (optional): masked mean pool of track encoder outputs, then `pt_mlp1`.

### Implementation plan

#### Promote `effective_rank` to `eval_metrics.py`
Currently `_effective_rank` is a private function inside `eval_encoder.py`.  Move it
into `src/fm4tag/models/components/eval_metrics.py` as a public `effective_rank(z)`
function so it can be imported by the Lightning modules without pulling in the full
evaluation pipeline.

#### Config additions (`default.yaml`)
Add a new top-level section (or under `pretrain:` / callbacks):
```yaml
eval:
  enabled: true          # master switch for online metric computation
  splits: [val]          # val | train | both
  n_samples: 8192        # max embeddings per object per epoch (subsampled)
  objects: null          # null = all; or list of object names to restrict
  log_uniformity: true
  log_effective_rank: true
```

#### Changes to `PretrainModule`
- Accumulate `z` tensors per object during `validation_step` (if `eval.splits` includes
  `val`) into a new `self._emb_acc: dict[str, list[Tensor]]` buffer — exactly like the
  existing `self._val_acc` for losses.
- In `on_validation_epoch_end`: concatenate accumulated embeddings, call `uniformity(z)`
  and `effective_rank(z)`, log as `val_{obj}/uniformity` and `val_{obj}/effective_rank`,
  then clear the buffer.
- Mirror for train split in `on_train_epoch_end` if requested.
- Respect `cfg.eval.n_samples` by capping what gets appended to the buffer.
- Guard everything behind `cfg.get('eval', {}).get('enabled', False)` so existing runs
  are unaffected.

#### Changes to `FinetuneModule`
- Same pattern — accumulate `z = backbone[obj].pt_mlp1(enc.flatten(1, 2))` during
  `validation_step`, compute and log metrics at epoch end.
- The backbone's `pt_mlp1` weights reflect the pretrained geometry (if loaded from a
  pretrained checkpoint) or random init (train-from-scratch) — valid either way.
- The forward pass already runs the backbone; only a `pt_mlp1` call on the intermediate
  encoder output is needed (no changes to the head or the loss).

#### Logging
- Log via `self.log(...)` with `on_step=False, on_epoch=True` — same style as existing
  per-object breakdowns.
- Optionally include columns in the epoch summary table (`_format_epoch_table`).

#### DDP note
For multi-GPU runs, gather embeddings across ranks with `self.all_gather(z)` before
calling `uniformity` / `effective_rank`, since each rank only sees a data shard.

---

## 2. Reuse `pt_mlp1` as the constituent/global projection in the classifier head

### Motivation
During pretraining, `encoder.pt_mlp1` maps flattened encoder output `(N, F*dim)` →
`(N, proj_out)`.  During finetuning, `MultiStreamClassifierHead` independently defines
`self.const_proj` and `self.global_proj` that perform exactly the same shape of
transformation.  These are currently **separate, independently-initialised** weights:

- The head discards the embedding geometry learned by `pt_mlp1` during pretraining.
- Uniformity / effective-rank metrics computed via `pt_mlp1` (item 1) measure a
  different space than what the head actually projects into.

The fix: **always** use `encoder.pt_mlp1` as the projection inside the head.
`self.const_proj` and `self.global_proj` in `MultiStreamClassifierHead` are removed;
the encoder projections are passed in and stored directly.  There is no fallback option.

### Dimensional consistency check
| Quantity | Constituent `Encoder` | `GlobalEncoder` |
|---|---|---|
| `pt_mlp1` input | `dim × (F_cat + F_con)` | `num_features × dim` |
| `pt_mlp1` output (`proj_out`) | `proj_in // 2` (auto) | `proj_in` (= input) |
| `const_proj` input | `n_feat × dim` (same) | same |
| `const_proj` output | `cls_dim` | `cls_dim` |

`cls_dim` in the head config becomes redundant for the projection layers and is instead
inferred automatically from `proj_out` of the encoders (enforced in engine).

Note: the `GlobalEncoder`'s `pt_mlp1` maps to `proj_in` (not `proj_in//2`), which
differs from the constituent encoder by default.  The head and engine need to handle
potentially different `cls_dim` per stream, or `proj_out` must be aligned via config.

### Implementation plan

#### Remove `self.const_proj` and `self.global_proj` from `MultiStreamClassifierHead` ([heads.py](src/fm4tag/models/components/heads.py))
- Replace the `cls_dim` / `mlp_dropout` MLP construction with required constructor
  parameters `const_projs: nn.ModuleList` and `global_proj: nn.Module`.
- `_cls_dim` is inferred from the output size of the provided projections.
- Remove `cls_dim` and `mlp_dropout` from the constructor signature (they are no longer
  needed for the projection step; `mlp_dropout` still applies to `cls_mlp`).
- The `on_load_checkpoint` rename entries for `head.global_agg` / `head.const_phi`
  ([finetune_module.py:217](src/fm4tag/models/finetune_module.py#L217)) become
  irrelevant — remove them from `_renames` as well.

#### Remove `cls_dim` from head config
`cls_dim` in `default.yaml` under `head:` is no longer meaningful for the projections.
Remove it or repurpose it only for the `cls_mlp` final classification MLP width if a
separate override is still desired.

#### Changes to `engine.py` (head construction)
1. After building encoders, extract `encoder.pt_mlp1` for each constituent object and
   the global encoder's `pt_mlp1`.
2. Infer `cls_dim` from the `proj_out` dimension of the first constituent encoder;
   assert all constituent encoders agree.
3. Pass `const_projs=nn.ModuleList([enc.pt_mlp1 for enc in constituent_encoders])` and
   `global_proj=global_encoder.pt_mlp1` to `MultiStreamClassifierHead`.

#### Checkpoint compatibility
- `pt_mlp1` weights live under `backbone.<obj>.pt_mlp1.*` in pretrain checkpoints.
- With the head directly holding references to `backbone.<obj>.pt_mlp1`, the weights
  are loaded as part of the backbone — no separate head key remapping needed.
- Old finetune checkpoints that stored `head.const_proj.*` / `head.global_proj.*` will
  no longer match the new state-dict layout.  The `on_load_checkpoint` hook in
  `FinetuneModule` must be updated: drop the old keys (they are superseded by the
  backbone's `pt_mlp1` weights).

#### Benefit for metric consistency (connects to item 1)
The `z` vectors evaluated by uniformity / effective rank during pretraining are produced
by the **same weights** that project constituents inside the head during finetuning —
making the metrics directly comparable across all training phases.

---

## Notes / open questions

- Decide whether to also compute jet-level uniformity (masked mean pool over valid track
  embeddings, then `pt_mlp1`) during online monitoring, as `eval_encoder.py` already does
  post-hoc.
- When `projhead_style: same`, `pt_mlp1 is pt_mlp2` already — consistent with
  `use_encoder_proj` reusing `pt_mlp1`.
- Consider fixing a random seed for the embedding subsample index at the start of
  training (`setup()`) so the same subset of validation samples is used every epoch,
  making the metric time-series more comparable.
- For the `GlobalEncoder`, `pt_mlp1` output dim equals `num_features * dim` (i.e. no
  reduction), while the constituent encoder halves it by default.  This asymmetry may
  matter when choosing a shared `cls_dim`.