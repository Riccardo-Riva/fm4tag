# FM4tag — usage guide

FM4tag pretrains transformer encoders on jet data with a multi-view
contrastive + denoising objective, then fine-tunes them (encoders +
aggregator + classifier head) for flavour tagging.

```
            per-object encoders          shared aggregator        head
jets    ──► GlobalTransformerEncoder ─┐
tracks  ──► Encoder (col/row/rowcol) ─┤─► TransformerAggregator ─► MLP ─► logits
(other) ──► Encoder ...              ─┘        z_jet (B, D)
```

- **Pretraining** (`PretrainModule`): per-object contrastive loss on
  projections, categorical/continuous denoising reconstruction, and a
  jet-level contrastive loss on the aggregator output. Which terms run is
  controlled by `loss_weights` (weight 0 ⇒ not computed, not logged, and the
  corresponding heads are frozen).
- **Fine-tuning** (`FinetuneModule`): cross-entropy on the head (+ optional
  jet-contrastive term), reusing pretrained encoder/aggregator weights.

## Install

```bash
uv sync            # creates .venv from pyproject/uv.lock
source .venv/bin/activate
pytest -m "not ddp" -q          # quick check (DDP tests: pytest tests/ddp)
```

## Data

HDF5 files with one structured dataset per object:

- `file[<global_object>]` — shape `(N,)`, one record per jet, containing the
  global input fields (e.g. `pt_btagJes`) and the label field.
- `file[<constituent>]` — shape `(N, C)` records with categorical +
  continuous fields and a boolean `valid` field (padding mask).

Three YAML side-files (paths set in the config):

- `variables.*` (inline in the config): which fields are inputs, the label
  name, `unique_labels`, and per-categorical-feature `cat_classes`.
- `norm_dict.yaml`: `{object: {feature: {mean, std}}}` — applied to
  continuous features in the dataset.
- `class_dict.yaml`: per-class weights used when `use_class_weights: true`.

## Configs

One **base config per model** in [configs/](../src/fm4tag/configs/):
[default.yaml](../src/fm4tag/configs/default.yaml) (model B; used when no
`--config-name` is given) and
[saint_reduced.yaml](../src/fm4tag/configs/saint_reduced.yaml).  A base selects
one option of each config group and holds the training settings; it never
includes another base, and every key is written in exactly one file (a group
option or the base — enforced by `tests/test_config_layout.py`), so a value
changes only on the command line or in a cells file.  A new model is a new base
file; a new variant of a part is a new option file in its group.

| Group | Options | Sets |
|---|---|---|
| `data/` | `split_152051` | files, objects, `variables`, the dataset half of `datamodule` |
| `arm/` | `pretrain` `scratch` `finetune` `frozen` | `phase`, `encoder_ckpt`, freezing / unfreeze epoch (+ gradient clipping for pretraining) |
| `encoder/` | `rowcol_concat_v0` (model B) `rowcol_concat_v1_small` `rowcol_v0` `col_v0` `row_v0` | `encoder`: one encoder per object |
| `aggregator/` | `transformer` | `aggregator` |
| `head/` | `mlp` | `head` (finetune only) |
| `views/` | `cutmix_mixup` | `lambda`, `views` (augmentation pipelines, both phases) |
| `callbacks/` | `default` `default_monitors` `smoke` | `callbacks`, `callbacks_finetune` |
| `logger/` | `csv_wandb` `csv` | `logger` |
| `hpo/` | `tpe` | `hpo` (read by `fm4tag-hpo` only) |

In the base: run control (`seed`, `experiment_name`, `output_dir`, ...),
`chunk_size` and `row_qk_norm`, how the data is read (`datamodule.batch_size`,
workers, ...), the `pretrain` / `finetune` modules (loss weights; `finetune.backbone_lr`),
`optimizer` (lr, weight decay and the lr schedule, one block for both
modules: the runner passes it to the module that runs), `trainer`.  Run configs written before 2026-11-25 call the encoder
block `encoders` (`build_encoders` still reads it, so old checkpoints rebuild)
and the logger block `loggers`; they also write the lr and schedule inside
`pretrain` / `finetune`, and the backbone lr as `optimizer.backbone_lr`.

Anything can be overridden on the command line with Hydra dot-notation.

## Running

```bash
# Pretrain (P0; P1 adds pretrain.loss_weights.jet_contrastive=1.0).  default.yaml
# (model B) is the default base; another model: --config-name=saint_reduced.
# Swap a part with its group, e.g. encoder=col_v0.
fm4tag arm=pretrain

# Finetune from a pretraining checkpoint (FT; arm=frozen keeps the encoders frozen)
fm4tag arm=finetune \
    encoder_ckpt=outputs/fm4tag-pretrain-test/version_0/checkpoints/epoch010-step5000.ckpt

# Train from scratch (SC)
fm4tag arm=scratch

# Resume an interrupted fit / evaluate / predict
fm4tag ... action=fit    ckpt_path=/path/to/last.ckpt
fm4tag ... action=test   ckpt_path=/path/to/best.ckpt
fm4tag ... action=predict ckpt_path=/path/to/best.ckpt   # writes predictions.pt
```

Checkpoint knobs:

- `encoder_ckpt` — pretraining checkpoint whose **encoder** (and, with
  `load_aggregator: true`, aggregator) weights initialise the finetune model.
- `ckpt_path` — a full Lightning checkpoint of the *current* phase (resume /
  test / predict).
- `finetune.unfreeze_at_epoch` — holds the pretrained parts (encoders +
  aggregator) at `lr=0` for this many epochs, then ramps them onto the lr
  schedule (`optimizer.schedule`). `0` trains everything from step 0 (from-scratch); `>= max_epochs`
  keeps them frozen. DDP-safe (all params are in the optimiser from step 0).

From a notebook — use Hydra's `compose` API so the config groups are resolved
(a plain `OmegaConf.load` does **not** process the `defaults:` list, so
`cfg.encoder` and the other group blocks would be missing):

```python
from hydra import compose, initialize_config_dir
from fm4tag.runner import run

with initialize_config_dir(version_base=None,
                           config_dir='/abs/path/to/src/fm4tag/configs'):
    cfg = compose(config_name='default', overrides=['arm=pretrain', 'encoder=rowcol_v0'])
trainer = run(cfg, phase='pretrain', action='fit')
```

Encoder representation quality without fine-tuning:

```bash
fm4tag-eval --config-name=my_config ckpt_path=outputs/.../best.ckpt
```

## Logging conventions

All keys follow `<split>/<component>/<metric>`:

- `train/tracks_embedding/loss_contrastive`, `train/jets_embedding/loss_denoising_con`
- `train/aggregator/loss_contrastive` (jet-level term)
- `val/head/loss_ce`, `val/head/acc`, `val/head/auroc` (finetune)
- `<split>/loss` — the weighted total (checkpointing/early-stopping monitor)
- `val/<embedding>/uniformity|effective_rank` — from the `EmbeddingMetrics`
  callback (clean, no-augmentation embeddings)

## Hyper-parameter search (Optuna)

Configured by the `hpo:` section; run with:

```bash
fm4tag-hpo hpo.n_trials=50
```

- `search_space.<phase>` is a list of
  `{param: <dotted.config.path>, type: float|int|categorical, ...}` entries.
  Overrides are applied to an **unresolved** config copy, so interpolations
  like `lam: ${lambda}` follow a tuned `lambda`.
- `metric` / `direction` define the objective (any logged key, e.g.
  `val/head/auroc` + `maximize`); it is reported every validation epoch and
  the `pruner` may stop bad trials early.
- `phases: [pretrain, finetune]` runs both per trial, feeding the pretrain
  phase's best checkpoint into finetune. Pruning applies to the last phase.
- Use **one device per trial**; parallelise with several `fm4tag-hpo`
  workers sharing `hpo.storage: sqlite:///hpo.db` (or a proper RDB).
- The wandb logger is dropped per trial by default (`hpo.disable_wandb`).

## Caveats / gotchas

- **Intersample (row) attention makes outputs batch-dependent — also at
  inference.** A jet's embedding/prediction depends on the other jets in its
  batch: the row step attends over the other jets of its chunk
  (`chunk_size` consecutive jets; the whole batch when `chunk_size: null`).
  Keep evaluation batch composition fixed when comparing numbers. (It is at
  least permutation-equivariant within a chunk and identical between train and
  eval.)
- **Staged unfreeze is DDP-safe:** `finetune.unfreeze_at_epoch` keeps every
  parameter in the optimiser from step 0 (so DDP registers a sync hook for
  each) and holds the pretrained parts at `lr=0` until the unfreeze epoch. The
  backbone backward + all-reduce still run during the frozen phase — correctness
  over the small cost of a true `requires_grad` freeze.
- **Zero-weight loss components freeze their heads** (reconstructors,
  aggregator, projectors) at module construction, so multi-GPU runs work
  with the default DDP strategy — `ddp_find_unused_parameters_true` is no
  longer needed.
- **`CategoricalShift` / `ContinuousFeatureDilation` need `setup()`**, which
  no framework code calls yet; they warn and act as no-ops until wired up.
- **First view = denoising input.** The original (unaugmented) batch is
  always the reconstruction target; put the strongest corruption in view 0.
- Checkpoints from before the aggregator masked-pool fix (commit `b2821de`)
  have slightly different numerics; re-validate before comparing.
