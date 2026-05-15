# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**fm4tag** is a PyTorch Lightning library for pretraining and fine-tuning transformer-based encoders on mixed categorical/continuous particle physics data (jets + tracks from HDF5 files), targeting jet flavour tagging at the LHC (ATLAS experiment).

Two-phase transfer learning workflow:
1. **Pretrain** — self-supervised (contrastive InfoNCE + denoising) on each object encoder independently
2. **Finetune** — supervised classification using a frozen-then-unfrozen encoder

## Package management

Uses **uv** exclusively. Do not use `pip` directly.

```bash
uv add <package>           # add a runtime dependency
uv add --dev <package>     # add a dev dependency
uv sync                    # install all dependencies
```

## Development commands

```bash
# Lint / format
uv run ruff check src/
uv run ruff format src/     # single-quote style enforced

# No tests directory exists yet — add tests under tests/ when writing them
# uv run pytest tests/

# Install the package so the entry points are available
uv sync

# Run via entry points (all config via YAML + Hydra dot-notation overrides)
fm4tag --config-name=default phase=pretrain action=fit
fm4tag --config-name=default phase=finetune encoder_ckpt=<path>     # from pretrained
fm4tag --config-name=default phase=finetune ckpt_path=<path>         # resume
fm4tag --config-name=default phase=finetune action=test  ckpt_path=<path>
fm4tag --config-name=default phase=finetune action=predict ckpt_path=<path>

# Evaluate encoder representations (uniformity, effective rank) without fine-tuning
fm4tag-eval --config-name=default ckpt_path=outputs/exp/version_0/checkpoints/best.ckpt

# Hyperparameter optimisation (Optuna)
fm4tag-hpo --config-name=default

# Load config from outside the repo
fm4tag --config-path=/my/configs --config-name=my_experiment phase=finetune

# From a notebook (no Hydra)
from omegaconf import OmegaConf
from fm4tag.engine import run
cfg = OmegaConf.load('configs/default.yaml')
run(cfg, phase='finetune', action='fit')

# Profiling (disabled by default — enable via CLI)
fm4tag profiler.enabled=true profiler.type=simple    # simple | advanced | pytorch
```

Outputs write to `outputs/<experiment_name>/version_N/`. TensorBoard: `tensorboard --logdir outputs/`.

## Source layout

```
src/fm4tag/
  augmentations/   embed_data(), add_noise(), mixup_data()
  callbacks/       MemoryMonitorCallback, _PrecisionProgressBar
  cli/engine.py    Hydra entry point; run() public API; builder helpers
  datamodules/     PT_FT_DataModule (Lightning DataModule)
  datasets/        DatasetCatCon (HDF5 lazy loading), cat_con_collate_fn
  losses/          InfoNCELoss, DenoisingLoss
  metrics/         uniformity(), effective_rank()
  models/
    encoder.py     GlobalEncoder, Encoder (SAINT-style transformer)
    heads.py       MultiStreamClassifierHead
    mlp.py         simple_MLP, sep_MLP, MLP_dropout
    transformers.py Transformer, RowColTransformer, Classifier_Transformer, …
    attention.py   RowColAttention, RowAttention
    blocks.py      PreNorm, FeedForward, Residual
  modules/
    pretrain_module.py   PretrainModule (LightningModule)
    finetune_module.py   FinetuneModule (LightningModule)
  eval_encoder.py  Standalone encoder evaluation (fm4tag-eval entry point)

configs/           YAML configs (repo root, NOT under src/)
scripts/           Utility scripts (plot_losses, plot_uniformity, model_summary)
slurm/             SLURM job scripts
```

## Architecture

### Multi-object design

The refactored codebase handles **multiple object types** simultaneously. `_build_encoders()` in `engine.py` creates a `ModuleDict` with one encoder per object:
- `cfg.global_object` (e.g. `jets`) → `GlobalEncoder` — continuous-only, per-feature grouped Conv1d MLP, no attention
- each `cfg.constituent_objects` (e.g. `[tracks]`) → `Encoder` — SAINT-style transformer for mixed cat/con features

During pretraining, each encoder receives gradients only from its own per-object loss (contrastive + denoising computed independently). During finetuning, all encoders are stored as `FinetuneModule.backbone` (a `ModuleDict`) for `BackboneFinetuning` callback compatibility.

### Data flow

```
HDF5 → DatasetCatCon.__getitem__
      → (label, global_feats, constituents{categorical, continuous, valid})
      → cat_con_collate_fn  →  batch dict:
          label:                              (B,)
          global:                             (B, F_g)
          constituents[obj]["categorical"]:   (B, C, F_cat)  long
          constituents[obj]["continuous"]:    (B, C, F_con)  float32
          constituents[obj]["valid"]:         (B, C)         bool
```

### Encoding pipeline

```
constituents (B, C, F_cat/F_con)
  → flatten valid: (N_valid, F_cat/F_con)
  → embed_data()  → (N_valid, F_cat, dim), (N_valid, F_con, dim)
        categorical: embeds(x + offset)
        continuous:  grouped Conv1d MLP  [cont_fc1 → relu → cont_fc2]
  → Encoder.forward() → (N_valid, F_cat+F_con, dim)
  → encoder.pt_mlp1(X.flatten(1,2)) → (N_valid, proj_out)
  → scatter back to (B, C, proj_out), zeros at invalid positions

global (B, F_g)
  → GlobalEncoder.forward() → (B, F_g, dim)
  → encoder.pt_mlp1(X.flatten(1)) → (B, F_g*dim)

→ MultiStreamClassifierHead  → (B, y_dim) logits
```

### MultiStreamClassifierHead

Receives **pre-projected** embeddings (already through `encoder.pt_mlp1`) — does not have its own projection layers. Per constituent stream: cross-constituent `Classifier_Transformer` → masked mean pool over valid constituents. Concatenates global + all constituent streams, then 2-layer classification MLP.

### Projection head reuse (`pt_mlp1`)

`encoder.pt_mlp1` serves double duty:
- **Pretraining**: contrastive projection head (InfoNCE loss is computed on `pt_mlp1` output)
- **Finetuning**: the same weights are used as the projection into the classifier head

This means uniformity / effective rank logged during pretraining measure the same embedding space used during finetuning.

### Encoder attention variants (`attentiontype`)

| Value | Transformer | Notes |
|---|---|---|
| `col` | `Transformer` | within-sample (column) attention only — default |
| `colrow` | `RowColTransformer` | alternates col and row (cross-sample) attention |
| `row` | `RowTransformer` | cross-sample attention only |
| `concat` | `Concat` | concatenates features, no attention |

### Online representation metrics

Controlled by the `eval:` config section. At the end of each epoch, `uniformity` and `effective_rank` are computed on subsampled `pt_mlp1` embeddings and logged to TensorBoard/CSV. Gathering across DDP ranks uses `self.all_gather()` with size-matched tensors — all ranks must enter the collective in the same order.

### BackboneFinetuning

When `freeze_encoder: true`, `FinetuneModule.configure_optimizers` registers only head parameters. `BackboneFinetuning` callback adds `self.backbone` parameters at epoch `unfreeze_backbone_at_epoch` with lr = `backbone_initial_ratio_lr × current_lr`.

When `freeze_encoder: false` (default), backbone uses `optimizer.backbone_lr` and the head uses `optimizer.lr` from the start.

### Checkpoints

`_load_pretrained_encoders()` in `engine.py` handles two checkpoint formats:
- New: `encoders.<obj_name>.*` (multi-encoder PretrainModule)
- Legacy: `encoder.*` (single-encoder format)

`FinetuneModule.on_load_checkpoint` drops stale head keys (`head.global_proj.*`, `head.const_proj.*`, `head.global_agg.*`, `head.const_phi.*`) from old checkpoints and handles shape mismatches in `backbone.*.pt_mlp*` layers.

### Config

Configs live in `configs/` at the repo root (not under `src/`). The Hydra `config_path` in `engine.py` is set to `'configs'` relative to the source file. Copy `configs/default.yaml` to add a new experiment config. Required top-level keys: `phase`, `action`, `global_object`, `constituent_objects`, `variables`, `encoder`, `head`, `pretrain`, `optimizer`, `trainer`, `callbacks`.

`engine.run()` can be called from notebooks with a manually loaded `OmegaConf` config — Hydra is not required in that path.
