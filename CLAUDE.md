# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**fm4tag** is a PyTorch Lightning library for pretraining and fine-tuning transformer-based encoders on mixed categorical/continuous particle physics data (jets + tracks from HDF5 files), targeting jet flavour tagging at the LHC (especially in the ATLAS experiment).

The project follows a two-phase transfer learning workflow:
1. **Pretrain** — self-supervised (contrastive + denoising) on the encoder
2. **Finetune** — supervised classification using a frozen-then-unfrozen encoder

## Package management

The project uses **uv** exclusively. Do not use `pip` directly.

```bash
uv add <package>           # add a runtime dependency
uv add --dev <package>     # add a dev dependency
uv sync                    # install all dependencies
```

## Development commands

```bash
# Lint / format
uv run ruff check src/           # lint
uv run ruff format src/          # format (single-quote style enforced)

# Tests
uv run pytest tests/             # run all tests
uv run pytest tests/test_foo.py  # run a single file

# Run — install the package first so the entry point is available
uv sync

# All configuration lives in the YAML file.  Use Hydra dot-notation to
# override individual keys without editing the file:
fm4tag --config-name=default                                            # uses default.yaml defaults
fm4tag --config-name=default phase=pretrain action=fit                  # pretrain
fm4tag --config-name=default phase=finetune encoder_ckpt=<path>         # finetune from pretrained encoder
fm4tag --config-name=default phase=finetune ckpt_path=<path>            # resume finetune
fm4tag --config-name=default phase=finetune action=test ckpt_path=<path>
fm4tag --config-name=default phase=finetune action=predict ckpt_path=<path>

# Load a config file from OUTSIDE the repo (--config-path is absolute or relative to CWD):
fm4tag --config-path=/my/external/configs --config-name=my_experiment phase=finetune

# Equivalent using Python module syntax (no install required):
python -m fm4tag.engine --config-name=default phase=pretrain

# The fully-resolved config is always saved alongside the run outputs:
#   outputs/<experiment_name>/version_N/config.yaml   ← use this to reproduce a run
#   outputs/<experiment_name>/version_N/hparams.yaml  ← Lightning hyperparameter log (nested under cfg:)
```

## Architecture

### Data flow (per constituent, e.g. track)

```
HDF5 → DatasetCatCon.__getitem__
      → (label, global_feats, constituents{categorical, continuous, valid})
      → cat_con_collate_fn  →  batch dict with shapes:
          label:       (B,)
          global:      (B, F_g)
          constituents[obj]["categorical"]: (B, C, F_cat)  long
          constituents[obj]["continuous"]:  (B, C, F_con)  float32
          constituents[obj]["valid"]:       (B, C)          bool
```

### Encoding pipeline (inside Lightning modules)

```
raw (B, C, F_cat/F_con)
  → flatten valid constituents → (N_valid, F_cat/F_con)
  → embed_data()               → (N_valid, F_cat, dim), (N_valid, F_con, dim)
         categorical: embeds(x + offset)
         continuous:  grouped Conv1d MLP  [cont_fc1 → relu → cont_fc2]
  → saint_encoder.forward()    → (N_valid, F_cat+F_con, dim)
         (transformer operates on already-embedded tokens)
  → scatter back to (B, C, F, dim) with zeros at invalid positions
  → MultiStreamClassifierHead   → (B, y_dim) logits
```

`MultiStreamClassifierHead` does **not** use a CLS token. It aggregates all F feature tokens per constituent by flattening them, projecting with an MLP, running a cross-constituent transformer, then **masked mean pooling** over valid constituents to get the jet-level representation.

### Key classes

| Class | File | Role |
|---|---|---|
| `DatasetCatCon` | `data/datasets.py` | HDF5 lazy loading, normalisation, padding |
| `PT_FT_DataModule` | `data/datamodule.py` | Lightning DataModule; `phase="pretrain"\|"finetune"` |
| `saint_encoder` | `models/components/encoder.py` | Transformer encoder for tabular constituent data |
| `MultiStreamClassifierHead` | `models/components/heads.py` | Flatten+project per constituent, cross-constituent transformer, masked mean pool → logits |
| `InfoNCELoss` | `models/components/losses.py` | Symmetric NT-Xent contrastive loss |
| `DenoisingLoss` | `models/components/losses.py` | CE (categorical) + MSE (continuous) reconstruction |
| `PretrainModule` | `models/pretrain_module.py` | LightningModule: self-supervised pretraining |
| `FinetuneModule` | `models/finetune_module.py` | LightningModule: supervised fine-tuning |
| `embed_data` | `data/augmentations.py` | Embeds raw indices/floats using encoder's tables |

### Config and engine

Configs live in `src/fm4tag/configs/`. They are plain OmegaConf YAML loaded by Hydra. Required top-level keys: `phase`, `action`, `global_object`, `constituent_objects`, `variables`, `encoder`, `head`, `pretrain`, `optimizer`, `trainer`, `callbacks`. Add a new config by copying `default.yaml` and changing the relevant sections.

`engine.py` is the core engine. The `@hydra.main` decorator makes it the entry point for `fm4tag` / `python -m fm4tag.engine`. Its public function `run(cfg, *, phase, action, encoder_ckpt, ckpt_path)` can also be called directly from notebooks without Hydra.

### BackboneFinetuning callback

When `freeze_encoder: true` in the config, `FinetuneModule.configure_optimizers` only registers the **head** parameters. The `BackboneFinetuning` callback (enabled automatically) finds `FinetuneModule.backbone` and adds its parameters to the optimiser at epoch `unfreeze_backbone_at_epoch` with a reduced learning rate (`backbone_initial_ratio_lr × current_lr`).

### Checkpoints and outputs

Lightning's TensorBoardLogger writes to `outputs/<experiment_name>/version_<N>/`.
`ModelCheckpoint` saves to `…/checkpoints/`.
`predict-classifier` saves `predictions.pt` (list of softmax tensors) in the same log dir.
`engine.run()` always writes `config.yaml` (fully resolved) to the log dir — use this to reproduce a run or to pass back to `--config-path` / `--config-name`.

To visualise training metrics, launch TensorBoard pointing at the outputs directory:
```bash
tensorboard --logdir outputs/
# then open http://localhost:6006 in a browser
# on a remote machine, forward the port first:
#   ssh -L 6006:localhost:6006 yourserver
```
