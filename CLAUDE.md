# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**fm4tag** is a PyTorch Lightning library for pretraining and fine-tuning SAINT-based transformer encoders on mixed categorical/continuous particle physics data (jets + tracks from HDF5 files), targeting jet flavour tagging at the LHC.

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
fm4tag --config-name=saintV0                                            # uses saintV0.yaml defaults
fm4tag --config-name=saintV0 phase=pretrain action=fit                  # pretrain
fm4tag --config-name=saintV0 phase=finetune encoder_ckpt=<path>         # finetune from pretrained encoder
fm4tag --config-name=saintV0 phase=finetune ckpt_path=<path>            # resume finetune
fm4tag --config-name=saintV0 phase=finetune action=test ckpt_path=<path>
fm4tag --config-name=saintV0 phase=finetune action=predict ckpt_path=<path>

# Equivalent using Python module syntax (no install required):
python -m fm4tag.engine --config-name=saintV0 phase=pretrain
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
  → ClassifierHead              → (B, y_dim) logits
```

Token at `F`-index 0 (first categorical feature) plays the role of a CLS token:
- **ClassifierHead** extracts it as the per-constituent summary
- **DenoisingLoss** skips reconstructing it (only reconstructs indices 1…F_cat−1)

### Key classes

| Class | File | Role |
|---|---|---|
| `DatasetCatCon` | `data/datasets.py` | HDF5 lazy loading, normalisation, padding |
| `PT_FT_DataModule` | `data/datamodule.py` | Lightning DataModule; `phase="pretrain"\|"finetune"` |
| `saint_encoder` | `models/components/encoder.py` | Transformer encoder for tabular constituent data |
| `ClassifierHead` | `models/components/heads.py` | Cross-constituent attention + pooling → logits |
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

Lightning's CSVLogger writes to `outputs/<experiment_name>/version_<N>/`.
`ModelCheckpoint` saves to `…/checkpoints/`.
`predict-classifier` saves `predictions.pt` (list of softmax tensors) in the same log dir.

### Reference implementation

The older pure-PyTorch implementation lives at
`/storage3/DSIP/rriva/research/ftag_data_embedding/tagger_pt/`
and is useful for cross-checking loss computations and model behaviour.
