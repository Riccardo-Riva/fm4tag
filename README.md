# FM4tag

Foundation-model-based transfer learning for particle jet flavour tagging in
high-energy physics. FM4tag pretrains transformer encoders on jet data with a
multi-view contrastive + denoising self-supervised objective, then fine-tunes
them (encoders + aggregator + classifier head) for flavour tagging.

```
            per-object encoders          shared aggregator        head
jets    ──► GlobalTransformerEncoder ─┐
tracks  ──► Encoder (col/row/rowcol) ─┤─► TransformerAggregator ─► MLP ─► logits
(other) ──► Encoder ...              ─┘        z_jet (B, D)
```

- **Pretraining** (`PretrainModule`): per-object contrastive loss on
  projections, categorical/continuous denoising reconstruction, and a
  jet-level contrastive loss on the aggregator output.
- **Fine-tuning** (`FinetuneModule`): cross-entropy on the head (+ optional
  jet-contrastive term), reusing pretrained encoder/aggregator weights, or
  trained from scratch.

Everything — data, architecture, augmentations, optimiser, Lightning trainer,
Optuna HPO — is driven by one Hydra config
([`default.yaml`](src/fm4tag/configs/default.yaml)), overridable on the
command line.

## Quick start

```bash
uv sync                          # creates .venv from pyproject.toml / uv.lock
source .venv/bin/activate
pytest -m "not ddp" -q           # quick check (DDP tests: pytest tests/ddp)

fm4tag phase=pretrain action=fit                       # pretrain
fm4tag phase=finetune action=fit \
    encoder_ckpt=outputs/.../checkpoints/best.ckpt      # finetune from a checkpoint
fm4tag phase=finetune action=fit                        # finetune from scratch

fm4tag-eval --config-name=my_config ckpt_path=outputs/.../best.ckpt   # representation quality
fm4tag-hpo hpo.n_trials=50                                            # Optuna sweep
```

See [docs/usage.md](docs/usage.md) for the full guide (data format, config
sections, checkpoint knobs, logging conventions, HPO, gotchas).

## Project layout

| Path | Contents |
|---|---|
| [src/fm4tag/](src/fm4tag) | Library: datamodules/datasets, encoders/aggregator/heads, losses, augmentations, callbacks, the Hydra `runner`, and configs |
| [scripts/](scripts) | Standalone analysis scripts (ROC comparisons vs. reference taggers, f_c/f_tau working-point search, permutation importance, timing/profiling) and their YAML configs |
| [slurm/](slurm) | Cluster job scripts for pretrain/finetune/classify-from-scratch, incl. seed-repeated parameter scans |
| [notebooks/](notebooks) | Interactive exploration and plotting |
| [tests/](tests) | Unit tests + `tests/ddp/` (CPU, gloo-backend DDP correctness tests) |
| [docs/](docs) | Usage guide, encoder data-flow, parameter-scan guide |

## Documentation

- [docs/usage.md](docs/usage.md) — full usage guide: data format, config
  reference, running pretrain/finetune/eval/HPO, logging conventions, known
  gotchas.
- [docs/encoder_dataflow.md](docs/encoder_dataflow.md) — how a per-object
  encoder turns raw features into tokens, embeddings, and reconstructions
  (col / row / rowcol backbones).
- [docs/param_scan_guide.md](docs/param_scan_guide.md) — running seed-repeated
  hyperparameter scans on Slurm and plotting the results.
- [tests/README.md](tests/README.md) — running the DDP test suite and what
  each test guards against.

## Requirements

Python >= 3.13, managed with [uv](https://docs.astral.sh/uv/). Key
dependencies: PyTorch Lightning, Hydra/OmegaConf, Optuna, h5py, wandb,
[puma-hep](https://github.com/umami-hep/puma) for ATLAS-style plots.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
