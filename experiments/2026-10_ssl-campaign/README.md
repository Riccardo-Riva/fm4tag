# SSL campaign: does contrastive pretraining help? (2026-10-16 → 2026-11-12)

**Status:** closed (code: tag `exp/ssl-wsd-lr-search`); its pretrainings `P_full_*` are used by the later experiments.

## Question
Does self-supervised contrastive pretraining of the encoders (P0, P1, P2) help training from scratch (SC), fine-tuning (FT) or frozen encoders (FR), with and without a downstream JC term?

## Data
`data_parallel_preprocess_251112_152051`: pretrain 39.68 M (labels unused) / train 18.6 M / val 1.86 M / test 1.86 M jets, disjoint by event number (duplicate check of 2026-10-28: no jet shared between these splits; about 2 % of training rows are resampled copies, 24 % for tau).

## Model
**Model B (`encoders=rowcol_concat_v0`, `chunk_size=512`, 7.47 M parameters).** As model A, but the row
(intersample) attention is **full softmax attention among the objects of the same 512-jet chunk** over the flattened
token grid (`row_mode: concat`, SAINT style), with **QK RMS-norm** (added 2026-10-22 against logit blow-ups) and a
zero-initialised out-projection.  Jets: attention among the 512 jets of a chunk; tracks: among all valid tracks of
those jets (about 3.7k).  Same aggregator (6 layers over each jet's tracks, mean pool) and MLP head (hidden 256).

## Training methods
**Pretraining** (30 epochs, lr 1e-3, batch 12288, gradient clip 1.0, seeds 42 / 424): P0 = per-object contrastive on the track and jet-feature embeddings (T 0.1, λ 0.4); P1 = P0 + JC (weight 1.0) on the aggregator output (aggregator trained and loaded downstream); P2 = JC only. **Downstream**: SC (all random), FT (pretrained encoders at lr/10 after a 2-epoch hold), FR (encoders frozen, aggregator + head trained); CE, or CE + 0.3 JC (λ 0.4, T 0.1). Finals: 40-epoch cosine, early stopping (patience 12), lr and batch size tuned per arm on seed 42 (S3), 4 runs per arm; unweighted CE.

## Stages and configs
Cells files in `configs/` (stage = file name): `smoke_*`, `S1_arch`, `S2_*`, `P_full`, `S3_*`, `F_final`, `C_gap`, `F_scjc_*`, `F_p1_nojc`, `L_*`; each header holds the design and the launch command. Launcher: the generic `slurm/ssl_pack.sh` (`slurm/ssl_campaign.sh` = the first, per-cell launcher). Test metrics / ROC: `slurm/arm_test_metrics_job.sh`, `slurm/arm_roc_job.sh`. Notes: `notes/ssl_campaign.md` (decision log), `notes/ssl_campaign_report.md`, `notes/ssl_tests_plan.pptx`; decks in `slides/`; diagnostics (NaN, spikes, varlen limits, duplicate check) in `diagnostics/`.

## Results
Best val CE (4 runs): SC 0.6240 < FT+JC←P1 0.6288 < FT←P0 0.6312 < FR+JC←P1 0.6436 < FR←P0 0.6500; controls: FT←P1 CE-only 0.6158 (test c-rej 10.8 vs SC 9.7, GN2v01 15.6), SC+JC stable only with clip 1.0 (= SC); label-fraction scan: pretraining never beats SC at 1–100 % labels.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | cells files (one per stage) and the YAML configs of the plot / analysis scripts | yes |
| `slurm/` | launchers specific to this experiment (generic ones are in the repo's `slurm/` and `scripts/`) | yes |
| `notes/`, `diagnostics/` | design notes, reports, one-off diagnostic scripts (where present) | yes |
| `outputs/` | run folders (`<STAGE>_<timestamp>/<cell>/seed_<s>/`: `.hydra/`, `metrics.csv`, checkpoints, `run_info.yaml`) and logs | no |
| `plots/` | every figure and metric table of the experiment | no |

## Related experiments
Continued by `2026-11_dropout`, `2026-11_wsd-lr-search`, `2026-11_four-way-jc-dropout`; architecture from `2026-07_kernel-profiling`.
