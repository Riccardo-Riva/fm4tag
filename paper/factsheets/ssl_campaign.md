# Fact sheet: SSL campaign (paper pillar 2), state on 2026-11-17

Scope: contrastive pretraining (P0, P1, P2) of the fm4tag encoders, then downstream training of the flavour tagger by
three methods: from scratch (SC), fine-tuning (FT), frozen encoders (FR), each with or without the downstream
jet-embedding contrastive term (JC). Worktree `/storage3/DSIP/rriva/research/fm4tag-chunk`, branch `feat/chunked-ia`
(local, never pushed). Run dirs: `/storage3/DSIP/rriva/research/fm4tag-chunk/slurm/ssl_campaign/<stage>_<ts>/<cell>/seed_<s>/`.

Provenance tags: **[F]** verified in files (resolved `.hydra/config.yaml`, `metrics.csv`, `run_info.yaml`, csv summaries,
code, dataset preprocessing yaml); **[M]** memory note
`/home/rriva/.claude/projects/-storage3-DSIP-rriva-research-fm4tag/memory/fm4tag-ssl-campaign.md`; **[C]** chat
transcript `/home/rriva/.claude/projects/-storage3-DSIP-rriva-research-fm4tag/88778f66-fe90-4305-896d-7e54372b655a.jsonl`
(line number); **[I]** inferred.

Extraction scripts I used (read-only): `/tmp/claude-1138/-storage3-DSIP-rriva-research-fm4tag/653cf529-153e-437a-bcf0-9bb91ed33820/scratchpad/ssl_campaign/extract.py`
(dumps every run's resolved config + metrics summary to `.../scratchpad/ssl_campaign/runs.jsonl`, 223 runs),
`group.py`, `compact.py` in the same folder.

Conventions used below
- "best val CE" = minimum over epochs of `val/head/loss_ce` (unweighted cross-entropy, `use_class_weights: false`, on the
  full 1.86M-jet val split, bf16-mixed); the evaluated checkpoint is that epoch's. Mean ± sample std (n−1) over the runs.
- Epoch numbers are **0-indexed** as in `metrics.csv` (epoch 9 = 10th epoch).
- Test rejections: test split, cuts 20 < pT < 250 GeV, |η| < 2.5; b-tag discriminant D_b with fractions f_c, f_τ.
  "Nominal" = f_c 0.2, f_τ 0.01 (GN2v01's). "Optimised" = per run, grid search on **validation** (step 0.002, f_c ∈ [0,0.15),
  f_τ ∈ [0,0.12)) maximising c-rejection at **80%** b-eff subject to light- and τ-rejection ≥ GN2v01's at 80%; if no
  point satisfies both, the point maximising min(u_rej/floor_u, τ_rej/floor_τ) is taken and the run is flagged
  infeasible. Rejections then quoted on **test at 77%** b-eff. [F] `scripts/configs/arm_test_metrics.yaml`, `scripts/arm_test_metrics.py:125-150`.
- GN2v01 = ATLAS production tagger scores stored as columns in the test file (not trained on our 18.6M jets: yardstick
  only). The salt GN2 retrained on dataset 112022 is NOT usable here (its train set overlaps this test set). [F] campaign_ssl.yaml comment; [M]

## 0. Data (verified in preprocessing yaml)

Dataset `/storage3/DSIP/rriva/datasets/transforming-jet-flavor/data_parallel_preprocess_251112_152051`, tt̄ (√s = 13.6 TeV
per dataset README [M]), jets 20 < pT < 250 GeV, kinematic resampling `countup` to the c-jet (pT, |η|) distribution, fixed
flavour counts per split [F] `*/output/base_train.yaml`:

| split | eventNumber % 100 | b | c | light | τ | total |
|---|---|---|---|---|---|---|
| pretrain | 0–63 | 10.24M | 10.24M | 15.36M | 3.84M | 39.68M |
| train | 64–93 | 4.80M | 4.80M | 7.20M | 1.80M | 18.60M |
| val | 94–96 | 0.48M | 0.48M | 0.72M | 0.18M | 1.86M |
| test | 97–99 | 0.48M | 0.48M | 0.72M | 0.18M | 1.86M |

- Composition in every split: b 25.8%, c 25.8%, light 38.7%, τ 9.7%. CE at the class priors = 1.29 (= the "collapsed" value 1.287-1.289). [F]/[I]
- The pretraining split is "unlabelled" only in the loss: its flavour mix was set by truth-label resampling. [F]/[I]
- Train split has ~2% duplicates from countup (18.24M distinct jets); no overlap between splits; ~43% of every 152051 split is in the
  old 112022 train set (λ-scan pillar data) → old checkpoints cannot be evaluated here. [M] (slurm/smoke/dup_check.py)
- Normalisation from the train split for all phases (`norm_dict_path: ${base_path}/train/...`). [F] P0 config.
- Inputs: jets pt_btagJes, eta_btagJes (2 features); tracks 10 continuous + 9 categorical hit-count features. [F] config.

## 1. Model (all campaign runs)

`encoders=rowcol_concat_v0` [F]: per-object encoders that embed every feature individually (dim 16), then one "rowcol" stack of
depth 6 = column (inter-feature) attention + row (inter-sample) attention in **SAINT "concat" mode** (row attention over the
concatenated feature vector), row attention all-pairs **within chunks of 512 jets** (`chunk_size 512`; tracks attend to the tracks
of the jets in their chunk), **QK-norm** (RMSNorm on q, k per head, `row_qk_norm: true`, every arm from 2026-10-22), ff_mult 2;
tracks: 1 column head, 8 row heads × 32; jets: 1 column head, 1 row head. Projectors to 128-d (`proj_out 128`) = the per-object
embeddings `tracks_embedding`, `jets_embedding` (the pretraining loss acts on these). Aggregator (GN2-like): transformer depth 6,
4 heads × 64, ff_mult 4, over tracks only (packed, valid tracks), 256-d output = "global jet embedding" (`aggregator`). Head:
MLP 256 hidden, dropout 0.02, 4 classes. [F] config; aggregator attends over tracks only, jet features concatenated after pooling [M].
Parameters (Lightning summary, `rank_0.err`): track encoder 5.3M, jet encoder 92.9K, aggregator 2.0M, head 66.8K, total 7.5M.
P0 pretraining: 5.4M trainable / 2.0M non-trainable (aggregator unused). FR: 2.0M trainable / 5.4M frozen. [F]
Optimiser for all: AdamW, weight decay 5.96e-5, warmup 2% then cosine to 0 over max_epochs (step-wise), bf16-mixed, 1 GPU. [F]

Augmented views (pretraining and downstream JC), λ = augmentation strength: view 1 = CutMix (each raw feature cell replaced from a
shuffled row with probability λ) + Mixup in embedding space (x' = (1−λ)x + λ x_perm); view 2 = identity (clean). [F] config +
`src/fm4tag/augmentations/{cutmix,mixup}.py`.
Contrastive loss = `MultiViewSupConLoss` with no labels (positives = the 2 views of the same object; negatives = other objects' other
view, `include_same_view_negatives: false`, positive in denominator, L_out), restricted to the anchor's 512-jet chunk, fp32
similarities. [F] `src/fm4tag/losses/contrastive.py`, `modules/pretrain_module.py`. Downstream JC = the same loss on the
aggregator outputs of the two views; CE always on the clean pass. [F] `modules/finetune_module.py:310-345`.

## 2. TABLE A: pretraining (final runs; verified against `.hydra/config.yaml` + `metrics.csv` of all 6 runs)

| | P0 | P1 | P2 |
|---|---|---|---|
| Run dir | `P_full_20261023_085735/p0/seed_{42,424}` | `P_full_20261026_094445/p1/seed_{42,424}` | `P_full_20261023_085735/p2/seed_{42,424}` |
| Data | pretrain split, 39.68M jets, labels not in loss | same | same |
| Loss terms (weights) | per-object contrastive on tracks_embedding + jets_embedding (1.0 each, summed) | P0 + JC on aggregator output (1.0) | JC on aggregator only (1.0); per-object 0 |
| Denoising | 0 (cat 0, con 0) | 0 | 0 |
| Trained | encoders + projectors; aggregator NOT trained (no gradient) | encoders + projectors + aggregator | encoders + projectors + aggregator (via JC) |
| λ (CutMix+Mixup) | 0.4 | 0.4 | 0.4 |
| Temperature | 0.1 | 0.1 (both losses) | 0.1 |
| Batch / chunk | 12288 / 512 (24 chunks) | 12288 / 512 | 12288 / 512 |
| Peak LR, schedule | 1e-3, 2% warmup + cosine, 30 epochs (~3229 steps/epoch), last.ckpt used downstream | same | same |
| Grad clipping | global norm 1.0 | 1.0 | 1.0 |
| QK-norm | on | on | on |
| Seeds | 42, 424 | 42, 424 | 42, 424 |
| Wall time | 43.4 h each (A40, node82) | 34.2 h each (L40S, node101) | 42.6-42.8 h (A40) |
| val total loss, last epoch (s42 / s424) | 5.442 / 5.505 | 5.668 / 5.729 | 0.2224 / 0.2232 |
| val L_track / L_jetfeat / L_JC (s42) | 2.468 / 2.975 / – | 2.466 / 2.971 / 0.231 | – / – / 0.222 |
| eff. rank tracks_embedding (of 128) | 121.4 / 113.5 | 126.1 / 122.4 | 58.7 / 60.7 |
| eff. rank jets_embedding (of 128; 2 input features) | 62.3 / 62.6 | 61.7 / 62.7 | 1.00 / 1.98 (collapsed) |
| eff. rank aggregator (of 256) | 174 / 172 (random, untrained: not meaningful) | 129.8 / 129.4 | 127.2 / 126.4 |
| uniformity tracks / jets / agg (s42) | −3.92 / −3.85 / (−3.88 random) | −3.93 / −3.85 / −3.93 | −0.50 / 0.00 / −3.93 |
| max pre-clip grad-norm window | 38 / 30 | 53 / 57 | 67 / 67 |
| Track row-attn max logit / min entropy, last epoch | 3568, 5952 / 0.69-0.72 | 2885, 2696 / 0.70-0.71 | 21.7 / 0.77 |
| Used downstream | FT←P0, FR←P0, label scan, dropout runs | FT/FR←P1 (CE only and +JC) | none (rriva 2026-10-26: no P2 arm "for now") |

Notes: eff. rank / uniformity on 8192 val samples (`EmbeddingMetrics n_samples 8192`) [F]. Pretraining train ≈ val loss throughout [M].
The P0/P1 val loss was still decreasing at epoch 29 (best epoch 28-29) [F]. Early-stopping callback (patience 12 on val/loss) present
but never triggered [F]. P2's jet-feature embedding collapses to rank ~1 (it receives no direct contrastive signal) [F].

## 3. TABLE B: downstream methods (final stage; every cell verified against the resolved `.hydra/config.yaml` of all 4 runs)

Common to all rows unless stated [F]: train split 18.6M jets (100%), val 1.86M; AdamW wd 5.96e-5; warmup 2% + cosine over
max_epochs; early stopping patience 12 on val CE and best-val-CE checkpoint; chunk 512; QK-norm on; JC (when on) weight 0.3, T 0.1, λ 0.4,
views [CutMix+Mixup, identity]; head dropout 0.02, all other dropouts 0; `load_aggregator: auto` (aggregator loaded only if the
pretraining trained it); bf16. FT: pretrained parts in a separate "backbone" param group at backbone_lr, held at lr 0 for the
first 2 epochs (`unfreeze_at_epoch 2`), then on the head's cosine. FR: encoders + projectors frozen (no grad, eval mode); a loaded
(P1) aggregator trains at backbone_lr = lr. Seeds: pretrained arms = pretrain seed {42, 424} × downstream seed {4242, 42424};
scratch arms = {4242, 42424, 424242, 4242424}. Seed 42 was used only for LR selection (S3) and is excluded from finals.

| Method | Stage/cell | Init | Trainable / frozen | Loss | Batch | LR (backbone) | Max ep | Clip | Dropout |
|---|---|---|---|---|---|---|---|---|---|
| SC | F_final_20261028_235642/sc | random | all trainable | CE | 28672 | 1.1e-3 | 40 | none | 0 (head 0.02) |
| SC+JC (no clip) | F_final_20261028_235642/scjc | random | all | CE + 0.3 JC | 14336 | 8e-4 | 40 | none | 0 |
| SC+JC clip 5 | F_scjc_clip_20261103_165640/scjc_clip5 | random | all | CE + 0.3 JC | 14336 | 8e-4 | 40 | 5.0 | 0 |
| SC+JC clip 1 (**adopted baseline**) | F_scjc_stab_20261109_175800/scjc_clip1 | random | all | CE + 0.3 JC | 14336 | 8e-4 | 40 | 1.0 | 0 |
| SC+JC clip 1 + warmup 10% | F_scjc_stab_.../scjc_clip1_wu10 | random | all | CE + 0.3 JC | 14336 | 8e-4, warmup_frac 0.1 | 40 | 1.0 | 0 |
| SC+JC clip 1 + lr 4e-4 | F_scjc_stab_.../scjc_clip1_lr4 | random | all | CE + 0.3 JC | 14336 | 4e-4 | 40 | 1.0 | 0 |
| FT←P0 | F_final_20261028_235640/ft_p0s{42,424} | P0 last.ckpt: encoders+projectors; aggregator random | all; encoders at lr/10 after 2-ep hold; aggregator+head at lr | CE | 28672 | 1.1e-3 (1.1e-4) | 40 | none | 0 |
| FR←P0 | F_final_.../fr_p0s{42,424} | P0 encoders; aggregator random | aggregator + head; encoders+projectors frozen | CE | 94208 | 2e-3 | 40 | none | 0 |
| FT←P1 (CE only) | F_p1_nojc_20261109_175801/ft_p1s{42,424} | P1: encoders+projectors+aggregator | all; encoders+aggregator at lr/10 after 2-ep hold | CE | 28672 | 1.1e-3 (1.1e-4) | 40 | none | 0 |
| FR←P1 (CE only) | F_p1_nojc_.../fr_p1s{42,424} | P1 encoders + aggregator | aggregator (loaded, at lr) + head | CE | 94208 | 2e-3 (aggregator 2e-3) | 40 | none | 0 |
| FT←P1+JC | F_final_.../ftjc_p1s{42,424} | P1 encoders + aggregator | all; pretrained at lr/10 after 2-ep hold | CE + 0.3 JC | 14336 | 8e-4 (8e-5) | 40 | none | 0 |
| FR←P1+JC | F_final_.../frjc_p1s{42,424} | P1 encoders + aggregator | aggregator (loaded, at lr) + head | CE + 0.3 JC | 53760 | 1.5e-3 | 40 | none | 0 |
| SC+drop | D_drop_20261113_215501/sc_d02 | random | all | CE | 25600 | 1.1e-3 | 80 | none | 0.2 everywhere* |
| SC+JC+drop | D_drop_.../scjc_clip1_d02 | random | all | CE + 0.3 JC | 12800 | 8e-4 | 80 | 1.0 | 0.2* |
| FT←P0+drop | D_drop_20261113_215456/ft_p0s{42,424}_d02 | as FT←P0 | as FT←P0 | CE | 25600 | 1.1e-3 (1.1e-4) | 80 | none | 0.2* |
| FR←P0+drop | D_drop_.../fr_p0s{42,424}_d02 | as FR←P0 | as FR←P0 (encoders in eval: dropout only in aggregator + head) | CE | 94208 | 2e-3 | 80 | none | 0.2* |
| FT←P1+JC+drop | D_drop_.../ftjc_p1s{42,424}_d02 | as FT←P1+JC | as FT←P1+JC | CE + 0.3 JC | 12800 | 8e-4 (8e-5) | 80 | none | 0.2* |
| FR←P1+JC+drop | D_drop_.../frjc_p1s{42,424}_d02 | as FR←P1+JC | as FR←P1+JC (encoder dropout inert) | CE + 0.3 JC | 53760 | 1.5e-3 | 80 | none | 0.2* |

*dropout 0.2 on: encoder rowcol block attn + ff (jets and tracks), aggregator attn + ff, head MLP. Pretrainings used for the
dropout runs are the no-dropout P0/P1 checkpoints. Not run: FT←P1 / FR←P1 CE-only with dropout. [F] configs; [C] line 15616.

LR selection (S3, seed 42, batch per arm filled to ~92% of a 45 GiB card, lr grid ×2 around 3e-4·√(B/2048)) and the approved
final LRs (rriva 2026-10-28) [F]/[M]: see §6.

Arm-specific facts [F]: FR←P0 has only one optimiser group (head = aggregator+head; aggregator random); FR←P1 and FT←P1 have a
"backbone" group (loaded aggregator / encoders+aggregator). Batch sizes differ across arms because each was sized to fill the
GPU (JC arms need 2 encoder passes → half batch; FR has no encoder backward → large batch).

## 4. TABLE C (the "kinds of training with P0 and P1" table the author asked for)

Best val CE, mean ± std over 4 runs (2 pretraining × 2 training seeds for pretrained, 4 seeds for scratch), full labels.

| Pretraining | Method | Downstream loss | No dropout (40-ep cosine) | Dropout 0.2 (80-ep cosine) | Label-fraction scan | In WSD redo (W_lr) |
|---|---|---|---|---|---|---|
| none | SC | CE | 0.6240 ± 0.0029 | 0.6407 ± 0.0138 | yes (1,3,10,30%) | sc, sc_d02 |
| none | SC | CE + JC | 0.6241 ± 0.0063 (clip 1.0; 4 other recipes §5.3) | 0.7533 ± 0.0072 | no | scjc, scjc_d02 |
| P0 | FT | CE | 0.6312 ± 0.0007 | **0.6105 ± 0.0004** | yes | ft_p0, ft_p0_d02 |
| P0 | FR | CE | 0.6500 ± 0.0008 | 0.6637 ± 0.0034 (best at ep 74-78 of 80) | yes | fr_p0, fr_p0_d02 |
| P0 | FT / FR | CE + JC | not run by design (P0 aggregator untrained) | not run | no | no |
| P1 | FT | CE | **0.6158 ± 0.0002** (best ep 34-39 of 40) | not run | no | ft_p1, ft_p1_d02 |
| P1 | FR | CE | 0.6361 ± 0.0031 | not run | no | fr_p1, fr_p1_d02 |
| P1 | FT | CE + JC | 0.6288 ± 0.0002 (best ep 36-39) | 0.7400 ± 0.0300 | no | ftjc_p1, ftjc_p1_d02 |
| P1 | FR | CE + JC | 0.6436 ± 0.0024 | 0.7854 ± 0.0149 | no | frjc_p1, frjc_p1_d02 |
| P2 | any | any | not run (only 15-epoch frozen probes in S2 / S2_qk) | – | – | – |

Reading [F]/[I]: the only method that beats SC on val CE is FT←P1 CE-only (−0.0082, ~2.8σ of SC's seed spread);
FT←P0 is worse than SC (+0.0072); FR always worse; P1 beats P0 for both FT (−0.0154) and FR (−0.0139) at the same recipe;
adding downstream JC hurts given P1 (FT +0.0130, FR +0.0075; batch/lr also differ); JC from scratch gives no gain.

## 5. RESULTS

### 5.1 Best val CE per method (no dropout) and test CE [F]

| Method | best val CE (mean ± std, n=4) | range | best epochs (0-idx) | test CE (mean) | stopped at |
|---|---|---|---|---|---|
| SC | 0.6240 ± 0.0029 | 0.6217-0.6282 | 9, 8, 9, 11 | 0.6236 | ES ep 20-23 |
| SC+JC (clip 1.0) | 0.6241 ± 0.0063 | 0.6208-0.6335 | 9, 12, 10, 8 | 0.6237 | ES 20-24 |
| FT←P0 | 0.6312 ± 0.0007 | 0.6303-0.6320 | 5, 4, 5, 4 | 0.6309 | ES 16-17 |
| FR←P0 | 0.6500 ± 0.0008 | 0.6488-0.6507 | 6, 5, 6, 7 | 0.6498 | ES 17-19 |
| FT←P1 (CE) | 0.6158 ± 0.0002 | 0.6156-0.6161 | 38, 34, 39, 36 | 0.6154 | 40 (cut off) |
| FR←P1 (CE) | 0.6361 ± 0.0031 | 0.6335-0.6397 | 20, 24, 19, 20 | 0.6357 | ES 31-36 |
| FT←P1+JC | 0.6288 ± 0.0002 | 0.6286-0.6290 | 38, 39, 36, 39 | 0.6287 | 40 (cut off) |
| FR←P1+JC | 0.6436 ± 0.0024 | 0.6408-0.6457 | 27, 24, 22, 24 | 0.6437 | ES 34-39 |

Pretraining-seed effect: FT←P0 s42 0.6316 vs s424 0.6308; FR←P1 s42 0.6335 vs s424 0.6387; FR+JC←P1 0.6415 vs 0.6456 [F].
Test CE equals best val CE within 5e-4 in every arm [F per_arm.csv].

### 5.2 Test rejections at 77% b-eff [F] `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/arm_comparison/test_metrics/per_run.csv`

Mean ± std over 4 runs.

| Method | Opt. c-rej | Opt. light-rej | Opt. τ-rej | feasible | mean f_c / f_τ | Nom. c-rej | Nom. light-rej | Nom. τ-rej |
|---|---|---|---|---|---|---|---|---|
| GN2v01 (yardstick) | 15.58 | 437.6 | 67.6 | – | 0.2 / 0.01 | 15.58 | 437.6 | 67.6 |
| SC | 9.69 ± 0.57 | 433.6 ± 4.3 | 65.1 ± 4.0 | 4/4 | 0.038 / 0.045 | 11.99 ± 0.45 | 256.5 ± 10.5 | 37.3 ± 4.2 |
| SC+JC (clip 1) | 9.79 ± 0.92 | 437.2 ± 14.6 | 65.5 ± 3.5 | 4/4 | 0.040 / 0.048 | 11.97 ± 0.48 | 264.4 ± 23.7 | 36.7 ± 3.0 |
| FT←P0 | 8.26 ± 0.39 | 452.2 ± 7.6 | 63.8 ± 0.1 | 4/4 | 0.024 / 0.051 | 11.05 ± 0.28 | 239.4 ± 7.7 | 34.7 ± 0.7 |
| FR←P0 | 4.66 ± 0.05 | 440.7 ± 7.1 | 55.3 ± 1.3 | **0/4 (f_c=0)** | 0.000 / 0.072 | 9.71 ± 0.09 | 193.0 ± 4.8 | 26.2 ± 3.2 |
| FT←P1 (CE) | **10.78 ± 0.10** | 436.7 ± 13.6 | 62.9 ± 0.9 | 4/4 | 0.055 / 0.038 | **12.62 ± 0.06** | **284.7 ± 1.9** | **39.5 ± 0.5** |
| FR←P1 (CE) | 7.42 ± 0.41 | 446.8 ± 8.2 | 65.4 ± 1.6 | 4/4 | 0.018 / 0.077 | 10.60 ± 0.43 | 237.3 ± 11.4 | 31.1 ± 2.9 |
| FT←P1+JC | 8.97 ± 0.06 | 441.8 ± 12.0 | 65.0 ± 1.3 | 4/4 | 0.032 / 0.055 | 11.26 ± 0.02 | 256.5 ± 3.4 | 36.3 ± 0.3 |
| FR←P1+JC | 6.41 ± 0.65 | 470.9 ± 22.8 | 63.0 ± 2.1 | 4/4 | 0.009 / 0.068 | 10.05 ± 0.22 | 216.2 ± 6.0 | 31.2 ± 2.0 |

Ratio to SC mean (optimised, c/light/τ): FT←P1 1.112/1.007/0.966; FT←P0 0.852/1.043/0.980; FR←P1 0.766/1.030/1.004;
FT←P1+JC 0.926/1.019/0.998; FR←P1+JC 0.662/1.086/0.967; SC+JC 1.010/1.008/1.005. Nominal: FT←P1 1.052/1.110/1.057 [F].
No campaign model reaches GN2v01's c-rejection (best 10.8 vs 15.6 at equal light/τ); with nominal fractions GN2v01's light
and τ rejections are ~1.5-1.8× higher. GN2v01 is not a same-data baseline [M]/[I].
Inputs to the score: scored runs are F_final + F_p1_nojc + F_scjc_stab/scjc_clip1 (collapsed SC+JC runs never scored) [F config].

### 5.3 SC+JC instability (collapse statistics) [F]/[M]

| Recipe | seeds 4242 / 42424 / 424242 / 4242424 best val CE | collapsed |
|---|---|---|
| no clip (F_final) | 0.6439@3 / 0.6498@3 / 0.6211@10 / 0.7127@0 | 3/4 (4242, 42424, 4242424) |
| clip 5.0 | 0.6208@11 / 0.6347@11 / 0.6631@1 / 0.6351@3 | 2/4 (424242, 4242424) |
| clip 1.0 | 0.6210 / 0.6335 / 0.6208 / 0.6210 | 0/4 → adopted SC+JC baseline (pre-registered rule) |
| clip 1.0 + 10% warmup | 0.6226 / 0.6225 / 0.6217 / 0.6961@38 | 0/4 by rule, but 4242424 damaged (val 0.954 at ep 1, track uniformity −0.48, recovers to 0.696) |
| clip 1.0 + lr 4e-4 | 0.6275 / 0.6369 / 0.6219 / 0.6224 | 0/4 (mean 0.6272 ± 0.0070) |

Collapse rule (pre-registered in F_scjc_stab.cells): val CE > 0.85 after a best epoch within the first 6. Signature [F]: within
one epoch val CE → 1.288 (class priors), val tracks_embedding uniformity → 0.00 (track embeddings collapse to a point), aggregator
eff. rank 135 → 24-41; pre-clip grad-norm window 27-77 vs ~1-2 normal; never recovers (val 0.75-1.29 afterwards). Row-attention
logits stay 5-9 (QK-norm holds), so not the pretraining logit mechanism [M]. Global-norm clipping at 5 under AdamW does not bound
the update. Overall 5 of 8 runs collapsed without/with clip 5; 0 of 4 with clip 1.0 (1 of 12 clip-1.0-family runs damaged).
Same JC loss was stable in P1 pretraining (clip 1.0, 2 seeds). Healthy SC+JC = SC on val CE (0.6241 vs 0.6240) and on test
rejections (§5.2) → no evidence JC helps from scratch in this setup.

### 5.4 Label-fraction scan (P0 only) [F] `L_scan_20261105_151751` + F_final for 100%

Nested subsets = first N rows of the pre-shuffled train file (`datamodule.train_max_rows`); val always full 1.86M; pretrained arms
from P0 seed 42 only (100% point: 2 pretraining seeds); 3 seeds (4242, 42424, 424242) per point (4 at 100%); no collapsed run.

| Labels | rows | batch | max ep | lr SC / FT (bb lr/10) / FR | SC | FT←P0 | FR←P0 | FT−SC | FR−SC |
|---|---|---|---|---|---|---|---|---|---|
| 1% | 186,000 | 4096 | 150 | 8.4e-4 / 2.1e-4 / 8.4e-4 | 0.7443 ± 0.0060 | 0.8079 ± 0.0009 | 0.8305 ± 0.0016 | +0.064 | +0.086 |
| 3% | 558,000 | 8192 | 100 | 1.2e-3 / 3e-4 / 1.2e-3 | 0.7163 ± 0.0020 | 0.7691 ± 0.0021 | 0.7891 ± 0.0036 | +0.053 | +0.073 |
| 10% | 1,860,000 | 28672 | 100 | 1.1e-3 / 5.5e-4 / 2.2e-3 | 0.6919 ± 0.0001 | 0.7253 ± 0.0014 | 0.7500 ± 0.0020 | +0.033 | +0.058 |
| 30% | 5,580,000 | 28672 | 60 | 1.1e-3 / 5.5e-4 / 2.2e-3 | 0.6640 ± 0.0014 | 0.6759 ± 0.0005 | 0.7064 ± 0.0011 | +0.012 | +0.042 |
| 100% | 18.6M | 28672 / 28672 / 94208 | 40 | 1.1e-3 / 1.1e-3 / 2e-3 | 0.6240 ± 0.0029 | 0.6312 ± 0.0007 | 0.6500 ± 0.0008 | +0.007 | +0.026 |

Best epochs: FR 2-3, FT 4-6, SC 5-14 (0-idx); all methods overfit within a few epochs at ≤10% (e.g. FR val CE to 2.0-2.6 at 1-3%).
Pretraining (P0) never overtakes SC; gap shrinks monotonically with labels — opposite of the usual SSL expectation. [F]
Pre-scan checks at 3% (seed 42): L_lr best val CE (lr 3e-4/6e-4/1.2e-3): SC 0.7280/0.7267/**0.7188**, FT←P0 **0.7649**/0.7743/0.7717,
FR←P0 0.7854/0.7851/**0.7801** (lr moves CE ≤0.01 vs method gaps 0.05-0.06). L_reg (none / wd 0.05 / dropout 0.1 aggregator-ff+head / both):
SC 0.7188/0.7156/0.7154/0.7173; FT 0.7649/0.7646/0.7628/0.7626; FR 0.7801/0.7798/0.7777/0.7781 → ≤0.003 effect. [F]
Earlier 10% single-seed side test (S3_lowlabel): SC 0.6952, FT←P0 (lr 2.2e-3) 0.7346, FT←P0 (5.6e-4) 0.7265, FR←P0 0.7463. [F]
NOT done: P1 at reduced labels (the only winning pretraining).

### 5.5 Dropout study (dropout 0.2, 80-epoch cosine) [F] D_drop + `plots/dropout/test_metrics/per_run.csv`

| Case | no-drop val CE | drop val CE | drop best epochs | opt. c/light/τ rel. to SC (drop) | feasible (drop) |
|---|---|---|---|---|---|
| SC | 0.6240 | 0.6407 ± 0.0138 (one seed 0.6595; others 0.628-0.642) | 20,14,22,23 | 0.957 / 1.030 / 0.954 | 4/4 |
| SC+JC (clip 1) | 0.6241 | 0.7533 ± 0.0072 | 2,7,4,2 | 0.643 / 0.549 / 0.449 | 0/4 (grid edge) |
| FT←P0 | 0.6312 | **0.6105 ± 0.0004** (test CE 0.6100 [C]) | 38,31,26,35 | **1.248 / 0.986 / 1.036** (abs c 12.10 ± 0.10) | 4/4 |
| FR←P0 | 0.6500 | 0.6637 ± 0.0034 | 78,75,77,74 (hit the 80-ep limit) | 0.470 / 0.817 / 0.679 | 0/4 |
| FT←P1+JC | 0.6288 | 0.7400 ± 0.0300 | 12,6,29,20 | 0.671 / 0.634 / 0.457 | 0/4 (edge) |
| FR←P1+JC | 0.6436 | 0.7854 ± 0.0149 | 36,5,8,29 | 0.466 / 0.281 / 0.227 | 0/4 |

FT←P0+drop nominal fractions: 1.149/1.086/1.196 of SC (abs 13.78 / 278.6 / 44.7). Confounds (no-drop vs drop): 80 vs 40-epoch
cosine; batch 28672→25600 (SC, FT←P0) and 14336→12800 (JC cells); lrs not re-tuned; FT←P0 backbone ratio only /1 and /10 ever tried;
FT←P0+drop best at ~62% of peak lr vs no-drop at ~95% [C line 15925, M]. rriva (2026-11-16) doubts these as "hyperparameter
artifacts" → the WSD redo (§7). Embedding detail [C 15616]: FT←P1+JC+drop track-embedding rank falls 126 → ~70, jet-feature 62 → 20.

### 5.6 C_gap control: why ~0.62 here vs ~0.60 in the λ-scan [F] `C_gap_20261028_235643`

| run | data | batch / lr / chunk | best val CE (own val) | best ep |
|---|---|---|---|---|
| sc_112022 | 112022 (51.16M distinct train jets) | 28672 / 1.1e-3 / 512 | 0.6036 | 8 |
| sc_bs2048 | 152051 (18.24M distinct) | 2048 / 3e-4 / 256 | 0.6221 | 4 |
| reference SC | 152051 | 28672 / 1.1e-3 / 512 | 0.6240 ± 0.0029 | 8-11 |

→ the gap is the labelled-dataset size (2.8× more distinct jets), not the batch size. Old λ-scan CE-only runs on 112022: 0.5998-0.606 [M].
Learning curve 1.86M 0.695 / 18.6M 0.624 / 52M ~0.600 fits one power law α ≈ 0.17 [M].

### 5.7 S3 learning-rate scan (seed 42, 40-ep cosine + ES, on seed-42 pretrainings) [F]

| Arm (batch) | lr → best val CE | chosen final lr |
|---|---|---|
| SC (28672) | 5.6e-4 0.6357; 1.1e-3 **0.6238**; 2.2e-3 diverged (1.287) | 1.1e-3 |
| SC+JC (14336, no clip) | 4e-4 0.6495; 8e-4 0.6224; 1.6e-3 **0.6209** (grid edge) | 8e-4 |
| FT←P0 bb = lr (28672) | 5.6e-4 0.6434; 1.1e-3 0.6444; 2.2e-3 diverged | – |
| FT←P0 bb = lr/10 (28672) | 5.6e-4 0.6298; 1.1e-3 0.6317; 2.2e-3 **0.6297** | 1.1e-3 (bb 1.1e-4) |
| FR←P0 (94208) | 1e-3 0.6501; 2e-3 0.6503; 4e-3 **0.6496** (flat) | 2e-3 |
| FT←P1+JC bb lr/10 (14336) | 4e-4 0.6332; 8e-4 0.6287; 1.6e-3 0.6287 | 8e-4 (bb 8e-5) |
| FR←P1+JC (53760) | 7.5e-4 0.6437; 1.5e-3 **0.6407**; 3e-3 diverged (1.29) | 1.5e-3 |

LR prior lr(B) = 3e-4·√(B/2048). Final LRs approved by rriva 2026-10-28 [M]. Side test S3_epochs (15-ep cosine): SC 0.6330 (worse than
40-ep 0.6238), FT←P0 0.6305 (= 0.6297) → keep 40 epochs [F]. P1 CE-only arms reuse the P0 recipes (not tuned for P1).

### 5.8 S1 architecture pilot (SC CE, B 24576, lr 1e-3, 100-ep cosine + ES, no QK-norm, 2 seeds) [F]

per_token c128 0.6204 ± 0.0014; per_token c512 0.6203 ± 0.0004; concat c128 0.6211 ± 0.0007; concat c512 0.6201 / 0.6318 (slow seed)
→ no quality difference; concat ~1.45× faster [M] → concat, chunk 512 chosen (rriva 2026-10-19).

### 5.9 S2 pretraining-recipe pilot (T × λ, 10-ep pretraining, B 12288, lr 1e-3, no QK-norm; judged by a frozen-encoder probe:
FR, 15 epochs, B 32768, lr 1e-3, best val CE; floor = frozen random encoders 0.6563) [F]

| Recipe | T0.1 λ0.1 | T0.1 λ0.4 | T0.3 λ0.1 | T0.3 λ0.4 | clipping |
|---|---|---|---|---|---|
| P0 | 0.6412 | **0.6353** | 0.7213 | 0.6596 | none |
| P1 | 0.6490 | **0.6318** | 0.6715 | 0.6467 | 1.0 |
| P2 (unclipped) | 0.8267 | 1.1735 | collapsed (loss = ln 511 = 6.236, agg rank 1.3), stopped | same, stopped | none |
| P2c (clipped) | 0.6511 | **0.6475** | 1.0340 (tracks collapsed) | 0.6760 | 1.0 |

Winner everywhere T 0.1, λ 0.4 (rriva confirmed P0 2026-10-21, P1 2026-10-21, P2 2026-10-22). QK-norm pilot (S2_qk, same recipes,
clip 1.0): probes P0 0.6345, P1 0.6319, P2 0.6517, random 0.6563; sc_qk 0.6211 (= S1) [F]. P2 beats random encoders by only 0.005 → P2
"outcome": weakest pretraining, jet-feature embedding collapses (rank 1-2), no downstream arm. Probes of P0 peak at ep 4-6 and overfit,
P1 probes peak late (ep 9-14) [F].

### 5.10 QK-norm root cause (pretraining instability) [C line 7194, M, F]

- First full 30-epoch pretrainings (no QK-norm, lr 1e-3, clip 1.0; stage P_full_20261021_*): P0 s42 collapsed (val loss 5.81 → 11.14,
  track eff. rank 2.8, max pre-clip grad-norm window 7.4e5), P0 s424 broke (val 5.74 → 7.45, grad max 1.9e8), P1 s42 spiked (6.00 → 6.16,
  grad max 1.0e6); P1 s424 OOM-failed at epoch 7 [F metrics/run_info]. All 4 stopped 2026-10-22.
- Diagnosis (`slurm/smoke/attn_logits.py`, spike_repro): attention-logit growth in the TRACK encoder's row (intersample) attention,
  blocks 0-1: max scaled logit q·k/√d 1e5-1e7, normalised entropy → 0 (one-hot over the 512-jet chunk), ‖W_qkv‖ 24 (ep1) → 95 (ep4) → 197
  (P1 ep7) → 324 (S2 P0 winner ep9); bf16 rounding (~1e4-1e5 on such logits) flips the argmax → gradient spikes; S1 from-scratch models
  healthy (logit ~20, ‖W_qkv‖ ~20) → pretraining-specific; LR only slows it. [C 7194]
- Fix: RMSNorm on q and k per head (learnable gain, fp32 stats; memory-lean custom autograd) for every arm; RowAttentionMonitor logs
  max logit/entropy. After fix: 30-epoch P0/P1/P2 stable (max grad window 30-67). [F]
- Caveat [F]: with the learnable gain, the track row-attn max logit still grows in the QK-norm runs: final P0 s42
  (P_full_20261023_085735, QK on) per-epoch max over blocks 8, 14, 18, 22, 25, 36, 58, 180, 348, 646, 893 at epochs 0-10 and 3568 at
  epoch 29 (s424: 5952; P1: 2885 / 2696), with min block entropy 0.69-0.72 (no one-hot collapse, no instability); P2 21.7; downstream
  SC 7.4. With dropout 0.1 (P_drop P0) it stays at 24 by epoch 10.
- No figure on disk for the no-QK logit growth (old runs predate the monitor); would need re-running attn_logits.py on the old checkpoints.

Other engineering facts usable in a methods/appendix section [M]: flash varlen backward overflows its 32-bit-indexed dq buffer beyond
~62k aggregator sequences (silent NaN) — fixed by slicing; fused SDPA grid-dim limit 65535 — fixed by slicing; Lightning 2.6.1
`save_last` bug (last.ckpt frozen at the last improving epoch) — fixed with a separate monitor-less checkpoint; cold-NFS data loading
~8× slower → page-cache warm-up.

## 6. Compute (from run_info elapsed) [F]
Pretraining 30 ep: P0 43.4 h (A40), P1 34.2 h (L40S), P2 42.7 h (A40), per seed. Downstream finals (no dropout): SC 4.8-5.4 h, SC+JC
10-17 h, FT←P0 3.9-6.1 h (ES), FR←P0 2.1-3.4 h, FT←P1 13.5-15.8 h, FR←P1 5.5-7.5 h, FT←P1+JC 17.7 h, FR←P1+JC 7.9-11.5 h. Dropout runs 6-42 h.
Planned "cost per arm" table (docs/ssl_campaign.md §5.7) never produced.

## 7. Still running / planned (as of 2026-11-17)

1. **P_drop** (stage `P_drop_20261116_132628`, launched 2026-11-16 13:26, 3 two-GPU packs 7120624-26): P0, P1, P2 pretrainings with dropout
   0.1 (encoder rowcol attn+ff, aggregator attn+ff; inert aggregator in P0), 30 epochs, lr 1e-3, clip 1.0, QK-norm; batch P0/P2 12288,
   P1 10752 (memory). Status from metrics [F]: p0_d01 s42 and s424 at epoch 10, p1_d01 s42/s424 at epoch 7, p2_d01 s42 at epoch 8,
   p2_d01 s424 collapsed from epoch 0 (val 4.37-4.9, track uniformity 0, rank 4.8; grad windows up to 6352) and was stopped
   (rriva 2026-11-17), replaced by seed 4242 (just started). Est. 61 / 86 / 66 h per run (P0/P1/P2) [M]. Downstream from these
   checkpoints: not yet planned.
   **New observation (not in the memory note):** p0_d01 seed 42 shows a developing instability: a 172 pre-clip grad-norm window at step
   29,799 (~epoch 9), then windows of 30-75 every 100 steps, train loss 5.69 → 5.91-6.00, val loss 5.585 (ep 8) → 5.616 → 5.738 (ep 10),
   while seed 424 keeps improving (5.559 at ep 10, max grad 11.7). Track row-attn max logit stays 24 (entropy 0.68) → not the QK mechanism.
2. **WSD redo (W_lr.cells, generated, NOT launched)** — rriva's plan 2026-11-16 [C 15934-15994, M]: redo all 16 downstream cells
   (8 methods × {no dropout, dropout 0.2}: SC, SC+JC (clip 1.0), FT←P0, FR←P0, FT←P1, FR←P1, FT←P1+JC, FR←P1+JC — including the two
   P1 CE-only dropout cells never run) with ONE batch size 12800 for every run ("the loss depends on it"), WSD schedule (warmup 1.6 epochs
   from 1% of peak, stable, linear decay to 0 over the last 20%), NO early stopping (best-val-CE checkpoint), from the existing no-dropout
   P0/P1 checkpoints. LR search: lr ∈ {3.75e-4, 7.5e-4, 1.5e-3} for all cells (all old tuned lrs map to ~7.3-7.6e-4 at B 12800 by √B);
   FT cells with backbone lr/10 and lr/30 at all three → 66 runs, seed 42, 40-epoch WSD (decay epochs 32-40). Finals: 80-epoch WSD
   (decay 64-80), 4 runs per cell = 64 runs. Measured at B 12800 on L40S [F W_measure_sweep_7126988/results.csv]: peak GiB / s per step:
   SC+JC+drop & FT+JC+drop 37.5 / 1.42-1.48; SC+JC, FT←P1+JC 35.2 / 1.04-1.12; SC/FT CE (+drop) 18.3-19.0 / 0.51-0.73; FR+JC 9.6-9.7 /
   0.42-0.60; FR 5.0-5.6 / 0.21-0.36; 1452 steps/epoch. Multi-training-per-card packing (SLOTS_PER_GPU) implemented (b3e2c2f), share
   slowdown job 7153578 pending; rriva wants to see the cell list before launch. Search ~845 card-h [M].
   **What it replaces and why:** all current no-dropout and dropout finals (Tables C, 5.1, 5.2, 5.5) — because (a) the batch differs across
   arms (28672/14336/94208/53760, and 25600/12800 in the dropout runs) and the contrastive/CE optimisation depends on it; (b) the dropout vs
   no-dropout comparison is confounded by 80 vs 40 epochs, batch changes and untuned lrs; (c) the no-dropout runs (SC, FT←P0, FR←P0) peak at
   epochs 4-11 with lr still at 84-95% of peak and are then early-stopped, while FT←P1 runs were still improving when the 40-epoch cosine
   ended (best at ep 34-39) → schedule-limited comparison; (d) FT backbone ratio only /1 and /10 tried (/30 untested); (e) P1 CE-only arms
   reuse P0-tuned lrs. With WSD+no-ES every arm sees the same lr shape and budget.
3. Not planned yet: P1 at reduced labels; downstream from P_drop checkpoints; P2 downstream; robustness (data/MC) tests promised in the
   poster abstract.

## 8. Plots

Paper-ready candidates (wording in YAML `texts:`, no "ATLAS", consistent style; I viewed the first three):
- `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/arm_comparison/summary/best_val_ce_per_method.png` — best val CE, 8 methods, 3 groups (viewed: clean).
- `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/arm_comparison/test_metrics/test_rejections_optimal_wp77.png` (+ `_gn2`, `nominal_wp77`, `optimal_wp70`, `optimal_wp85`) — test c/light/τ rejections, 8 methods (viewed: clean, footnote correct).
- `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/label_scan/learning_curve/best_val_ce_vs_labelled_jets.png` — label scan (viewed: clean; P0 only).
- `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/pretrain_curves/final/p{0,1}_s{42,424}_{loss_terms,loss_and_effective_rank,loss_and_uniformity}.png` — pretraining curves (12 PNG).
- `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/p0_vs_p1/summary/best_val_ce_p0_vs_p1.png`, `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/p0_vs_p1/test_rejections/relative_optimal_wp77.png`.
- `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/arm_comparison/epochs_scjc/ce_train_val.png` (+ `effective_rank.png`, `uniformity.png`) — SC+JC collapse illustration.
- `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/arm_comparison/epochs/ce_train_val_zoom.png`, `.../epochs_p1/ce_train_val_zoom.png` — per-epoch CE (overfitting / FT←P1 still improving).
Provisional / not paper-ready:
- Everything under `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/dropout/` — confounded, to be superseded by the WSD redo.
- All of the above method-comparison plots will be superseded if the WSD redo becomes the paper's result.
- ROC curves `plots/arm_comparison{,_p1}/roc_test_{nominal,optimal}{,_noref}/roc.png` and rej_vs_pt — one seed per method (choice of seed
  noted in report.yaml); for a paper use multi-seed bands or the summary plots.
- `/storage3/DSIP/rriva/research/fm4tag-chunk/plots/pretrain_curves/*.png` (top level, 11-06; QK pilots, P2) — superseded by `final/`.
- `plots/label_scan/{lr_check,reg_check}/*` — single-seed checks, appendix at most; label-scan `ce_per_epoch.png` y-range to 2.6.
- Missing figures: S2 T×λ probe grid (only a table / the old deck `docs/slides/ssl_campaign/ssl_campaign.pdf`), QK-norm logit growth,
  embedding-quality comparison across downstream methods as a single summary.
Slide decks with all plots: `/storage3/DSIP/rriva/research/fm4tag-chunk/docs/slides/report/ssl_campaign_plots.pptx` (78 slides, has the
"eight training methods" settings table), `/storage3/DSIP/rriva/research/fm4tag-chunk/docs/slides/ssl_campaign/ssl_campaign.pptx` (older).

## 9. Inconsistencies / referee targets

1. Memory note (11-16) says "all JC+drop cases 0.64-0.67 c" relative to SC; files: SC+JC+drop 0.643, FT←P1+JC+drop 0.671, but FR←P1+JC+drop 0.466.
2. Memory note (11-17) says P_drop runs "all fine except p2 s424"; files show p0_d01 seed 42 destabilising since step ~29.8k (see §7.1).
3. Paper plan `/storage3/DSIP/rriva/research/fm4tag/paper/plan_v0.tex` lists "Pretraining (L_JE-CL) + frozen / fine-tuning" — JC-only
   pretraining is P2, the weakest recipe (probe barely above random) with no downstream runs; what was actually run downstream is P0
   (feature-embedding CL, the plan's "\discussion" item) and P1 (both).
4. Cross-pillar tension: the λ-scan pillar reports JC helping (other data 112022, batch 2048, ISAB-era/per-token intersample attention,
   no chunking/QK-norm, long training with JC runs peaking at ep 60-99), whereas here SC+JC = SC (0.6241 vs 0.6240; c-rej 9.79 vs 9.69).
   Data size alone explains 0.60 vs 0.62 (C_gap), but the JC verdicts differ; the paper must state the setups differ.
5. Unequal recipes across arms: batch 28672/14336/94208/53760; FT←P0 final lr (1.1e-3, bb 1.1e-4) was the worst of three nearly equal S3
   points (0.6317 vs 0.6297/0.6298); SC+JC final 8e-4 while S3 best was 1.6e-3 at the grid edge; P1 CE-only arms use P0-tuned recipes.
6. Schedule limitation: SC/FT←P0/FR←P0 select a checkpoint at lr ~84-95% of peak (ES), FT←P1 (CE and +JC) are cut off at 40 epochs while
   still improving. FT←P0 trains the backbone for only ~2-3 epochs before its best epoch (2-epoch hold, best at ep 4-5).
7. FR←P0 optimised rejections are from infeasible searches (f_c = 0 in 4/4 runs): its "optimised" c-rej 4.66 is a closest-approach point.
8. Optimised fractions are chosen at 80% b-eff on val but quoted at 77% on test; the constraint "≥ GN2v01" holds on val at 80%, not
   necessarily on test at 77% (e.g. SC τ 65.1 < 67.6).
9. Pretraining data is label-resampled (fixed flavour composition 26/26/39/10%) → "unlabelled pretraining" is not realistic data.
10. The S2 recipe grid ran without QK-norm (P0, P2 also without clipping); finals use QK-norm (pilot probes agree within 0.001 for P0/P1).
11. Label scan uses a single pretraining seed (P0 s42) below 100%; P1 never scanned.
12. P0's logged "aggregator" rank/uniformity (174, −3.88) come from an untrained random aggregator.
13. QK-norm has a learnable gain: logits still reach 2.7k-6k in final P0/P1 (no instability, but "bounded logits" is not literally true).
14. Seed 42 used for every LR choice; finals use other seeds (fine), but L_lr/L_reg/S2/S3 are single-seed.
15. GN2v01 scores: provenance of its training sample vs these Open-Data events unknown [I]; it is not a same-data baseline.
16. FR←P1+JC+drop seed 42424 (p1 s424) formally meets the collapse rule (best 0.7906 at ep 5, then 0.854).
