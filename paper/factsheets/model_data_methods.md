# Fact sheet: model, data, losses, augmentations, metrics, evaluation (as implemented)

Area key: model_data_methods. Compiled 2026-11-17. READ-ONLY inspection of
`/storage3/DSIP/rriva/research/fm4tag` (main, the lambda-scan model) and
`/storage3/DSIP/rriva/research/fm4tag-chunk` (branch feat/chunked-ia, the SSL-campaign model).
Exact parameter counts were recomputed on CPU from each run's resolved `.hydra/config.yaml` with
`scratchpad/model_data_methods/count_params.py` (uses the repos' own `build_encoders/aggregator/head`).

Provenance tags: [V] verified-in-files, [M] from-memory-note, [I] inferred.

---------------------------------------------------------------------------------------------------

## 0. Two model variants to name in the paper

| | **Model A ("lambda-scan model")** | **Model B ("SSL-campaign model")** |
|---|---|---|
| repo / config | fm4tag main, `encoders=rowcol_v0` (git cbeb5ba..b908189 for the runs) | fm4tag-chunk, `campaign_ssl.yaml` -> `encoders=rowcol_concat_v0`, `chunk_size=512`, `row_qk_norm=true` |
| row (intersample) attention | ISAB (16 learned inducing points per feature token), `row_mode: per_token` | full softmax attention inside chunks of 512 consecutive jets, `row_mode: concat` (SAINT style), QK RMS-norm |
| context of the row attention | whole rank-local batch (1024 or 2048 jets; tracks: all valid tracks of those jets) | the 512-jet chunk (jets: 512; tracks: ~3.7k valid tracks of those jets) |
| trainable params (no reconstructors) | **2,221,992** | **7,470,984** |
| data | data_parallel_preprocess_251112_112022 | data_parallel_preprocess_251112_152051 |
| GPUs | DDP 2 GPUs (regimes A, C) or 1 GPU (regime B) | 1 GPU per training |

[V] configs: `.../fm4tag/slurm/classify_param_scan/run_20260828_153214/scratch_bs_1024_lr_3e-4_ce_1.0_jc_0.3_lam_0.4_temp_0.1/seed_42/outputs/2026-08-28/22-35-02/.hydra/config.yaml`;
`.../fm4tag-chunk/slurm/ssl_campaign/F_final_20261028_235642/sc/seed_4242/outputs/2026-10-29/14-21-00/.hydra/config.yaml`.
Code of the lambda-scan era == main HEAD except lint + ES patience (git diff bbae539..HEAD -- src/) [V].

## 1. Inputs

- Jet (global) features, 2, continuous: `pt_btagJes`, `eta_btagJes` [V default.yaml `variables.jets.inputs`].
- Track features, 19 per track: 10 continuous `d0, z0SinTheta, dphi, deta, qOverP, lifetimeSignedD0Significance, lifetimeSignedZ0SinThetaSignificance, phiUncertainty, thetaUncertainty, qOverPUncertainty`; 9 categorical hit counts with embedding cardinalities `numberOfPixelHits 13, numberOfSCTHits 25, numberOfInnermostPixelLayerHits 4, numberOfNextToInnermostPixelLayerHits 4, numberOfInnermostPixelLayerSharedHits 2, numberOfInnermostPixelLayerSplitHits 4, numberOfPixelSharedHits 2, numberOfPixelSplitHits 12, numberOfSCTSharedHits 4` (sum 70 embedding rows) [V].
  Observed on 152051 val (13,299,139 valid tracks): no categorical value reaches its cardinality (max PixelHits 11, SCTHits 20, PixelSplit 8) [V, computed].
- Same 2+19 inputs as the salt "GN2 only-JT" and GN2_open_data configs [V `/storage3/DSIP/rriva/packages/salt-pt/salt/logs/GN2_only_jt_20260817-T134649/config.yaml`, `salt-ririva/salt/configs/GN2/GN2_open_data.yaml`].
- Normalisation: continuous jet and track features z-scored with mean/std of the TRAIN split's `norm_dict.yaml` (also for the pretrain split); categorical values used as raw integer indices (no normalisation) [V `datasets/datasets.py` `_build_norm_arrays`, config `norm_dict_path: ${base_path}/train/output/norm_dict.yaml`].
- Tracks: padded grid of 40 slots per jet + `valid` mask. 152051 val: mean 7.15 valid tracks/jet (median 7, max 32), 17.9 % of the grid valid (82.1 % padding), 0.063 % of jets with 0 tracks; per flavour b 8.09, c 7.24, light 7.32, tau 3.71 [V computed]. A jet with 0 tracks pools to a zero vector [V aggregator].
- Labels: `flavour_label` 0=b, 1=c, 2=light, 3=tau; verified equal to HadronConeExclTruthLabelID 5/4/0/15 on 152051 val [V computed]. Truth track labels `ftagTruthOriginLabel`, `ftagTruthVertexIndex` exist in the h5 but fm4tag does not use them [V].

## 2. Architecture (code: `models/backbones/encoder.py`, `transformers.py`, `attention.py`, `aggregator.py`, `heads/heads.py`, `blocks.py`)

Pipeline: jets -> GlobalTransformerEncoder -> projector z_g; tracks -> Encoder -> projector z_t (per track) -> TransformerAggregator (per-jet track transformer + MEAN pool) -> concat [z_g, pooled z_t] = z_jet -> MLP head -> 4 logits.

- Per-feature embedding, width dim=16 (both models):
  * continuous: per-feature 2-layer MLP 1 -> 32 (ReLU) -> 16, separate weights per feature (grouped Conv1d) [V encoder.py L238-248, L309-313];
  * categorical: one `nn.Embedding(70,16)` with per-feature offsets [V L227-231, L260, L304-305].
  * Token grid: jets (B, 2, 16); tracks (N_valid, 19, 16) where N_valid = all valid tracks of the batch packed by boolean indexing (jet identity discarded in the encoder) [V view_encoding.py].
- Encoder stack: one `RowColTransformer`, depth 6; each depth step = pre-norm column attention + FF, then pre-norm row attention + FF. Pre-norm is `x + f(LN(x))` (`PreNormResidual`; fixes SAINT's `f(LN(x)) + LN(x)`), final `norm_out` LayerNorm. FF = GEGLU (Linear(d, 2*mult*d) -> GEGLU -> Linear(mult*d, d)) [V blocks.py, transformers.py].
  * Column attention: within one object, across its feature tokens (2 for jets, 19 for tracks). Model A: 2 heads x 8, ff_mult 1. Model B: 1 head x 16, ff_mult 2 [V configs].
  * Row attention, Model A (`InducedRowAttention`, ISAB, per_token): for each feature token separately (projections shared over tokens), 16 inducing points per token (params (N,16,16)) read the whole rank-local set of objects, then every object reads the summary; 2 heads x 8; second cross-attention out-proj zero-initialised. Jets attend (via the inducing points) to the other jets of the rank's batch; each track to all valid tracks of all jets in the rank's batch [V attention.py L186-257, view_encoding.py].
  * Row attention, Model B (`RowAttention`, concat): token grid flattened to one row vector of width N*dim (tracks 19*16=304, jets 2*16=32); full softmax attention among the objects of the same 512-jet chunk (jets: the chunk's 512 jets; tracks: all valid tracks of the chunk's 512 jets, ~3.7k) via flash varlen; tracks 8 heads x 32 (inner 256), jets 1 head x 32; row FF GEGLU mult 2 at width 304 / 32; QK RMS-norm (`LeanRMSNorm`, learnable gain, per head dim 32) on q and k; `to_out` zero-initialised [V attention.py L375-486, view_encoding.py L36-80, rowcol_concat_v0.yaml].
- Projectors (simple_MLP Linear-ReLU-Linear): tracks 304 -> 228 -> 128 (both models); jets 32 -> 64 -> 32 (A) / 32 -> 64 -> 128 (B) [V].
- Aggregator `TransformerAggregator`: for each jet, a 6-layer pre-norm transformer over the jet's valid track projections (dim 128, 4 heads x 64 = 256 inner, GEGLU ff_mult 4, no final LN), then MEAN pool over valid tracks (not attention pooling); concat with the jet projection -> z_jet of dim 160 (A) / 256 (B). Jet kinematics never interact with tracks before the head. Model A runs it on the padded grid with key-padding mask + re-masking; Model B on packed tracks, one varlen group per jet, fp32 pooled mean [V aggregator.py both repos].
- Head `MultiStreamClassifierHead`: MLP z_jet -> 256 (ReLU, dropout 0.02) -> 4 [V].
- No auxiliary tasks (no track origin, no vertexing) [V]. Pretraining-only heads: per-feature reconstructors (sep_MLP), frozen during fine-tuning (14,130 params) [V].

### Exact parameter counts [V computed]
| module | Model A (lambda-scan) | Model B (SSL campaign) | S1 per_token (chunk, no QK) |
|---|---|---|---|
| jet encoder (excl. reconstructor 1,154) | 38,144 (stack 32,736: col 11,520 / row 21,216; projector 4,192) | 91,712 (stack 80,064: col 16,320 / row 63,744; projector 10,432) | 28,448 |
| track encoder (excl. reconstructors 12,976) | 164,772 (stack 58,848: col 11,520 / row 47,328; projector 98,852; embeddings+LN 7,072) | 5,335,620 (stack 5,229,696: col 16,320 / row 5,213,376; projector 98,852) | 128,964 |
| aggregator | 1,976,832 (89 % of model) | 1,976,832 (26 %) | 1,976,832 |
| head | 42,244 | 66,820 | 42,244 |
| total trainable | 2,221,992 | 7,470,984 | 2,176,488 |
| Lightning summary | "2.2 M trainable, 14.1 K non-trainable" (rank_0.err) | "7.5 M trainable" ; FR arms "2.0 M trainable, 5.4 M non-trainable" | — |
GN2 only-JT (salt): 2,236,005 trainable (`metadata.yaml`) — same size as Model A.

### fm4tag vs GN2
GN2 open-data (salt `GN2_open_data.yaml`): per-track init MLP on [19 track + 2 jet features] -> 256 (hidden 256), 4-layer transformer over the jet's tracks (embed 256, 8 heads, out 128), GlobalAttentionPooling, jet head 128-64-32 -> 4, aux track-origin (8 classes, weight 0.5) + vertexing (weight 1.5), OneCycle lr max 5e-4, batch 3000. "GN2 only-JT" = same minus the aux tasks (+1 register token, flash-varlen), class-weighted CE (1.5,1.5,1.0,4.0), fp16-mixed, 2 GPUs x 3000, 50 epochs max, best ckpt epoch 15, trained on 112022 train [V].
fm4tag HAS: per-track transformer over tracks of a jet (the aggregator; 6 x 128 vs GN2 4 x 256), pooled jet vector, MLP head. LACKS: jet kinematics concatenated to every track input, attention pooling (uses mean), auxiliary tasks, class-weighted CE. ADDS: per-feature tokenisation (categorical embeddings), column attention, intersample (row) attention across jets/tracks of other jets, per-object projectors, separate jet-feature encoder.

## 3. Augmentations (`augmentations/cutmix.py`, `mixup.py`, `base.py`; unchanged on the chunk branch)
- Two views: view 0 = Compose[CutMix(lam=λ), Mixup(lam=λ)], view 1 = Identity (clean). One λ (`lambda:` top-level key) drives both [V default.yaml].
- CutMix, stage RAW (normalised continuous values and categorical integer indices, after flattening valid tracks): one random permutation of the rows; each cell (row, feature) replaced by the same feature of the permuted row with probability λ (Bernoulli keep prob 1-λ, independent per cell, independent masks for categorical and continuous, shared permutation) [V L57-93]. Rows = jets of the whole batch (jet stream) or ALL valid tracks of the whole batch (track stream; a track may receive features of a track of another jet). Padded slots never take part (valid tracks only) [V view_encoding].
- Mixup, stage EMBEDDING: x' = (1-λ) x + λ x[perm], fixed λ (random_lam False), same permutation for categorical and continuous tokens [V L158-181]. Tracks: applied to the per-feature input embeddings before the transformer; jets: applied to the jet encoder OUTPUT tokens, after the transformer (asymmetry, documented in `encode_global_view`) [V].
- Augmentations act on the whole batch even in Model B (permutation not restricted to the 512-jet chunk) [V: augmentations untouched by the branch diff].
- In fine-tuning, CE is computed on the clean forward only; the augmented view enters only the JC term (identity view = the clean pass, reused). So λ is inert when JC weight = 0 [V finetune_module.py `_compute_loss`, `_encode_jet_views`].

## 4. Losses
- CE: `F.cross_entropy(logits, label, weight=class_weights)` with `use_class_weights: false` -> UNWEIGHTED in every lambda-scan and SSL run [V configs]. Logged `val/head/loss_ce`.
- Contrastive (all of them): `MultiViewSupConLoss` (Khosla et al. SupCon generalised to unlabelled views), temperature 0.1, `loss_type: out`, `include_pos_in_denom: true`, `include_same_view_negatives: false` [V losses/contrastive.py]. With 2 views (augmented, clean) each anchor has 1 positive (its other view) and the denominator holds only the other view's items (positive + all other samples) -> symmetric cross-view InfoNCE (CLIP-like, NT-Xent without same-view negatives), L2-normalised embeddings (cosine sim / T), averaged over both views as anchors [I from code].
  * Model A: pool gathered across DDP ranks with gradient (`all_gather_with_grad`): negatives = all jets of the global batch (2047 negatives in regimes A/B, 4095 in C).
  * Model B: within each 512-jet chunk, no gather, fp32 similarities (511 negative jets; tracks: all other valid tracks of the chunk) [V `_chunked`].
- JC ("jet contrastive", `aggregator/loss_contrastive`): on z_jet = aggregator output (160-d / 256-d), the head's own input; NO separate projection head [V].
- Per-object contrastive (pretraining only, key `contrastive`): on `jets_embedding` (jet projector output) and `tracks_embedding` (per-track projector output, 128-d); each object's loss multiplied by the same weight and summed [V pretrain_module.py].
- Denoising (implemented, pretraining only): categorical = sum over the 9 features of CE; continuous = MSE over the 10 features; reconstruct the original from the first (augmented) view; default weights 0.2/0.2. NOT used in any reported run (lambda scan = fine-tuning only; P0/P1/P2 set denoising 0) [V].
- Total loss: weighted sum with weights used AS-IS (no normalisation since a6183fe, 2026-07-20, ancestor of every scan commit) [V].
  * lambda scan: L = 1.0·CE + w_JC·JC, w_JC ∈ {0, 0.3, 1.0}; λ scan at w_JC = 0.3 [V sweep_info].
  * P0: L_tracks + L_jets; P1: L_tracks + L_jets + L_JC; P2: L_JC (all weights 1.0, T = 0.1, λ = 0.4) [V P_full configs].
  * SSL downstream: CE (+ 0.3·JC for JC arms, λ = 0.4, T = 0.1) [V].

## 5. Embedding-quality metrics (`metrics/uniformity.py`, `metrics/effective_rank.py`, `callbacks/embedding_metrics.py`; identical in both repos)
- Uniformity (Wang & Isola): z normalised to the unit sphere; if > 4096 rows a random subset of 4096; U = log mean_{i<j} exp(-t ||z_i - z_j||²) with t = 2 (`uniformity.py` L30-35). More negative = more uniform; high-dim limit ≈ -2t = -4 [I].
- Effective rank (Roy & Vetterli): centre z (subtract column mean, no normalisation), singular values s, p = s/Σs, erank = exp(-Σ p log p) (`effective_rank.py` L55-62); range [1, D] with D = 160 (A) / 256 (B) for the aggregator.
- Callback: on the VALIDATION split only, clean forward in eval mode, the first 8192 rows of each embedding (global cap over DDP ranks; i.e. the first val batch(es), same jets every epoch), embeddings `aggregator` (global jet embedding z_jet), `jets_embedding`, `tracks_embedding`; logged `val/<emb>/<metric>` [V].
- Alignment (mentioned in plan_v0) is NOT implemented anywhere [V grep].

## 6. Dataset
- Source: ATLAS Open Data JetSet (README: CERN Open Data record 93940, ~50M simulated ttbar events at 13.6 TeV, training-dataset-dumper h5; ≤40 tracks per jet associated by a pT-dependent ΔR cone; ≤5 truth hadrons) [V `/storage3/DSIP/rriva/datasets/transforming-jet-flavor/README.md`]. Files mc-flavtag-ttbar-{small,medium,large}.h5: 48,698,675 events, 199,752,328 jets (vds.h5) [V]. The files are disjoint (b-jets show 1.00 rows per distinct jet in dup check) [V/I].
- Kinematic selection (UPP region `lowpt`): 20 < pT(btagJes) < 250 GeV; |η| < 2.5 (max |η| in val 2.498) [V base.yaml]. Extrapolated from the small file (95.98 % pass): ~191.7M jets pass -> the poster's "~190M jets" [V computed + I]. Natural composition after selection (small file): light 48.8 %, b 38.6 %, c 8.9 %, tau 3.7 % [V].
- Preprocessing: umami-preprocessing (UPP, hash 4f2a0f6, writer v0.2.17) `--prep --resample --merge --norm`; resampling `countup` to the c-jet (pT, η) distribution, 50 pT bins in [20, 250] GeV x 40 η bins in [-2.5, 2.5]; per-split requested counts b:c:light:tau = 16:16:24:6 -> fractions 25.8 / 25.8 / 38.7 / 9.7 %; output shuffled, lzf-compressed [V h5 attrs]. Resampling duplicates: c never, b/light ≤1 %, tau 1.2-1.67 rows per distinct jet [V dup_check log].
- Splits by eventNumber (no event shared across splits):

| dataset | split | eventNumber rule | rows | distinct jets |
|---|---|---|---|---|
| 112022 (lambda scan) | train | %62 in 0-51 | 51,999,998 | 51,157,189 |
| | val | %62 in 52-56 | 4,999,997 | 4,841,993 |
| | test | %62 in 57-62 (=57-61) | 5,000,005 | 4,839,676 |
| 152051 (SSL) | pretrain | %100 in 0-63 | 39,680,000 | 39,020,475 |
| | train | %100 in 64-93 | 18,600,000 | 18,241,670 |
| | val | %100 in 94-96 | 1,860,000 | 1,752,248 |
| | test | %100 in 97-99 | 1,860,000 | 1,753,529 |

  Overlap: 0 within each dataset; 112022 train contains 43 % of the distinct jets of every 152051 split (train 7,906,299; val 752,628; test 753,759; pretrain 16,943,776) -> never evaluate 112022-trained models (incl. salt GN2 only-JT) on 152051 [V dup_check_6670528.log].
- Jet collection: not stated in the README (R=0.4 anti-kt per truth-jet fields; presumably PFlow) [I].

## 7. Evaluation
- D_b = ln[ p_b / (f_c p_c + f_τ p_τ + (1 - f_c - f_τ) p_light) ] (`roc_comparison.discriminant`); truth from HadronConeExclTruthLabelID; cuts 20 < pt < 250 GeV, |eta| < 2.5 on `pt/eta` (keeps 99.98 % of 152051 test) [V].
- Rejection of flavour X at working point WP = N_X / N_X(D_b ≥ cut), cut = (1-WP) quantile of the b-jet D_b [V `_eval_point`]. WPs 70, 77, 80, 85 %.
- Nominal fractions f_c = 0.2, f_τ = 0.01 [V].
- Constrained ("optimal") fractions: grid over (f_c, f_τ) with step 0.002; choose max c-rejection at WP_search s.t. light-rej ≥ floor·light-rej_ref and τ-rej ≥ floor·τ-rej_ref (reference at its nominal fractions, same WP); if infeasible, flag and take max of min(u/floor_u, τ/floor_τ) [V fraction_constrained_search.py].
  * SSL campaign (`arm_test_metrics.yaml`): per run, search on VAL at WP 0.80, floor 1.0, box f_c < 0.15, f_τ < 0.12, reference GN2v01 columns of the h5 (`GN2v01_pb/pc/pu/ptau`), then applied to TEST at all WPs; inference bf16, batch 28672 [V].
  * lambda scan (`fraction_constrained_lambda_scan_final.yaml`): `test_file: null` -> search on the 112022 TEST split, WP 0.80, reference salt GN2 only-JT (evaluated on 112022 test), inference fp32 batch 1024 [V].
- Checkpoint used: best epoch on `val/head/loss_ce` (from metrics.csv) [V].

## 8. Training setup
| | lambda scan (Model A) | SSL campaign (Model B) |
|---|---|---|
| optimiser | AdamW, wd 5.96e-5 | same |
| lr | 3e-4 (all cells) | per arm (S3): pretrain 1e-3; SC 1.1e-3; SC+JC 8e-4; FT←P0/P1 1.1e-3 (backbone 1.1e-4); FT+JC←P1 8e-4 (8e-5); FR←P0/P1 2e-3; FR+JC←P1 1.5e-3 |
| schedule | cosine over max_epochs (50 or 100), warmup_frac 0 (1 step) | cosine over 40 ep, 2 % linear warmup from 1 % lr; pretraining LinearLR + CosineAnnealingLR, 30 ep; WSD (warmup 1.6 ep, flat, linear decay last 20 %) implemented 2026-11-16 for the redo |
| early stopping | val/head/loss_ce, patience 15 | val/head/loss_ce, patience 12 (pretraining: val/loss, ran full 30) |
| batch | A: 1024/rank x 2 GPUs; B: 2048 x 1; C: 2048/rank x 2 | pretrain 12288; SC/FT 28672; +JC 14336; FR 94208; FR+JC 53760; redo 12800 all |
| clipping | none | pretraining 1.0; scjc_clip1 1.0; other downstream none |
| precision | bf16-mixed, TF32 matmul | bf16-mixed |
| distributed | DDP NCCL (P2P disabled), custom BatchSliceSampler | 1 GPU |
| sampler | 8 random contiguous spans per batch, offsets redrawn each epoch with replacement; val/test ordered | same |
| dropout | 0 (head 0.02); one jc=0 + dropout 0.1 baseline | 0 (head 0.02); D_drop 0.2 everywhere; P_drop 0.1 |
| seeds | 42, 424, 4242, ... | same series |
| hardware | L40S (+A40 for some) | L40S 45 GiB, A40 |

## 9. Diagrams found
- `/storage3/DSIP/rriva/research/diary/figures/gn2.png` (5431x2331, GN2 architecture; plan_v0 references it as images/gn2.png).
- `/storage3/DSIP/rriva/research/salt-ririva/paper/salt-arch.png` (upstream salt JOSS figure).
- `images/fm4tag.png` referenced in plan_v0 but not found anywhere (needs drawing).
- ASCII diagrams: fm4tag `README.md`, `docs/encoder_dataflow.md` (both repos).
