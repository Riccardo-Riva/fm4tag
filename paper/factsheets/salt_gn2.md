# Fact sheet: salt_gn2 (GN2 multi-task CE + jet-contrastive study in salt)

Compiled 2026-11-17. Read-only on every repo and run dir. Numbers marked **verified** were recomputed from
`metrics.csv` / prediction files / configs in this session; the scripts and raw outputs are in
`/tmp/claude-1138/-storage3-DSIP-rriva-research-fm4tag/653cf529-153e-437a-bcf0-9bb91ed33820/scratchpad/salt_gn2/`
(`summarize_scans.py` → `s1014.tsv`, `sold.tsv`, `smid.tsv`; `hpo_summary.py` → `hpo_all.txt`, `hpo_strong.txt`;
`test_rej.py` → `test_rej_jc03.csv`, `test_rej_jc1.csv`; `one_rej.py`).

Conventions used below
- "best val CE" = minimum over epochs of `val/jets_classification_loss` of one seed; cell value = mean ± std (sample std) over the
  **completed** seeds (`run_info.yaml status: completed`), unless stated. Epochs are the 0-based `epoch` column of metrics.csv.
- weighted CE = class-weighted with the preprocessing class_dict, `flavour_label` weights [b, c, light, tau] = [1.5, 1.5, 1.0, 4.0]
  (`/storage3/DSIP/rriva/datasets/transforming-jet-flavor/data_parallel_preprocess_251112_112022/train/output/class_dict.yaml`,
  identical for the 152051 train slice). Unweighted = plain CE (as fm4tag).
- Rejections: ftag `get_discriminant` with f_c = 0.2, f_tau = 0.01 (nominal GN2 fractions), jets with 20 < pT < 250 GeV, |eta| < 2.5,
  ftag `calculate_rejection` at fixed b-efficiency (70/77/80/85%). "val rej@best" = `val/{c,u,tau}jets_rej_70` logged by the
  RejectionMetrics callback at the best-val-CE epoch. Test rejections: one checkpoint (best val CE) of the best seed of a cell.
- Divergence flag (my definition, applied uniformly): epoch-mean train CE rises > 0.01 nats above its running minimum.
- I_NCE = log K − L_NCE (nats), K = global batch, logged as `*_jets_contrastive_ince` (salt/models/ssl.py:160).

## 1. Data
- All λ scans and HPO studies: `data_parallel_preprocess_251112_112022` — train 51,999,998 jets, val 4,999,997, test 5,000,005
  (4,998,732 after the pT/eta cuts) (**verified**, h5 shapes). Resampled ("countup") to the c-jet (pT, eta) spectrum, so val/test
  are resampled too and train contains duplicates (from `salt-ririva/future_developments.md` §5).
- 11-06 GN2 run: `data_parallel_preprocess_251112_152051` — pretrain 39,680,000 / train 18,600,000 / val 1,860,000 / test 1,860,000
  (**verified**). Same UPP config, different split; this is the split of the fm4tag SSL campaign. The 112022 train overlaps
  152051 (fm4tag memory `fm4tag-ssl-campaign.md`): models trained on 112022 must not be evaluated on 152051 test.

## 2. The GN2 model and recipe (salt `salt/configs/ssl/GN2.yaml`, git 6e43f64)
- Inputs: 19 track variables (d0, z0SinTheta, dphi, deta, qOverP, IP significances, uncertainties, hit counts) + jet pT, eta
  (`pt_btagJes`, `eta_btagJes`), up to 40 tracks/jet.
- init net: per-track dense 256 (hidden [256], ReLU), 71.4K params; encoder `salt.models.Transformer` 4 layers, embed 256,
  8 heads, out_dim 128, LayerNorm + final norm, flash-attn varlen kernel, 2.1M params; `GlobalAttentionPooling` (129 params);
  classification head dense [128, 64, 32] → 4 classes (b, c, light, tau), 27.0K params. **Total 2.2M trainable params**
  (Lightning ModelSummary, `rank_0.err` of the 10-14 seeds). The SSL projector adds 65.9K (dense hidden [256] → 128), dropped at
  inference. **No auxiliary tasks** (no track-origin / vertexing heads) — "GN2-like", not the full ATLAS GN2.
- Optimiser AdamW, weight decay 1e-5; salt OneCycle stepped per batch: initial 1e-7 → max LR (warm-up `pct_start`) → cosine to
  1e-5 over `max_epochs`; bf16-mixed; no gradient clipping; dropout/drop-path/label smoothing 0; `reads_per_batch: 8`
  (each batch stitched from 8 random contiguous runs, redrawn every epoch); ES and checkpointing on `val/jets_classification_loss`.
- Recipes actually used (from each run's `sweep_info.txt` + saved `config.yaml`, **verified**):

| Run | CE | LR peak | warm-up | GPUs × batch (K) | epochs | ES patience | attention |
|---|---|---|---|---|---|---|---|
| 09-28 jc0.3 scan | unweighted | 3e-4 | 1% | 2 × 2048 (4096) | 100 | 12 | torch-meff |
| 10-01 jc1 scan | weighted | 5e-4 | 1% | 2 × 3000 (6000) | 100 | 15 | flash-varlen |
| HPO trials | weighted | searched | 1% (5% strong_cl) | 1 × 6000 (6000) | 40 | 15 | flash-varlen |
| 10-13 scan | weighted | 1.9e-3 | 5% | 1 × 3000 (3000) | 100 | 15 | flash-varlen |
| 10-14 scan, 11-06 GN2 | weighted | 1e-3 | 5% | 2 × 3000 (6000) | 100 | 20 | flash-varlen |

## 3. The contrastive (JC) implementation in salt (code read: salt/augmentations/saint.py, salt/models/ssl.py)
- Exactly two views: a = Identity (also the supervised view), b = `SaintCorruption`: CutMix on the normalised inputs (each feature
  of each valid track — and of the jet stream — replaced with probability λ by the same feature of a random other valid row of the
  whole batch) then Mixup on the initial embeddings (x ← (1−λ)·x + λ·x_partner, partner a random other valid row). The launcher
  sets `cutmix_replace_prob = mixup_other_weight = λ` (one knob). λ = 0 → identical views; λ = 1 → view b is another jet.
- Views are fused into one forward pass (exact, since salt encoders never mix jets). Validation predictions use clean inputs.
- `ContrastiveTerm` on `global_rep` (128-d pooled jet embedding): shared projector Dense(128 → 256 → 128), L2 normalisation after
  projection (cosine similarity), `ContrastiveLoss` = symmetric InfoNCE (SAINT/CLIP style), positives = the same jet in the other
  view, negatives = the other jets **of the other view only**, gathered with gradients across DDP ranks → K = global batch
  (6000 on 2 × 3000). Similarities in fp32. Default τ = 0.1 (scanned 0.05–1 in HPO; 0.5 in the 10-13/10-14 scans).
- Loss combination `loss_mode`: `wsum` (salt default): L = CE + w·L_NCE (w = `jet_contrastive` = jc); `GLS`: geometric mean
  √(CE·L_NCE) implemented with detached weights w_i = L/(N·L_i) (salt/modelwrapper.py ~l.329), so the CE:CL gradient ratio is
  L_NCE/CE (≈10 at τ ≳ 0.4) — high-τ GLS trials are effectively CE-only (**inferred** from the code).
- Embedding-quality monitors (`salt.callbacks.EmbeddingMetrics`): effective rank (exp of the entropy of the normalised singular
  values of the centred pooled embedding, 8192 val jets) and uniformity (Wang & Isola, t = 2, ≤ 4096 jets) of the pooled embedding.

## 4. λ scans (from scratch, CE + jc·JC)

### 4.1 run_20260928_195621 — jc 0.3, τ 0.1, **unweighted CE**, LR 3e-4, 2 × 2048, ES 12 — COMPLETE, 41 seeds, 0 divergences
**verified** (matches memory note and `plots/best_vs_lambda_summary.csv`):

| cell | n | best val CE (unweighted) mean ± std | min | mean best epoch | val I_NCE at best (log K = 8.32) | test rej @70% best seed c/light/tau | @77% c/light/tau |
|---|---|---|---|---|---|---|---|
| CE only | 8 | 0.60075 ± 0.00029 | 0.60024 | 15.6 | – | 36.6 / 1172 / 192 (s4242424) | 14.45 / 362 / 48.9 |
| λ 0.2 | 8 | 0.60065 ± 0.00021 | 0.60047 | 30.6 | 7.90 | 36.9 / 1007 / 188 (s424242) | 14.57 / 333 / 50.4 |
| λ 0.3 | 6 | 0.60074 ± 0.00026 | 0.60049 | 48.7 | 7.67 | 36.8 / 993 / 167 (s424) | 14.56 / 330 / 44.8 |
| λ 0.4 | 5 | 0.60067 ± 0.00045 | 0.60021 | 73.2 | 7.26 | 37.0 / 1054 / 198 (s42424) | 14.50 / 334 / 51.2 |
| λ 0.5 | 5 | 0.60178 ± 0.00105 | 0.60081 | 76.2 | 6.50 | 36.6 / 1020 / 190 (s424) | 14.49 / 338 / 48.9 |
| λ 0.6 | 5 | 0.60113 ± 0.00107 | 0.59998 | 77.6 | 5.89 | 37.0 / 1035 / 179 (s424242) | 14.58 / 337 / 47.1 |
| λ 0.7 | 4 | 0.60109 ± 0.00084 | 0.60049 | 82.5 | 5.47 | 36.6 / 1049 / 190 (s42424) | 14.45 / 335 / 49.3 |

- Divergence: 0/41 (max epoch-mean train-CE rise 0.0075). But a few high-λ seeds have isolated **val-CE excursions** (up to
  ~+0.01) that trigger ES at a worse optimum: λ0.5 s42424 (0.60335, stopped ep 51), λ0.6 s424 (0.60279, ep 62), λ0.7 s424 (0.60232,
  ep 69) — these drive the large std at λ ≥ 0.5.
- Budget-limited cells: 3/5 (λ0.5), 2/5 (λ0.6), 3/4 (λ0.7) seeds ran to epoch 99 (still improving). CE-only stops at ep ~16–20,
  where the 100-epoch cosine LR is still ~95% of peak (**inferred** from the schedule): CE-only never anneals, high-λ cells do.
- I_NCE falls monotonically with λ: a clean knob on view information (InfoMin readout).
- Test-set GN2v01 (ATLAS production tagger, in the same file) @70: 42.5 / 1475 / 283; @77: 15.64 / 451 / 72.6 (**verified**).

### 4.2 run_20261001_091525 — jc 1.0, τ 0.1, weighted CE, LR 5e-4, 2 × 3000 — COMPLETE, 24 seeds, CL cells diverge
**verified**:

| cell | n | best val CE (weighted) | min | best epochs | diverged (my flag) | val rej@best c/u/tau (mean) | val I_NCE | eff. rank / uniformity @best | test rej @70 best seed |
|---|---|---|---|---|---|---|---|---|---|
| CE only | 8 | 0.60038 ± 0.00037 | 0.59978 | 8–16 | 0/8 | 36.3 / 876 / 207 | – | 27.0 / −0.69 | 36.3 / 907 / 225 (s4242424) |
| λ 0.3 | 8 | 0.60792 ± 0.00372 | 0.60228 | 13–71 | 6/8 | 31.1 / 799 / 185 | 7.90 | 51.4 / −2.38 | 34.5 / 831 / 217 (s42424, only stable seed) |
| λ 0.5 | 8 | 0.61672 ± 0.00223 | 0.61444 | 18–33 | 8/8 | 27.7 / 727 / 158 | 6.55 | 43.6 / −1.83 | 28.6 / 736 / 161 (s4242424) |

- @77% test: CE only 14.57 / 294 / 64.0; λ0.3 14.00 / 276 / 62.7; λ0.5 12.26 / 250 / 53.3 (**verified**).
- CE-only rejections keep improving after its best-CE epoch: val max-over-epochs vs at-best-CE: c 37.25 vs 36.30, light 1029 vs 876
  (−15%), tau 246 vs 207 (−16%) (`plots/best_vs_lambda_summary.csv`, **verified**).

### 4.3 Clipping test (seed 424, λ 0.5, jc 1.0 recipe) — **verified**
- Unclipped control run_20261006_174245: best 0.62045 @18, diverges at ep 20 — bit-identical to the original seed (deterministic).
- `gradient_clip_val=1.0` run_20261006_174240: best 0.61307 @32, divergence at ep 36 (my flag; memory says "epoch 37"), val rej@best
  29.3 / 776 / 193. Clipping only delays the divergence. From memory: pre-clip norm ~5 at every step (clip always active);
  attention-pooling grad grows ×50 (0.19 → 10.9) in the epoch before the spike, then init-net/encoder grads reach ~1e5.

### 4.4 run_20261009_092914 — GLS, jc 1, τ 0.4, LR 1.6e-3, 1 × 3000 — CANCELLED after ~6 h (10-09 ~15:50)
Only 3 of 7 cells started, each 1 truncated seed: CE-only 0.60022 @16 (20 epochs), λ0 0.60961 @5 (12 epochs, flagged), λ0.1
0.60265 @8 (10 epochs). Not usable. Cancelled because the author stopped trusting GLS (memory).

### 4.5 run_20261013_124341 — wsum, w 0.2, τ 0.5, LR 1.9e-3, 1 × 3000, warm-up 5%, ES 15 — STOPPED 10-14 (**verified**)
11 completed seeds, 8 diverged (CE-only 1/2, λ0 1/1, λ0.3 2/2, λ0.4 2/2, λ0.5 2/2; λ0.1 and λ0.2 1/1 survived). Survivors:
CE-only 0.60105, λ0.1 0.60021, λ0.2 0.60095. Recipe too aggressive (half the batch at LR 1.9e-3).

### 4.6 run_20261014_153400 — wsum, w 0.2, τ 0.5, LR 1e-3, 2 × 3000, 5% warm-up, ES 20 — **PARTIAL: all jobs cancelled 2026-10-15 12:36:51** (~21 h after launch, together with both wsum_strong_cl HPO jobs). Memory note has no results; **computed here**:

| cell | completed seeds | best val CE (weighted) mean ± std | per seed | best epochs | eff. rank / uniformity @best | val I_NCE | val rej@best c/u/tau |
|---|---|---|---|---|---|---|---|
| CE only | 6 | 0.59951 ± 0.00035 | .59955 .59972 .59907 .60005 .59940 .59926 | 11–18 | 22.6 / −0.51 | – | 36.3 / 914 / 221 |
| λ 0 (identical views) | 3 | 0.59956 ± 0.00068 | .60016 .59970 .59882 | 14–18 | 44.3 / −1.28 | 1.98 | 36.3 / 926 / 225 |
| λ 0.1 | 2 | 0.59985 ± 0.00028 | .60005 .59966 | 11, 19 | 13.4 / −1.56 | 1.95 | 36.6 / 895 / 215 |
| λ 0.2 | 1 (+1 truncated) | 0.59997 | .59997 (s424 truncated at ep 32: .59914 @15) | 18 | 14.9 / −1.63 | 1.92 | 36.4 / 937 / 256 |
| λ 0.3 | 1 (+1 truncated at ep 10) | 0.59992 | .59992 | 15 | 16.4 / −0.86 | 1.85 | 36.9 / 830 / 222 |
| λ 0.4, 0.5 | 0 | never started | | | | | |

- Divergence: 0 before the best epoch; one mild post-best train-CE bump (λ0.3 s42, +0.012 at ep 30, no effect on its best).
- Unlike τ 0.1, τ 0.5 + w 0.2 does **not** delay convergence (best epochs 11–19 like CE-only): the CL term is weak
  (I_NCE ≈ 1.9 nats; at τ 0.5 the bound is ≤ log K − log(1 + (K−1)e^{−4}) ≈ 4.0 nats, **inferred**).
- No test predictions and no plots exist for this scan.
- Seed time on 2 × A40/L40S: CE-only ~3.1–3.8 h, CL ~6.3–10.7 h.

## 5. CE-only references vs LR (all weighted CE, seed 42 unless n given) — **verified**
| LR | recipe | best val CE | note |
|---|---|---|---|
| 1e-4 | 1 × 6000, 40 ep, 1% wu (gls_tau_lambda_lr ref) | 0.60401 @11 | |
| 3e-4 | same | 0.60090 @11 | |
| 5e-4 | same (gls_tau_lambda ref) | 0.60005 @9 | |
| 5e-4 | 2 × 3000, 100 ep (10-01 scan), n = 8 | 0.60038 ± 0.00037 | |
| 1e-3 | 1 × 6000, 40 ep | 0.59947 @11 | |
| 1e-3 | 2 × 3000, 100 ep, 5% wu, ES 20 (10-14 scan), n = 6 | 0.59951 ± 0.00035 | |
| 1.5e-3 | 1 × 6000, 40 ep (run_20261009_160205), n = 8 | 0.59861 ± 0.00042 | 0/8 diverged; val rej@best 37.2 / 900 / 222 |
| 1.9e-3 | 1 × 6000, 40 ep (run_20261011_235028), n = 8 | 4/8 diverged (ep 3–6, best 0.631–0.639); survivors 0.59574, 0.59571, 0.59577, 0.59817 | survivors' val rej 38.1 / 958 / 222; best epochs 36–39 (still improving at ep 39) |
| 2e-3 | 1 × 6000, 40 ep | 0.62751 @2 (diverged) | |
- Unweighted CE-only (09-28): 0.60075 ± 0.00029 (n = 8). salt-pt `GN2_only_jt` (fm4tag's "GN2" reference, single seed, same
  architecture, weighted CE, LR 5e-4, 2 × 3000, **50-epoch** schedule, fp16, no ES): 0.59814 @15; test @70 38.0 / 928 / 229,
  @77 15.10 / 284.5 / 65.4 (**verified**, file `/storage3/DSIP/rriva/packages/salt-pt/salt/logs/GN2_only_jt_20260817-T134649/`).
- Seed std of a CE-only cell: 0.0003–0.0004 → effects < ~0.0008 are not resolvable with 4–8 seeds.

## 6. HPO studies (Optuna TPE, journals parsed with plain python; objective = best weighted val CE, 1 GPU × 6000, 40 epochs)
| study | search | trials (state at end) | best trial (val CE) | params | verdict |
|---|---|---|---|---|---|
| gls_tau_lambda (run_20261005_172909) | τ 0.05–1, λ 0.05–0.7, LR 5e-4 | 32 (21 C, 7 P, 4 R; cancelled 10-09) | t3 0.59897 @35 | λ 0.439, τ 0.102 | vs CE-only 5e-4 seed-42 ref 0.60005: −0.0011, but isolated among diverged/pruned neighbours; below CE-only at 1e-3/1.5e-3 |
| gls_tau_lambda_lr (run_20261006_153550) | + LR 1e-4–2e-3 | 21 (14 C, 3 P, 4 R; cancelled 10-09) | t14 0.59873 | λ 0.23, τ 0.59, LR 1.30e-3 | ≈ CE-only at 1.5e-3 (0.59861 ± 0.00042) |
| gls_tau_lr (run_20261007_123722) | τ, LR; λ 0.44 | 40 (33 C, 3 P, 4 R; stopped 10-12) | t20 0.59770 | τ 0.995, LR 1.94e-3 | weak CL (τ ~1) at the LR edge; CE-only at 1.9e-3 reaches 0.5957 |
| gls_tau_lr_bs | + batch 3000–24000 | never ran (cancelled pending) | – | – | – |
| wsum_tau_lr_w (run_20261009_160333) | τ, LR, w 0.01–3; λ 0.44 | 37 (29 C, 8 F) | t29 0.59548 @38 | w 0.021, τ 0.21, LR 1.90e-3 | w → 0: CE-only; w ≥ 0.8 gives 0.603–0.625 |
| wsum_tau_lr_w_lambda (same ts) | + λ 0–0.7 | 37 (27 C, 2 P, 8 F) | t31 0.59644 | λ 0.025, w 0.17, τ 0.91, LR 1.90e-3 | TPE drives λ → 0 (identical views) |
| wsum_strong_cl (run_20261014_141456) | w 0.1–1, λ 0.1–1, τ 0.1–1, LR 5e-4–2e-3; 5% wu | 9 (5 C, 4 R; cancelled 10-15) | t0 0.59899 | w 0.104, λ 0.93, τ 0.149, LR 1.07e-3 | no 5%-warm-up CE-only ref; ≈ CE-only at 1e-3 |
| wsum_strong_cl_lam044 (run_20261014_141652) | same, λ 0.44 | 8 (4 C, 4 R; cancelled 10-15) | t2 0.59888 | w 0.323, τ 0.566, LR 1.78e-3 | ≈ CE-only at 1.5e-3 |
- General conclusion: **no CL setting beats a CE-only reference at matched LR by more than ~1σ (0.0004)**; every apparent gain is
  explained by the LR (and schedule), and when the search is free it switches CL off (w → 0.01–0.02, λ → 0.02–0.16, τ → 1).
  Trials at LR ≥ 1.8–1.9e-3 diverge (wsum t17 at ep 3–6, gls_tau_lr t5).
- Pruner caveat (memory): salt's median pruner compared a trial's current value with others' best-so-far → biased against slow
  (regularised) trials; replaced by a custom `best_median` pruner (commit 54dd04d) from 10-12 on.

## 7. GN2 CE-only on the 152051 train slice (run_20261106_092332) — fair GN2 reference for the SSL campaign — **verified**
- Recipe = 10-14 recipe (2 × 3000, LR 1e-3, 5% warm-up, 100 ep, ES 20, weighted CE with the train slice's own class/norm dicts),
  train 18.6M, val 1.86M; job 6796870, git 6e43f64; 8/8 seeds completed by ~2026-11-07 01:20, ~1.8–2.2 h per seed on 2 × A40.
- Best val CE (weighted) **0.61463 ± 0.00062** (n = 8; per seed .61408 .61407 .61544 .61505 .61524 .61465 .61479 .61370), best
  epochs 10–16, 0 divergences. Val rej@best (70% WP, 152051 val): c 30.7 (29.5–31.5), light 679 (617–755), tau 168 (141–187).
  Eff. rank / uniformity of the pooled embedding: 24.9 ± 1.5 / −0.70 ± 0.06.
- **No `salt test` predictions exist yet → no test rejections**; not yet used anywhere in fm4tag (only an squeue line in the fm4tag chat).
- val CE not comparable with fm4tag SC (fm4tag uses unweighted CE: `use_class_weights: false` in
  `fm4tag-chunk/slurm/ssl_campaign/F_final_20261028_235642/sc/seed_4242/.../.hydra/config.yaml`).

## 8. Instabilities and calibration
- Divergence counts (my flag): jc 0.3 / LR 3e-4 / τ 0.1: 0/41; jc 1 / LR 5e-4 / τ 0.1: 14/16 CL seeds (CE-only 0/8); CE-only LR 1.9e-3
  (1% wu, 1 × 6000): 4/8, early drift at ep 3–6; CE-only LR 2e-3: 1/1; 10-13 scan (LR 1.9e-3, 1 × 3000, 5% wu): 8/11; 10-14 scan
  (LR 1e-3, 2 × 3000, 5% wu): 0/13 before best. Mechanism (memory): loss spikes at LR 80–95% of peak with no clipping, train CE and
  CL rising together; ES keeps the pre-collapse checkpoint, so divergences show up as worse best val CE.
- Calibration check (run `slurm/calibration/output/run_20261006_174236`, CE-only, LR 5e-4, checkpoint every epoch, `salt test` on val):
  from epoch 9 to 24 the weighted val CE rises +0.0090 raw (0.60004 → 0.60908), +0.0071 after temperature scaling, +0.0058 after
  T + per-class bias; fitted T 1.03 → 1.10 → the post-minimum rise is mostly genuine overfitting, not overconfidence; c-rej@70
  keeps rising to ~37.7 at epoch 18 while CE turns up at 9 (**verified**, `plots/calibration_vs_epoch.csv`). Light rejection is
  ±10% noisy epoch-to-epoch, so best-over-epochs rejections are biased up (memory).

## 9. Plots available (absolute paths) and the configs that made them
- `/storage3/DSIP/rriva/research/salt-ririva/slurm/scratch_lambda_scan/output/run_20260928_195621/plots/val_loss_ce_lambda_scan.png`
  (final, 10-07; `slurm/scratch_lambda_scan/plot_val_curves.py --config slurm/scratch_lambda_scan/configs/val_loss_lambda_scan.yaml`):
  every seed's val CE vs epoch, coloured by λ, stars at best. Shows the delayed convergence with λ. Paper candidate.
- `.../run_20260928_195621/plots/best_vs_lambda_jets_classification_loss.png` (final, 10-07; `plot_best_vs_lambda.py`, same config):
  best val CE vs λ, mean ± std + single seeds. **Main GN2 paper figure candidate.**
- `.../run_20260928_195621/plots/roc_lambda_scan.png` (`plot_roc.py --config configs/roc_lambda_scan.yaml`, 10-01) — **STALE**:
  predates the final seeds (no λ 0.6/0.7, old best seeds); predictions for the new best seeds exist, re-run plot_roc.py.
- `/storage3/DSIP/rriva/research/salt-ririva/slurm/scratch_lambda_scan/output/run_20261001_091525/plots/` (final, 10-06; configs
  `*_lambda_scan_jc1.yaml`): `val_loss_ce_lambda_scan.png`, `best_vs_lambda_{jets_classification_loss,cjets_rej_70,ujets_rej_70,
  taujets_rej_70}{,_at_best_ce}.png`, `roc_lambda_scan.png`, and `train_epoch/*.png` (train CE/CL per step of every CL seed —
  divergence illustration; `scripts/train_step/plot_train_step.py`, untracked).
- `/storage3/DSIP/rriva/research/salt-ririva/slurm/calibration/output/run_20261006_174236/plots/calibration_vs_epoch.png`
  (`slurm/calibration/calibration_vs_epoch.py`).
- Nothing for the 10-13/10-14 scans, the HPO studies (only Optuna API snippet in the README) or the 11-06 run.
- All plot texts are in the YAML configs; ATLAS label settings (`atlas_first_tag: Simulation Internal`) are in the ROC configs —
  the fm4tag convention is no "ATLAS" on any plot (fm4tag memory), check before reuse.

## 10. Pretrain → fine-tune in salt
Overlays exist (`salt/configs/ssl/{pretrain,pretrain_saint,finetune,probe}.yaml`), but no salt pretraining / FT / FR run exists
(no run dirs, `logs/` has only 4 GN2 test dirs from 09-28). The GN2 side of pillar 2 is only the 11-06 CE-only (SC) reference.
