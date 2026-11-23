# Repository and experiments inspection — 2026-11-23

An inspection of the fm4tag repository, its worktrees and the experiments run so far, done on 2026-11-23 before
cleaning up the repository and merging the recent work into `main`. Nothing was changed during the inspection. This
file is updated as decisions are taken (see [Decisions and log](#13-decisions-and-log)).

Paths are absolute. `chunk` below means `/storage3/DSIP/rriva/research/fm4tag-chunk`; `main` means
`/storage3/DSIP/rriva/research/fm4tag`.

---

## 1. State of the repository

| Worktree | Branch | State on 2026-11-23 |
|---|---|---|
| `/storage3/DSIP/rriva/research/fm4tag` | `main` = `origin/main` (`d858d71`) | Oldest of the active lines (last work 2026-10-12). Uncommitted change to `scripts/roc_comparison.py`; untracked `paper/` (464 KB) and `plot_paths.txt`. |
| `/storage3/DSIP/rriva/research/fm4tag-chunk` | `feat/chunked-ia` (`f713105`) | 126 commits ahead of `main`, 0 behind; never pushed. All work 2026-10-12 → 2026-11-23. Untracked: `docs/slides/` (31 MB decks), `docs/ssl_campaign_report.md`, `docs/ssl_tests_plan.pptx`, `docs/ssl_tests_plan_preview.jpg`. |
| `/storage3/DSIP/rriva/research/fm4tag-chunk-s0` | detached `6565953` | An ancestor of `feat/chunked-ia` (a frozen snapshot for the S0 benchmark). Only git-ignored content: `slurm/profiling/` (3 MB). |
| `/storage3/DSIP/rriva/research/fm4tag-old` | `old-version` (`8378ab0`) | Clean: no untracked and no ignored files. Branch already contained in `main`'s history (152 commits behind). |

Merge feasibility:
- `main` is an ancestor of `feat/chunked-ia`, so a **fast-forward** is possible: one author, the largest added file is 44 KB,
  and the branch diff contains no secrets.
- **Blocker:** the uncommitted change to `scripts/roc_comparison.py` in `main` (an Agg-canvas fix for puma's `draw()` and
  a `wp_line` option) is **not** on the branch, and the branch changes the same file (149 diff lines). Git refuses
  the fast-forward until the change is committed or stashed. It must then be re-applied and the overlap resolved by hand.
- **Absolute paths:** 42 tracked files on the branch contain 311 absolute references to `/storage3/DSIP/rriva/research/fm4tag-chunk`:
  36 files in `scripts/configs/` (cells files and YAML configs pointing at result folders and checkpoints), 5 slurm
  scripts with `REPO=...fm4tag-chunk`, and `scripts/make_wsd_cells.py`.

Git-ignored results (not in git):

| Where | Size | Content |
|---|---|---|
| `chunk/slurm/ssl_campaign/` | 101 GB | every SSL-campaign run (stage dirs, checkpoints, metrics) |
| `chunk/plots/` | 3.6 GB | all campaign plots and test-metric tables |
| `chunk/slurm/{arm_roc,plot_logs,smoke,profiling}` | about 0.5 GB | inference / plot logs, smoke tests |
| `main/scripts/plots/` | 24 GB | λ-scan plots, ROC galleries, fraction searches |
| `main/slurm/classify_from_scratch/`, `classify_param_scan/` | 16 GB each | λ-scan and CE+JC grid runs |
| `main/slurm/old_tests/` | 1.8 GB | first pretrain / finetune tests |
| `main/outputs/`, `main/plots/`, `main/scripts/logs/` | about 150 MB | early runs, definitive λ-scan plots, logs |

### Branches and stashes

| Branch | Ahead / behind `main` | Status | Recommendation |
|---|---|---|---|
| `feat/chunked-ia` | 126 / 0 | all recent work | fast-forward `main` to it |
| `chore/gpu-num-overridable` | 0 / 0 | = `main` | delete |
| `profiling/rowmode-kernel-study` | 0 / 19 | merged, local only | delete |
| `refactor/model-reframing` | 0 / 50 | merged (also on origin) | delete |
| `review/all`, `review/claude-deep-dive` | 0 / 82 | merged, same commit (also on origin) | delete |
| `old-version-no-2` | 0 / 122 | merged | delete |
| `refactor/multi-view-uscl` | 0 / 128 | merged (also on origin) | delete |
| `refactor/jet-embedding-loss` | 0 / 129 | merged, remote already gone | delete |
| `old-version` | 0 / 152 | merged, checked out in `fm4tag-old` | delete after removing the worktree (see section 9) |
| `categorical-jet-features` | 2 / 124 | **unmerged** (June: categorical jet features, "wip", +1464 lines) | keep for now |
| `refactor/deep-dive` | 5 / 154 | **unmerged** (May: framework refactor, "multi-view WIP: not working") | keep for now |

Stashes (kept for now): `stash@{0}` is an old tweak of slurm paths and GPU counts; `stash@{1}` comes from a branch that no
longer exists (`experiment/embedding-management`, pretraining eval metrics, 433 changed lines).

---

## 2. What changed in the code since `main`

- **Models**
  - Row attention is all-pairs inside chunks of jets, through flash varlen (`RowGroups`); ISAB was deleted.
  - The out-projection is zero-initialised, and QK-norm (RMSNorm) was added to the row attention against logit blow-ups.
  - The aggregator runs on packed valid tracks, one group per jet.
  - Attention dropout uses one SDPA call per group length. Kernel launches are sliced under the 2^31 and 65535 limits.
- **Training**
  - The contrastive loss is computed inside jet chunks, without a cross-rank gather.
  - Downstream modes are from scratch, fine-tuned and frozen: parameter groups by provenance, backbone lr, staged
    unfreeze, `freeze_encoders`. A checkpoint records whether its aggregator was trained.
  - Pretraining clips the gradient norm at 1.0, and `last.ckpt` follows training to the final epoch.
  - A warmup-stable-decay (WSD) schedule was added, plus `train_max_rows` and monitors for memory, gradient norm and
    row-attention logits.
- **Configs:** a new `arm` group (`pretrain`, `scratch`, `finetune`, `frozen`), `campaign_ssl.yaml`, the
  `rowcol_concat_v0` encoder preset, dropout keys, and schedule keys (`schedule`, `warmup_epochs`, `decay_frac`).
- **Cluster tooling**
  - `slurm/ssl_pack.sh` runs a cells file × seeds as 1-GPU trainings inside 2-GPU jobs that share a task queue,
    respecting the 10-job limit.
  - The shared-card ledger (`slurm/pack_gpu.sh`, `SLOTS_PER_GPU`, `@mem`) runs several trainings per card.
  - `slurm/gpu_preflight.sh`, batch-size sweeps and launchers. See `docs/gpu_packing.md` and `docs/ssl_campaign.md`.
- **Analysis:** scripts for per-epoch curves, summaries, test rejections (`arm_test_metrics.py` with fraction searches),
  ROC vs GN2, the lr search, the deck builder and the cell generators. All plot wording lives in YAML. There are 13 new
  test files (+1537 lines).

---

## 3. Experiments (all, chronological) and where the outputs are

Before fm4tag (2025): SAINT-based pretraining, own Pythia/Delphes samples, salt/GN2 learning. Recorded in
`/storage3/DSIP/rriva/research/diary/`, not in this repo. Full timeline:
`main/paper/factsheets/history_presentations.md`.

| Period | Experiment | Result | Outputs |
|---|---|---|---|
| 2026-03 → 06 | first fm4tag tests: column-transformer pretraining, scratch / FT / FR | exploration; DDP and intersample-attention issues found | `main/outputs/`, `main/slurm/old_tests/` |
| 2026-07 → 08 | CE + JC grids (lr × jc, concat jc × λ), review fixes, TabICL/ISAB port, kernel profiling | `per_token` model; JC needs a noise level λ | `main/slurm/classify_from_scratch/`, `main/slurm/profiling/` |
| 2026-08-18 → 10-12 | "beat GN2" campaign: λ (noise) scan of CE + JC, 52M-jet dataset `251112_112022`, seeds, fraction search | JC with λ ≥ 0.25 beats no-JC on val CE (0.6015–0.6024 vs 0.6035) and passes GN2's c-rejection with test-optimised fractions (best +0.6% vs GN2; no-JC −2.2%) | `main/slurm/classify_param_scan/`, `main/scripts/plots/lambda_seed_bestval_*`, `run_20261001_*`, `main/plots/val_loss_ce/`; fact sheet `main/paper/factsheets/fm4tag_lambda_scan.md` |
| 2026-09-28 → 10-14 | λ scan ported to salt (GN2 + contrastive) | neutral at λ ≤ 0.4, worse at λ ≥ 0.5 (41 seeds) | `/storage3/DSIP/rriva/research/salt-ririva/slurm/scratch_lambda_scan/output/` |
| 2026-10-16 → 10-21 | SSL campaign S0 benchmark, S1 architecture, S2 pretraining recipe | concat attention, chunk 512, 40-epoch downstream; T 0.1, λ 0.4 for P0/P1/P2; clipping and QK-norm fix pretraining blow-ups | `/storage3/DSIP/rriva/research/fm4tag-chunk-s0/slurm/profiling/`, `chunk/slurm/ssl_campaign/{S1_arch_*,S2_*}` |
| 2026-10-23 → 10-28 | full pretrainings P0/P1/P2 (30 ep, seeds 42/424); S3 lr per arm | the checkpoints used downstream | `chunk/slurm/ssl_campaign/P_full_20261023_085735` (P0, P2), `P_full_20261026_094445` (P1), `S3_*` |
| 2026-10-28 → 11-12 | finals (4 runs per arm) and controls: SC+JC clipping, FT/FR from P1 with CE only, label-fraction scan, val-CE gap check | SC 0.6240 < FT+JC←P1 0.6288 < FT←P0 0.6312 < FR+JC←P1 0.6436 < FR←P0 0.6500; FT←P1 CE-only 0.6158 beats SC (test c-rej 10.8 vs 9.7, GN2v01 15.6); pretraining never beats SC at 1–100 % labels | `chunk/slurm/ssl_campaign/{F_final_*,F_p1_nojc_*,F_scjc_*,L_lr_*,L_reg_*,L_scan_*,C_gap_*}`; `chunk/plots/{arm_comparison,arm_comparison_p1,p0_vs_p1,label_scan,pretrain_curves}` |
| 2026-11-06 | GN2 (salt) trained from scratch on the same 18.6M slice, CE only, 8 seeds | best class-weighted val CE 0.6137–0.6154 (see section 5) | `/storage3/DSIP/rriva/research/salt-ririva/slurm/scratch_lambda_scan/output/run_20261106_092332/` |
| 2026-11-13 → 11-16 | dropout 0.2 downstream (24 runs, 80 ep); P0/P1/P2 pretrained again with dropout 0.1 | FT←P0 + dropout best (0.6105; test c-rej 12.1 = 1.25 × SC); all JC + dropout cases fail; P2 s424 collapsed (replaced by s4242) | `chunk/slurm/ssl_campaign/{D_drop_2026111*,P_drop_20261116_132628}`; `chunk/plots/dropout/` |
| 2026-11-17 → 11-23 | WSD lr search: 66 runs, batch 12800 for all, 40 ep, no early stopping, seed 42 | FT←P0 + dropout gain survives lr tuning (0.608 vs 0.6305); most cases flat in lr | `chunk/slurm/ssl_campaign/{W_lr_full_*,W_lr_shared_*,W_measure_*,W_share_*}`; `chunk/plots/wsd_lr_search/` |

Design and decision logs: `chunk/docs/ssl_campaign.md` (tracked) and the header comment of each cells file in
`chunk/scripts/configs/campaign/`. Decks: `chunk/docs/slides/` (untracked, last rebuilt 2026-11-12).

---

## 4. What we have, what is still to do

**Have:**
- the λ scan (fm4tag main era) with a positive result for CE + JC on 52M jets
- a full SSL campaign on 18.6M jets: pretraining gives no gain over scratch except FT←P1 with CE only and FT←P0 with
  dropout
- a protocol-consistent lr search under WSD
- 8 GN2 trainings on the same slice
- a paper skeleton in `main/paper/`

**Still to do:**
1. WSD finals: 16 cells × 4 seeds × 80 epochs ≈ 1,550 card-hours; first 3 extra lr runs for the cases at the
   edge of the grid (SC + dropout at 3e-3 and 6e-3, FR←P0 at 3e-3).
2. Test evaluation of the GN2 runs, as a same-data reference (section 5).
3. Downstream from the dropout-pretrained checkpoints (`P_drop`), not launched.
4. ~~Duplicate check of the training file~~: already done on 2026-10-28 (criticality 3).
5. A controlled JC test (section 8).
6. Rebuild the deck and paper numbers from the finals.

---

## 5. Questions checked on 2026-11-23

### 5.1 Package versions of the two virtual environments
The two `.venv` (`main/.venv`, `chunk/.venv`) are **identical**: Python 3.13.7, the same 137 packages at the same
versions (torch 2.10.0, lightning 2.6.1, triton 3.6.0, numpy 2.3.5, pandas 3.0.1, h5py 3.15.1, hydra-core 1.3.2,
puma-hep 0.5.1). `pyproject.toml` and `uv.lock` are unchanged between the branches. The only difference is the editable
install of fm4tag: each venv points at its own `src/`. After the merge, `main/.venv` already points at the right
place, so `chunk/.venv` can be deleted with nothing lost.

### 5.2 GN2 trained on the 18.6M dataset — yes
- **Sweep:** salt `run_20261106_092332` (repo `salt-ririva`, git `6e43f64`, config `salt/configs/ssl/GN2.yaml`), submitted
  2026-11-06.
- **Data:** the same files as the SSL campaign (`data_parallel_preprocess_251112_152051` train / val / test).
- **Setup:** CE only (`jet_contrastive=0`), batch 3000 per rank on 2 GPUs (6000), lr 1e-3, one-cycle with 5 % warmup,
  early stopping with patience 20, max 100 epochs.
- **Seeds:** 8, all completed (31–37 epochs). Best validation loss 0.6137–0.6154 (mean ≈ 0.6146), at epochs 10–16.
- **Not comparable yet:**
  - salt's loss is **class-weighted** CE (`use_class_dict: True`, weights 1.5, …), while fm4tag reports **unweighted** CE
    (`use_class_weights: false`).
  - No test evaluation file was found for these runs.
- **Next:** evaluate them on the test set and add them as a reference arm in `arm_test_metrics` (test rejections,
  and an unweighted CE). This is the like-for-like GN2 baseline that GN2v01 (trained on other data) cannot be.

### 5.3 Class-weighted cross-entropy — never used in fm4tag (checked 2026-11-23)
- **The option:** fm4tag has had `use_class_weights` (per-class CE weights from `class_dict.yaml`) since 2026-07-06,
  with default `false`. No branch has ever set it to `true`.
- **The runs:** the saved run configs all have `use_class_weights: false` and `class_weights: null`. That includes the
  157 λ-scan configs (`classify_param_scan`), the 78 CE + JC grid configs (`classify_from_scratch`), and every SSL stage
  checked (S1, F_final, D_drop, W_lr_full, W_lr_shared).
- **Where weighted CE was used:** only in salt. Salt scan 2 (from 2026-10-01) and the GN2 runs (including the 18.6M one)
  use weights b 1.5, c 1.5, light 1.0, τ 4.0.
- **Consequence:**
  - fm4tag and GN2 are trained on different losses. Their validation losses cannot be compared.
  - The rejections are partly shaped by the loss, not only by the model. The τ weight of 4 pushes GN2 towards τ
    rejection. This matches what the λ scan saw at nominal fractions: fm4tag had far higher light rejection and far
    lower τ rejection than GN2.
- **To decide before the finals / four-way test:** either keep unweighted CE in fm4tag (the whole history uses it) and
  compare to GN2 only through test rejections, or switch fm4tag to the GN2 weights for the comparison runs. Switching
  changes the selection metric, so the lr search would strictly have to be redone.

---

## 6. Criticalities of the work

Severity: H = can change a conclusion, M = weakens a claim, L = hygiene.

1. **H — too few independent samples, no uncertainty analysis.** The pretrained arms use 2 pretraining seeds × 2 downstream
   seeds. A check of the existing runs on 2026-11-23 shows the split of the spread differs by arm:
   - **Fine-tuned:** the pretraining seed matters little (the two seed means differ by 0.0003–0.0008 in val CE, about
     the downstream spread).
   - **Frozen:** the pretraining seed dominates (frozen from P1: 0.0053 between the two pretraining seeds vs 0.0010
     within; c-rej 7.73 vs 7.11).

   So the frozen means rest on effectively two samples. Other problems:
   - The lr search used one seed, with differences of about 0.003.
   - From scratch with dropout (0.6406) is driven by one seed.
   - c-rejection is much noisier than val CE (e.g. FT←P0 c-rej sd 0.44 within a pretraining seed).
   - No confidence intervals or tests are reported anywhere.

   See section 7.
2. **H — protocols differ between stages that are compared with each other.** Batch size per arm (12800 to 94208),
   schedule (40-epoch cosine, 80-epoch cosine, 40-epoch WSD), early stopping on/off, and lrs tuned per arm on one seed
   all vary. The pretrainings differ too:
   - P1 with dropout ran at batch 10752 instead of 12288 (fewer contrastive negatives).
   - P0 seed 42 with dropout was early-stopped at epoch 21, and downstream would load its last, not best, checkpoint.
   - One P2 seed was replaced.

   Many plots and the deck mix these protocols. The WSD redo fixes this for the downstream runs only.
3. **H — unexplained overfitting.** Without dropout, from scratch / FT←P0 / FR←P0 peak at epochs 5–7 and then val CE
   rises to 0.78–0.97, even for frozen encoders on 18.6M jets. *Corrected 2026-11-23:* the duplicate check **was** done
   on 2026-10-28 (job 6670528, `experiments/2026-10_ssl-campaign/diagnostics/dup_check.py`, log in
   `diagnostics/outputs/dup/`). In the 18.6M dataset, train / val / test / pretrain share **no** jet; about 2 % of the
   training rows are resampled copies (τ: 1.24 rows per distinct jet). So duplicates do not explain the overfitting,
   which stays unexplained. The λ-scan dataset (`112022`) shares about 4 % of its jets with every split of `152051`,
   so models trained on one must not be tested on the other.
4. **M/H — evaluation.**
   - GN2v01 was trained on other data, so "beats GN2" is not like-for-like (the same-data GN2 now exists, section 5.2).
   - Fractions optimised per run make gaps look larger (FT←P0 + dropout: 1.25 × SC optimised vs 1.15 × at nominal
     fractions).
   - **In the λ scan the fractions were optimised on the test set** (fact sheet `fm4tag_lambda_scan.md`, § 3).
   - Test-set statistical errors on rejections are not shown (the binomial error is small, ±0.3 % on c-rej at 80 %,
     smaller than the run-to-run spread).
5. **M — the benefit of JC is not shown to be a contrastive effect** (clarified on 2026-11-23, see section 8).
6. **M — pretraining protocol.** Early stopping and the last-vs-best checkpoint choice differ between pretrainings. The
   recipe was chosen on frozen probes with few seeds. Pretraining is fragile (blow-ups, the P2 collapse).
7. **M — code coverage.** Attention dropout is tested mostly on CPU. The shared-card ledger (`slurm/pack_gpu.sh`,
   `pack_worker.sh`) has no automated test; it was checked with a fake command and then worked in the lr search.
8. **M — reproducibility.**
   - Results live only in git-ignored folders of one worktree, referenced by 311 absolute paths.
   - The deck (last rebuilt 2026-11-12) is stale, and `docs/ssl_campaign.md` was not updated after the finals.
   - Manual interventions (killed, renamed and resumed runs) are recorded only in the assistant's notes and in one
     `CHANGES.txt` (`P_drop` stage).
   - Report material (`paper/`, `docs/slides/`) is outside git.
9. **L — repository clutter:** 9 merged branches, 2 old stashes, 2 unused worktrees.

---

## 7. How to make the experiments consistent and systematic (criticality 1, and 2)

1. **Write a protocol card before each study, and freeze it** (a short YAML or md next to the cells file). It holds:
   - the question
   - one primary metric (best val CE, which tracks optimised c-rejection closely: Spearman −0.98 in the λ scan)
   - the secondary metrics (test c/light/τ rejection at nominal **and** at val-optimised fractions)
   - the planned comparisons (only these are tested)
   - seeds, protocol version and stopping rule

   Results that break the card are reported as exploratory.
2. **One canonical protocol** (`protocol_v1`): batch 12800 for every arm, WSD 80 epochs, no early stopping, the
   best-val checkpoint, the same dropout policy per study, and the lr chosen on a tuning seed that never enters the finals.
   The same applies to pretraining: one batch size, no early stopping, always the last checkpoint (or always the best,
   decided once). Every plot and table states its protocol version; results from other protocols are labelled
   "superseded" and not mixed.
3. **Seeds as a nested design**, sized with the variance already measured:
   - Frozen arms need **≥ 4 pretraining seeds** (the pretraining seed dominates their spread). Fine-tuned arms can stay
     at 2 pretraining × 2 downstream seeds.
   - Use the **same downstream seeds in every arm** (paired design), so differences between arms are computed per
     seed pair and the shared noise cancels.
   - From the measured sd (0.0005–0.003 in val CE), 4 runs detect differences of about 2–3 sd. State that as the
     smallest effect the study can see.
4. **Uncertainties in every table:**
   - Report mean ± 95 % CI: a t-interval over pretraining-seed means for pretrained arms, over runs for scratch.
   - Report paired differences between arms with their CI.
   - For test rejections, add a bootstrap over test jets, and report the bootstrap and run-to-run errors separately.
5. **Never optimise on the test set.** Fractions are chosen on validation (already the case in the SSL campaign, not in
   the λ scan, which should be redone on validation before the paper). The test set is evaluated once, for the final
   models.
6. **Measure the noise floor:** rerun one configuration with the same seed 2–3 times to see the GPU nondeterminism, so a
   "difference" smaller than that is never reported.
7. **One analysis entry point:** a single script reads the experiment registry (section 9) and produces all
   tables and plots. Cross-stage mixing then needs an explicit, visible choice instead of happening silently.

---

## 8. Clarification of criticality 5 (JC) — 2026-11-23

The question was whether the October λ scan had shown that JC helps. It did, in that setup:

| Comparison (λ scan, 52M jets) | Best val CE |
|---|---|
| no JC (regime A, 50 epochs, n = 4) | 0.60345 ± 0.00049 |
| λ 0.25 / 0.3 / 0.35 (regime A) | 0.60243 / 0.60192 / 0.60153 |

With test-optimised fractions, c-rejection went from −2.2 % vs GN2 (no JC) to +0.5 % (λ 0.45).

Why it is still listed as a criticality:
- **A dropout baseline matches it.** In the same scan, no JC + dropout 0.1 reached **0.60162 ± 0.00096** (n = 6), as
  good as the best λ. JC delays overfitting (best epoch moves from about 8 to 60–99), so its gain looks like
  regularisation, not something specific to the contrastive objective.
- **LR confound.** Without JC, runs stop at peak lr; the best high-λ seeds stop with the lr annealed (Spearman 0.91
  between lr at the best epoch and best CE).
- **Test-set fractions.** The fractions were optimised on the test set.
- **Not reproduced elsewhere.** In the salt port, contrastive was neutral (λ ≤ 0.4) or worse (41 seeds). In the SSL
  campaign, CE + JC never beat CE only:

  | Case | CE only | CE + JC |
  |---|---|---|
  | from scratch (finals) | 0.6240 | 0.6241 |
  | from scratch (WSD) | 0.6188 | 0.6219 |
  | fine-tuned from P1 (WSD) | 0.6154 | 0.6282 |
  | frozen from P1 (WSD) | 0.6307 | 0.6357 |

**Suggested test:** the same model, data, WSD schedule and seeds, comparing CE vs CE + JC vs CE + dropout vs
CE + JC + dropout (with λ and the JC weight scanned a little). This separates "contrastive" from "regularisation".

---

## 9. Worktrees and experiment types — proposal (answer to the concern about deleting worktrees)

What each worktree holds:
- `fm4tag-old`: **no experiments at all**. It is clean and has no ignored files. Its code (`old-version`) is already in
  `main`'s history.
- `fm4tag-chunk-s0`: only the S0 benchmark profiles (3 MB).
- The experiments are in `main` (pre-October campaigns) and `chunk` (SSL campaign).

So removing the two old worktrees loses nothing once the S0 profiles are copied. To keep each code state easy to reach,
**tag** it before deleting a branch, e.g. `archive/old-version` and `archive/s0-snapshot`. A tag costs nothing and
`git checkout <tag>` gives the code back.

Proposed layout in `main` (git-ignored, but each campaign has a small tracked description):

```
results/
  README.md                         index: one row per campaign (question, dates, code tag, protocol, status, headline)
  2026-03_first-tests/              <- main/outputs, main/slurm/old_tests
  2026-07_ce-jc-grids/              <- main/slurm/classify_from_scratch
  2026-08_lambda-scan/              <- main/slurm/classify_param_scan, main/scripts/plots/*
  2026-08_kernel-profiling/         <- main/slurm/profiling
  2026-10_ssl-campaign/             <- chunk/slurm/ssl_campaign (stage dirs kept as they are), chunk/plots
    S0_benchmark/                   <- fm4tag-chunk-s0/slurm/profiling
  2026-11_gn2-same-data/            (pointer to the salt sweep, not copied)
docs/experiments/<campaign>.md      tracked card per campaign: question, design, code tag, launch commands,
                                    stage list with status, headline numbers, plots, manual interventions
```

- **What a stage is:** a name such as `F_final`, `D_drop` or `W_lr` already says it. The card for its campaign says
  what it means.
- **What code made it:** every run dir already holds its `.hydra/` config and `run_info.yaml`, and every stage dir a
  `sweep_info.txt` with the git hash.
- **Moving:** all paths are on the same filesystem (`/storage3`), so moving the 101 GB is a rename and takes seconds.
  The 311 absolute paths in tracked files must then be rewritten once, ideally to a single `results_root` setting.

---

## 10. Last pretraining runs (P_drop, dropout 0.1) and which pretrainings to use — 2026-11-23

All six runs ended. Validation loss is in eval mode (no dropout):

| Pretraining | No dropout (P_full, batch 12288) | Dropout 0.1 (P_drop) |
|---|---|---|
| P0, seed 42 | 5.442 (30 ep) | **early-stopped at epoch 21**; best 5.585 at epoch 8, last 5.760 |
| P0, seed 424 | 5.505 | 5.454 (30 ep) |
| P1, seed 42 | 5.668 | 5.914 (batch 10752) |
| P1, seed 424 | 5.729 | 5.778 (batch 10752) |
| P2 | 0.222 / 0.223 | 0.269 (s42) / 0.278 (s4242); **s424 collapsed** |

Effective rank of the embeddings at the end:
- **Jet embedding:** 55–59 with dropout vs 62–63 without.
- **Track embedding:** 118–120 vs 114–126.
- **P1 aggregator:** 132 vs 129–130.

**Reading:**
- **On the pretraining objective, dropout did not help.**
  - P0 is equal at best, and one seed stopped early.
  - P2 is worse, and one seed collapsed.
  - P1 is worse even though it ran at a smaller batch. Fewer negatives normally make a contrastive loss *smaller*, so
    a higher value is a real degradation.
- **Not yet known:** whether these checkpoints make better starting points. No downstream run uses them.

**Recommendation:**
- **Use the no-dropout pretrainings for everything now:** P0 from `P_full_20261023_085735/p0/seed_{42,424}` and P1 from
  `P_full_20261026_094445/p1/seed_{42,424}`, 30 epochs, batch 12288.
  - They are complete and consistent with each other.
  - They are what the WSD lr search was tuned on (seed 42).
  - The two best downstream results so far start from them (FT←P0 + dropout, FT←P1 CE only).
- **P2 has no downstream arm** and stays aside.
- **P_drop is a separate question** ("does dropout during pretraining help downstream?"). It can be answered later
  with a small FT/FR test at the chosen lrs. Its P0 seed 42 would need care: the checkpoint loaded downstream is the
  last (epoch 20), not the best (epoch 8).
- **If the frozen-encoder claims matter:** add 2 more no-dropout pretraining seeds for P0 and P1 (section 7, point 3),
  about 2.5–3.5 days each on one GPU.

---

## 11. What the WSD lr search says about the training phase, and whether to keep it — 2026-11-23

What it showed (66 runs, one seed, batch 12800 for all, WSD 40 epochs, no early stopping):
1. **The choice of lr hardly matters.** Over a factor of 4 the best validation CE moves by 0.001–0.004 in most cases,
   about the seed noise. The exceptions are from scratch with dropout (strongly better at the highest lr, at the edge
   of the grid) and the JC + dropout cases (erratic).
2. **The new protocol is as good as or better than the old per-arm protocols in every case.**

   | Case | WSD, one seed | Old protocol, mean of 4 |
   |---|---|---|
   | from scratch | 0.6188 | 0.6240 |
   | fine-tuned from P1 | 0.6154 | 0.6158 |
   | FT←P0 + dropout | 0.6082 (40 ep) | 0.6105 (80 ep) |
   | frozen from P0 | 0.6435 | 0.6500 |
   | frozen from P1 | 0.6307 | 0.6361 |
   | frozen from P1 + JC | 0.6357 | 0.6436 |

   The frozen cases gained most, probably from the smaller batch (12800 instead of 94208, so about 7 times more steps
   per epoch) rather than from the schedule. These runs cannot separate the two.
3. **Without dropout, from scratch / FT←P0 / FR←P0 / SC+JC peak at epochs 5–7 of the stable phase and then overfit.**
   The decay makes their val CE rise *faster*, so for them the best checkpoint is an early one at peak lr. With
   dropout (FT←P0, FR←P0) and for FT←P1, the runs still improve at the last epoch, so they profit from a longer run.
4. **The FT←P0 + dropout gain survives the search:** 0.6082–0.6101 in all six lr / backbone settings, against 0.6305
   for the best tuned no-dropout FT←P0.

**Keep it?** Yes, as the canonical training protocol: batch 12800, WSD, no early stopping, best-val checkpoint, lr per
case from the search. It is the one protocol all cases share, and it never did worse. Points to settle:
- **Epochs:** the lrs were chosen at 40 epochs. The finals at 80 epochs lengthen the stable phase, which helps only
  the cases that still improve (the dropout ones), and costs about twice the compute for the cases that peak at epoch
  5–7. Either run the finals at 40 epochs (consistent with the search, half the cost) or at 80 (as decided; better for
  the still-improving cases). Whichever is chosen, it holds for every case.
- **Edge cases:** extend the lr grid where the edge clearly wins (from scratch + dropout at 3e-3 and 6e-3; frozen
  from P0 at 3e-3), 3 runs.
- **Class weighting:** decide before the finals (section 5.3), because it changes the loss that selects the lr.
- **Optional, cheap:** for the cases that overfit early, a short decay branched from the best stable-phase checkpoint
  (a known advantage of WSD) may lower their val CE. Try it as a side test, not in the protocol.

---

## 12. Four-way test: CE vs CE + JC vs CE + dropout vs CE + JC + dropout (proposed, 2026-11-23)

Question: is the gain seen with JC in the λ scan a contrastive effect, or the regularisation that dropout gives as
well (section 8)?

**Protocol v1 (owner's decision, 2026-11-23), common to all four tests:**

| Setting | Value |
|---|---|
| Training | from scratch (all layers trained), 1 GPU per training |
| Data | `data_parallel_preprocess_251112_152051` (train 18.6M, val 1.86M, test 1.86M) |
| Batch | 12800 |
| Schedule | WSD, 80 epochs: warmup 4 epochs (5 %), constant to epoch 64, linear decay to 0 over epochs 64–80 (20 %) |
| Learning rate | 7.5e-4 |
| Loss | **class-weighted** CE (`use_class_weights=true`: b 1.5, c 1.5, light 1.0, τ 4.0, as GN2) |
| Selection | no early stopping; best-validation-CE checkpoint |
| Seeds | 4242, 42424, 424242, 4242424, the same in all four tests (paired); 42 stays out |
| Model | `rowcol_concat_v0`, chunk 512, QK-norm (as the whole SSL campaign) |

**The four tests:**

| Test | JC (weight / λ / T) | Dropout | Grad. clip | GPU mem. | Hours / run (L40S alone) | Runs per card | Runs |
|---|---|---|---|---|---|---|---|
| T1: CE | – | 0 | 1.0 | 18.3 GiB | 18 | 2 | 4 |
| T2: CE + JC | 0.3 / 0.4 / 0.1 | 0 | 1.0 | 35.2 GiB | 37 | 1 | 4 |
| T3: CE + dropout | – | 0.2 everywhere | 1.0 | 19.0 GiB | 24 | 2 | 4 |
| T4: CE + JC + dropout | 0.3 / 0.4 / 0.1 | 0.2 everywhere | 1.0 | 37.5 GiB | 53 | 1 | 4 |

- Hours are from the batch-12800 measurement (job 7126988) × 80 epochs. Sharing a card makes each CE run about 2.35 times
  slower (job 7153578). On an A40 the JC runs take about 2.2 times longer (T4 ≈ 115 h, still inside the 7-day limit).
- Total about 560 card-hours. Packing: T2 and T4 on full cards (L40S or A40), T1 and T3 two per card (L40S). That is
  4 + 2 = 6 jobs of 2 GPUs. Expected duration 2.5–5 days, depending on the cards.
- Outputs: test rejections with `arm_test_metrics` (nominal and validation-optimised fractions), plus the GN2
  same-data runs (section 5.2) as reference once evaluated.

**Points flagged before submission:**
1. **lr 7.5e-4 handicaps T3.** In the search, from scratch + dropout was strongly lr-dependent (0.6572 / 0.6472 / 0.6230
   at 3.75e-4 / 7.5e-4 / 1.5e-3). At 7.5e-4 it was 0.024 worse than at 1.5e-3, while T1, T2 and T4 are flat or best
   at 7.5e-4. A single common lr of 1.5e-3 would be about neutral for T1 and T2 (0.6207 vs 0.6188, 0.6220 vs 0.6219),
   best for T3, and worse for T4 (0.7485 vs 0.7322, a cell erratic at every lr). These are unweighted-CE, 40-epoch
   numbers.
2. **Gradient clipping at 1.0 in all four tests**, so that JC is the only difference. Before, only CE + JC from scratch
   was clipped (it collapses without).
3. **Dropout 0.2, as in the SSL campaign.** The λ-scan baseline that matched JC used 0.1.
4. **Class weights** were never used in fm4tag, so the lr search, and every earlier number, was on unweighted CE. These
   four tests form a new, self-contained set.

---

## 13. Decisions and log

**2026-11-23, decisions by the owner:**
- Fast-forward `main` to `feat/chunked-ia`.
- Delete the merged branches (after the worktree concern is addressed, section 9).
- Move the results into the `main` worktree.
- Keep the more recent venv: the two are identical (5.1), so `main/.venv` is kept and `chunk/.venv` removed.
- Leave the unmerged branches (`categorical-jet-features`, `refactor/deep-dive`) and the two stashes for now.

**2026-11-23:** this inspection was written. No changes to git, the worktrees or the results yet. Waiting for approval
of the plan before acting.

**2026-11-23 (later):**
- Checked: no fm4tag run ever used class-weighted CE (5.3).
- Reported on the P_drop pretrainings (section 10) and on the lr search (section 11).
- The owner will do the cleanup steps, then run the four-way JC / dropout test (section 8).
- To decide: class weighting, and 40 vs 80 epochs for the finals.

**2026-11-23 (owner's decisions for protocol v1):**
- Batch 12800, WSD, no early stopping, best-validation checkpoint.
- lr 7.5e-4.
- 80 epochs with 5 % warmup and 20 % decay.
- Class-weighted CE in fm4tag.
- Tests on 1 GPU each, several per card when the memory allows.
- The four-way test design is in section 12, waiting for the owner's OK on the flagged points.

**2026-11-23, owner's answers on the flagged points:** lr **1e-3** for all four tests (backbone at lr / 10 whenever a
run has pretrained parts; the four from-scratch tests have none); gradient clipping 1.0 in all four; dropout 0.2;
cleanup before submitting.

**2026-11-23, cleanup done** (commits `589cef0` … `2444670` on `main`, nothing pushed):
1. The uncommitted `roc_comparison.py` change was set aside, `main` fast-forwarded to `feat/chunked-ia` (126 commits),
   and the change re-applied (clean automatic merge, lint and 21 related tests pass) and committed.
2. Committed: this report, `docs/ssl_campaign_report.md`, `docs/ssl_tests_plan.pptx` (+ preview), `paper/`,
   `plot_paths.txt`. `docs/slides/` (31 MB of generated decks) moved here but ignored.
3. Results moved from `fm4tag-chunk` (and the S0 profiles from `fm4tag-chunk-s0`) to the **same relative paths** in
   this worktree: `slurm/ssl_campaign` (101 GB), `plots/<7 campaign dirs>`, `slurm/{arm_roc,plot_logs,smoke}`,
   `slurm/profiling/*`. All moves were renames on the same filesystem. The 7 absolute cache symlinks in `plots/` were
   re-pointed. In 44 tracked files the path prefix `fm4tag-chunk(-s0)` was replaced by `fm4tag`; all 170 distinct paths
   were checked to exist. A launcher dry run from `main` composes the cells and finds the pretraining checkpoints.
   *This keeps the existing layout instead of the `results/<campaign>/` regrouping proposed in section 9.* That
   regrouping would have required changing every launcher and all 311 paths. The index
   `docs/experiments/README.md` gives the same "what is in which folder" view.
4. Tags created: `archive/<branch>` for every deleted branch, `archive/s0-snapshot`, `exp/lambda-scan-final`
   (`d858d71`), `exp/ssl-wsd-lr-search` (`f713105`).
5. Worktrees `fm4tag-old`, `fm4tag-chunk-s0`, `fm4tag-chunk` removed (only caches and the duplicate `.venv` were left
   in them). Local branches deleted: the 9 merged ones and `feat/chunked-ia`. Kept: `categorical-jet-features`,
   `refactor/deep-dive`, both stashes.
6. `/storage3/DSIP/rriva/research/fm4tag-chunk` is now a **symlink to `fm4tag`**, so the old paths recorded inside run
   folders (`.hydra/config.yaml` with `encoder_ckpt`, caches, wandb) still resolve.
7. Not done (needs the owner): pushing `main` (130 commits ahead of `origin/main`) and deleting the 6 stale branches on
   `origin`.

**2026-11-23, four-way test submitted** (`scripts/configs/campaign/T4way.cells`):
- **Checks before submitting:** every cell composes with the agreed settings; dropout is 0.2 everywhere in t3/t4 and
  0 everywhere in t1/t2 (head included); class weights [1.5, 1.5, 1.0, 4.0]; 2 new tests that the weights enter the
  CE; 292 tests pass.
- **Stages:** `T4way_full_20261123_164524` (t2, t4; jobs 7385094-97, full cards, L40S or A40) and
  `T4way_shared_20261123_164528` (t1, t3; jobs 7385098-99, two per L40S card).
- **Expected:** CE cells about 2-2.5 days, JC cells 1.5-5.5 days depending on the card type.


**2026-11-23 (evening), experiments reorganised into `experiments/<campaign>/`** (owner's request):
- **Layout:** every experiment has its own folder with `README.md` (question, data, model, training methods, stages,
  results, related experiments), `configs/` (cells and analysis configs), `slurm/` (its own launchers), `notes/` /
  `diagnostics/` where present, and the git-ignored `outputs/` and `plots/`.
- **The 8 folders:** `2026-03_early-runs`, `2026-07_ce-jc-grids`, `2026-07_kernel-profiling`, `2026-08_lambda-scan`,
  `2026-10_ssl-campaign`, `2026-11_dropout`, `2026-11_wsd-lr-search`, `2026-11_four-way-jc-dropout`. Plus `_template/`
  and the index `experiments/README.md` (which replaces `docs/experiments/README.md`).
- **How it was done:** `scripts/maintenance/restructure_experiments_2026-11-23.py` moved 131 tracked files (git mv)
  and 201 ignored folders and files (renames on the same filesystem), then rewrote the paths in 184 tracked files.
  1004 path mentions were checked; every concrete path exists except three old plotting-script defaults that already
  pointed at a missing folder.
- **Launchers:**
  - `slurm/ssl_pack.sh` and `slurm/dropout_batch_sweep.sh` put new stages and sweeps into the `outputs/` of the
    experiment whose `configs/` holds the cells file (or `EXPERIMENT=`). The config-driven `scripts/submit_*.sh` put
    their results into that experiment's `plots/`. All three use `slurm/experiment_of.sh`.
  - Campaign-specific launchers moved into their experiment's `slurm/`. Generic tools (`arm_roc_job.sh`,
    `arm_test_metrics_job.sh`, `cpu_pytest.sh`, `gpu_pytest.sh`, `gpu_health.sh`) are now tracked in `slurm/`, with a
    default log folder `slurm/logs/`.
- **Old paths:** `slurm/ssl_campaign/<stage>` are symlinks to the moved stage folders, and `fm4tag-chunk` is a
  symlink to the repo. Old run records (`encoder_ckpt` in `.hydra/config.yaml`) keep resolving.
- **Four-way test:** the first submission (jobs 7385094–99) was cancelled before it started. It is relaunched from
  `experiments/2026-11_four-way-jc-dropout/` with 6 seeds per cell, and head dropout 0 in the no-dropout cells is
  confirmed.

**2026-11-23, owner's decisions on the pretrained arms:**
- Fine-tuned and frozen runs of protocol v1 use **one pretraining seed per pretraining type**: the one with the lowest
  pretraining validation loss, chosen before any downstream result. That is seed 42 for both P0 (5.4420 vs 5.5054)
  and P1 (5.6664 vs 5.7292). It runs with 6 training seeds.
- Results with frozen encoders are therefore conditional on that pretraining, since the pretraining seed dominates
  their spread.
- Downstream keeps loading the **last** pretraining checkpoint (epoch 30). For P0 and P1 it is at most 0.0015 from the
  best pretraining validation loss. It would matter only for the early-stopped dropout pretraining P0 seed 42.
