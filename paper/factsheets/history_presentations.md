# Fact sheet: project history, presentations, motivation, bibliography, figures

Area key: `history_presentations`. Compiled 2026-11-17. Read-only pass over the research diary,
the repos (fm4tag, fm4tag-old, fm4tag-chunk, salt-ririva, packages/salt-pt, embedding2tagging,
tabicl, jetset2-open-data, datasets/transforming-jet-flavor), memory notes and chat transcripts.

Provenance tags: **[F]** verified in files, **[M]** from a memory note, **[C]** from a chat transcript,
**[W]** from a web page (fetched today), **[I]** inferred.

---

## 0. What this area contributes to the paper

- The Introduction story. The project began in March 2025 when a supervisor suggested
  *tabular foundation models*. It then ran through SAINT pretraining (2025), a Lightning
  rewrite (Feb 2026), a GN2-like aggregator plus jet-embedding contrastive loss (Jun 2026), the
  ISAB/tabicl redesign (Jul 2026), the multi-task λ scan (Jul–Sep 2026), the port to salt for
  GN2 (Sep–Oct 2026) and the SC/FT/FR SSL campaign with chunked intersample attention and
  QK-norm (Oct–Nov 2026).
- The motivations written in the diary: jets as tabular data, intersample attention, contrastive
  learning, InfoMin and mutual information, and robustness to data/MC and MC/MC (from the Xbb
  AQP: MC-to-MC maps, a GN2Xv01 campaign bias).
- Prior presentations: the SIF 2026 talk is verified, IFAE 2026 has a date but no content on
  disk, a poster was made in Nov 2026, and there are internal decks.
- A BibTeX inventory with reusable keys, and the references that are cited but missing from it.
- Candidate figures. The two that matter most for the requested **P0/P1 × SC/FT/FR table**
  already exist as slides.

---

## 1. Dated timeline

| Date | Event | Source |
|---|---|---|
| 2024-06-29 / 07-04 | Diary started. References of supervisors M. Cristoforetti and R. Iuppa (FBK DSIP + Univ. Trento). | [F] diary/days/day240629.tex, day240704.tex |
| 2024-10 → 2025-02 | Self-study: D2L, attention/RNNs, DL1 thesis (Guth), ATLAS tracking, PyTorch (MLHEP2023). | [F] day241028–day250214 |
| 2025-02-21 → 03-06 | Learning salt/GN2. Salt tutorial running on the FBK cluster on 03-05. | [F] day250221, day250304–06 |
| 2025-03-18 | No clear direction yet. The first idea is hit-level information / trackless b-tagging. | [F] day250318 |
| 2025-03-26/27 | Supervisor meeting. Topics: foundation models (Vigl, Golling MPM), GNNs. **Marco suggests "Tabular Foundation Models"** and "how tabular data can be embedded. Only embedding without a classification task." | [F] day250326, day250327 |
| 2025-03-31 | ATLAS ML Workshop 2025. | [F] day250331 |
| 2025-04-09 | Weekly checkpoint talk "Transformers for tabular data" (Google Slides, external). | [F] day250409 |
| 2025-04-10/13 | **SAINT applied to track data.** h5 jets+tracks unified into GN2-like track inputs. A 270M-jet rucio dataset is the target. | [F] day250410, day250413 |
| 2025-04-16 | Plan: first use SAINT with frozen parameters, then fine-tuning. This is the origin of the FR/FT idea. | [F] day250416 |
| 2025-05-11/14 | SAINT pretraining pipeline. NCE-type contrastive loss (SAINT repo). Loss = contrastive λ0 + denoising-continuous λ2 + denoising-categorical λ3. lr 1e-5 needed (1e-3/1e-4 stuck at ~16). | [F] day250511–14 |
| 2025-05-15/20 | Own Pythia8+Delphes ttbar generation (4M events). Abandoned later in favour of ATLAS open data. | [F] day250515, day250520 |
| 2025-06 | AQP (ATLAS qualification task) on Xbb MC-to-MC efficiency maps. Literature includes ATL-PHYS-PUB-2020-009 (MC-MC SFs). | [F] day250610 |
| 2025-06-11/12 | SAINT + classifier working on the FTAG dataset (sum over particles + FFNN, batch size 1 issue). | [F] day250611–12 |
| 2025-06-16/18 | Set functions: DeepSets, Set Transformer, ParticleNet, EFN. TabDeco paper. | [F] day250616, day250618 |
| 2025-06-23 | The ATLAS Open Data (JetSet) sample is suggested by the AQP contact. | [F] day250623 |
| 2025-07/08 | "tagger_pt" pipeline (balanced datasets, BatchSampler). Pretraining 47M–170M params (e.s. 32/64, 120k jets). | [F] day250702–day250801 |
| 2025-09-01/03 | Strategy: large datasets, high dropout and noise. New pretraining loss weights λ0=1, λ1=5, λ2=2. Pretraining runs with e.s. 16–128 on the 30M-jet FTAG dataset. From-scratch with noise 0.14/0.1. | [F] day250901, day250903 |
| 2025-09-09 | Training time grows with batch size because of SAINT intersample attention. | [F] day250909 |
| 2025-09-16 | ROC light/τ inversion explained by the balanced dataset (vs GN2's unbalanced one). | [F] day250916 |
| 2025-10-13/15 | UPP preprocessing of slices (datasets 251013). First **GN2 salt run** (21M jets). SAINT pretraining on 26/27M. Attention types row/col/colrow/concat compared. | [F] day251013–15 |
| 2025-11-12 | Datasets `251112_112022` (train 52.00M / val 5.00M / test 5.00M) and `251112_152051` (pretrain 39.68M / train 18.60M / val 1.86M / test 1.86M) created. | [F] h5 metadata (this session), dir timestamps |
| 2025-12-01/08 | salt-pt fork: GN2 open-data runs, then "run with only jet classification task" (GN2 only-JT reference). | [F] packages/salt-pt git log |
| 2025-12-15 | Supervisor meeting. Topics: separate jet/track encoders; **test stability of a track's representation vs batch composition** ("SAINT does not ensure that the track representation is unique during inference"); multi-jet vs single-jet IA; move to Lightning. | [F] day251219 |
| 2026-02-20 | **fm4tag** repo created ("checkpoint from cern gitlab repo", Lightning). 02-24 encoder per object. 03-09 Optuna. 03-13→27 uniformity and effective rank. | [F] fm4tag git log |
| 2026-03-11 → 04-01 | col transformer pretrain/scratch tests. Pretraining run 65 ep (`run_20260320_112055`). 04-01: SC, FT (unfrozen), FT (frozen until ep 10), frozen runs. **DDP bug**: 2/4-GPU jobs used 1 GPU. IA × DDP issue: model behaviour depends on the number of devices. | [F] day260311, day260401 |
| 2026-04 (8–10 Apr?) | **IFAE 2026** (22nd edition, Bologna). Contribution content not on disk. | [W] web search, unconfirmed; poster abstract [C] |
| 2026-05-06 | fm4tag remake: repo structure, augmentation management (multi-view). | [F] day260506, git 74c11e1 |
| 2026-05-29 | Related work found: RS3L, "SSL strategies for jet physics" (arXiv:2503.11632), JetCLR. | [F] day260529 |
| 2026-06-10 | "Contrastive loss is sensitive to negative pair counts." | [F] day260610 |
| 2026-06-16 | **GN2-like `JetAggregator` (attention pooling) + composable loss; jet contrastive (JC) enabled.** GlobalTransformerEncoder for jet features. | [F] git 32cd810, c251cc9, 0694a47 |
| 2026-07-01 | Chunked row attention is not permutation-equivariant, so the fix is ISAB (Set Transformer). Timing table: chunked c=64 vs full is 3.6× faster at b=256 and 72.8× at b=8192 (numpy, 1 core, d=128). | [F] day260701 |
| 2026-07-06/08 | Full code review. Fixes: aggregator padding leak (b2821de), RowAttention mask (f0b28b9), DDP (5785e12). **λ inverted to a noise level (4fb18ec)**: the old `lam`=fraction kept had made view0 ~99% other jets, so contrastive collapsed (AUROC 0.89 → 0.53). DDP-safe staged unfreeze (a29388f). | [F] git; [M] fm4tag-review-session-state |
| 2026-07-09 | "Model is huge (24M params) but worse than GN2 (2M)". rowcol pretrainings died. **"A new set of runs should be made by varying the noise parameter"**: the origin of the λ scan. | [F] day260709 |
| 2026-07-13/14 | tabicl port: FlashAttention-2, **per-token ISAB replaces chunked row attention**, pre-norm residual fix (SAINT bug `fn(norm(x))+norm(x)`), final LayerNorm. Tracks backbone 2.75M → 0.76M params, model 5.45M → 3.3M. Sweeps switched to 2-GPU DDP. | [F] git 056d75e, 0e59382, 0c2a00e, 365b633; [M] tabicl note |
| 2026-07-14 → 08-11 | First CE+JC grids (fm4tag `slurm/classify_from_scratch/`): lr × jc (07-14, 07-20, 08-03/04), concat jc × λ{0.03,0.1,0.3} (08-07), jc=1 λ ∈ {0.15…0.9} (08-11). | [F] run dir names |
| 2026-07-20 | dim 128 → 16, num_inds back to 16: 9.7M → 2.3M params. | [F] git e9a7ba7; [M] |
| 2026-07-22 | `row_mode: per_token \| concat` (5532eb5). On 08-17 concat measured 11–23% faster (bs 1024–2048) but is a different model. | [F] git; [M] |
| 2026-08-18 | "Beat-GN2" Phase 0 (single seeds): per_token no-JC c-rej@77% 15.25 vs salt GN2 only-JT 15.10. JC hurt at that point. Concat sweep stopped. | [M] fm4tag-hpo-campaign |
| 2026-08-19 → 09-22 | λ scan with seeds (jc 0.3, per_token, bs 1024×2, lr 3e-4, τ 0.1). High-λ reseeds at 100 epochs. bs2048 1-GPU ladder (09-03). New seeds λ 0.45/0.55/0.64/0.7 (09-22). Interim result 08-29: λ ≥ 0.25 beats salt GN2 c-rej, λ 0.40 best (provisional, n=1). | [M] fm4tag-hpo-campaign; [F] run dirs |
| 2026-09-11 | **SIF 2026 talk** "Foundation models for robust jet flavour tagging for the ATLAS experiment", Section 1. | [W] 2026.congresso.sif.it/talk/303 |
| 2026-09 | Paper plan v0 written (plan_v0.tex, dated "September 2026"). | [F] fm4tag/paper/plan_v0.tex |
| 2026-09-28/29 | **Port to salt** (`feat/ssl-jet-contrastive`): encode()/run_tasks split, Augmentation hooks, ContrastiveLoss (SAINT structure, τ 0.1), phases pretrain/finetune/scratch, Optuna. Scan 1: CE + 0.3 CL, λ 0.2–0.7, 2 GPU × 2048, unweighted CE, 52M jets. | [F] day260929; salt git 1e6eaea, 690fa75 |
| 2026-09-30 | fm4tag λ-scan runs all finished. Definitive plots and fraction search requested (floors 1.0/1.0 on 10-01). Supervisor meeting prep (backbone/head split in salt, MI framing). | [C] d2e47d07 L732, L939; [F] day260930 |
| 2026-10-01 → 10-07 | salt scan 2 (w 1.0, LR 5e-4, 3000/GPU, weighted CE, flash): 14/16 contrastive seeds diverge. Calibration: val-CE rise is mostly real overfit. GLS HPOs. | [F] day261007 |
| 2026-10-07 | salt scan 1 final (41 seeds, 0 diverged): neutral at λ ≤ 0.4, worse at λ ≥ 0.5. | [M] salt-ssl-infomin-project |
| 2026-10-09 → 10-14 | GLS dropped; wsum HPOs. The apparent GLS gains were the LR alone. 10-14: new λ scan (2 GPU × 3000, LR 1e-3, w 0.2, τ 0.5, ES 20, `run_20261014_153400`). | [M] salt-ssl-infomin-project |
| 2026-10-12 | fm4tag final λ-scan ROC configs and plots committed (3ab272f, a14c0e7, 1e778f9). | [F] git |
| 2026-10-15 | **SSL campaign starts** (`fm4tag-chunk`, branch `feat/chunked-ia`): ISAB deleted, all-pairs row attention within **jet chunks** via flash varlen, SupCon within chunks, aggregator on packed tracks. Same day: **embedding2tagging** restart on JetSet2. | [F] git c1e28af, 6bf0617, d1fcdfe; e2t DIARY |
| 2026-10-16 | SSL test plan table (pptx) + supervisor report md. | [F] fm4tag-chunk/docs/ssl_tests_plan.pptx, ssl_campaign_report.md |
| 2026-10-17/19 | S1: per_token vs concat × chunk 128/512 are equal within noise (0.620–0.621), concat 1.45× faster. Chosen: concat, chunk 512, 40 epochs. **P2 added.** | [M]; [F] git 99b67d4, e4604d7 |
| 2026-10-19/21 | S2 T × λ grid judged by FR probes. Recipes P0 = P1 = P2 = T 0.1, λ 0.4. Gradient clipping 1.0 in pretraining. | [M]; [F] git 2560e63, 68b53bd, 08c0525 |
| 2026-10-22 | Cause of pretraining blow-ups: row-attention logit growth (max q·k/√d up to 1e5–1e7). Fix: **QK-norm (RMSNorm) in row attention**. | [M]; [F] git 9ec978d, 455cf7d |
| 2026-10-23 → 10-30 | Full 30-ep P0/P1/P2 pretrainings, S3 LR per arm, final runs. Result: SC 0.6240±0.0029 < FT+JC←P1 0.6288 < FT←P0 0.6312 < FR+JC←P1 0.6436 < FR←P0 0.6500 (best val CE, 4 runs). SC+JC 3/4 collapsed. | [M] fm4tag-ssl-campaign |
| 2026-11-03 → 11-06 | Presentation-style summary; label-fraction scan (P0 never beats SC at 1–100%); report deck (65 slides); GN2 CE-only on the 18.6M slice in salt. | [M]; [C] 88778f66 |
| 2026-11-09 | User asks about making a vertical A0 research poster. | [C] 88778f66 L13269 |
| 2026-11-11/12 | Controls: SC+JC clip1 = SC (0.6241 vs 0.6240). **FT←P1 CE-only 0.6158±0.0002 beats SC.** Test c-rej@77% optimised: FT←P1 10.8 vs SC 9.7 vs GN2v01 15.6. | [M] fm4tag-ssl-campaign |
| 2026-11-13 → 16 | Dropout 0.2 downstream: FT←P0+drop best (c-rej 12.1, ×1.248 SC), all JC+drop fail. P_drop pretrainings (dropout 0.1) launched 11-16. The user doubts these may be hyper-parameter artefacts. | [M] fm4tag-ssl-campaign |
| 2026-11-17 | Paper writing starts. | this task |

---

## 2. Presentations and the claims made there

| When | What | Content / claims | Provenance |
|---|---|---|---|
| 2025-03-26 | Internal checkpoint | Trackless b-tagging group work | [F] day250326 |
| 2025-04-09 | Internal checkpoint, Google Slides | Transformers for tabular data | [F] day250409 (link only) |
| 2025-11 | ATLAS Xbb weekly meeting (AQP) | MC-MC efficiency maps | [F] day260108 mentions "my talk at the weekly meeting of november" |
| **Apr 2026** | **IFAE 2026** (22nd ed., Bologna, 8–10 Apr per web) | **No material on disk.** The poster abstract says earlier stages were presented there. At that time the available results were the first fm4tag pretraining (65 ep, `run_20260320_112055`) plus SC/FT/FT-delayed/frozen downstream runs (launched 2026-04-01). `fm4tag/notebooks/effective_rank.png` and `uniformity.png` (2026-03-31) are probably from that period. | [W] date; [I] content |
| **2026-09-11** | **SIF 2026** (Congresso SIF, Section 1 Nuclear and Subnuclear Physics, Fri 11 Sep 2026, Edificio D6 Aula 118). Talk **"Foundation models for robust jet flavour tagging for the ATLAS experiment"**. Authors R. Riva, M. Cristoforetti, R. Iuppa, F.M. Follega. | Abstract claims: (i) a heavy-flavour ID challenge; (ii) the latent representation matters and the **MC generator choice can shift it**; (iii) embeddings **robust to changes in sample structure**; (iv) **tabular foundation models** on ATLAS Open Data with explicit jet-level representations; (v) **two-phase training: contrastive pre-training, then fine-tuning**; (vi) evaluation with dedicated embedding metrics and **performance across different samples**. | [W] https://2026.congresso.sif.it/talk/303 |
| 2026-10-01 | Supervisor meeting (paper draft update) | Backbone/head split in salt plus a contrastive loss on the global jet embedding. GN2 λ scan (CE + 0.3 CL, λ 0.2–0.7, 52M jets). MI / InfoMin framing. | [F] day260930 |
| 2026-10-16 | Supervisor md + 1-slide plan table | P0/P1 × FT/SC/FR, with and without JC; ~850 GPU-h estimate | [F] fm4tag-chunk/docs/ssl_campaign_report.md, ssl_tests_plan.pptx |
| 2026-10-20 → 11-05 | Living deck `ssl_campaign.pptx` (32 slides) | SSL campaign decisions and results | [F] fm4tag-chunk/docs/slides/ssl_campaign/ |
| 2026-11-06 → 11-12 | Report deck `ssl_campaign_plots.pptx` (78 slides) for the supervisors | Pretraining curves, 8 training methods, ROC vs GN2v01, label scan, pT backups | [F] fm4tag-chunk/docs/slides/report/ |
| Nov 2026 | **Poster** (conference not named; abstract pasted by the user) | SAINT-like encoder with per-feature embedding, self-attention over features, and **IA across all jets in a batch and their tracks**. **InfoNCE-type contrastive loss + denoising reconstruction loss**. Two strategies: SSL pretraining → fine-tuning, and multi-task. JetSet ~190M jets, tt̄, 13.6 TeV. Compared on classification and embedding metrics. Next steps: robustness to data/MC and MC/MC, and comparison with RS3L (Harris et al.). | [C] current session |

---

## 3. Motivation ideas in the diary and planned studies not yet done

**Motivations (as written in the diary):**
- **Jets as tabular data.** "Correlations between different features is not obvious. This seems
  to be something really new" (2026-07-06). Per-feature embedding allows inter-feature (column)
  attention (plan_v0 footnote). Origin: Marco's tabular-FM suggestion (2025-03-26/27, 2026-07-06,
  TabICLv2 paper arXiv:2602.11139).
- **Intersample attention.** A jet's or track's representation can draw on others (SAINT row
  attention). The diary records the problems: batch-size-dependent cost (2025-09-09), DDP and
  world-size dependence (2026-04-01, 2026-07-08), non-equivariance of chunking (2026-07-01),
  instability at large batches (2026-07-10), and inference-time dependence on batch-mates
  (2025-12-15).
- **Contrastive learning.** Used either to build in symmetries and invariances or for
  resampling. "At the moment I cannot use it for domain adaptation" (2026-07-06).
  "Contrastive loss is sensitive to negative pair counts" (2026-06-10).
- **InfoMin / mutual information** (2026-09-29/30, 10-07). I(v1;v2) ≥ I_NCE = log K − L_NCE
  (van den Oord 2018). InfoMin (Tian et al. 2020) predicts a **reverse-U** of downstream
  performance against view-shared information. The λ scan as an InfoMin probe. Caveat written
  by the author: with CE in the loss, the scan "replicates fm4tag rather than testing InfoMin".
  The proper test is pretraining → **linear probe**.
- **Robustness to data/MC and MC/MC.** From the AQP: MC-to-MC scale factors (ATL-PHYS-PUB-2020-009,
  2021-035) and a **GN2Xv01 MC-campaign bias**, where the tagger learned generator/campaign
  tracking differences (2025-11-05). In the SIF abstract and as the poster's next step.
- **Embedding quality monitored explicitly.** Uniformity (Wang & Isola 2020) and effective rank
  (fm4tag since 2026-03). CLS-embedding component sharpness (2025-06-23, 2025-09-01).

**Planned but not done (as of 2026-11-17):**
1. InfoMin test: contrastive-only pretraining → linear probe, downstream performance vs I_NCE (reverse-U?). [F] day260929/day261007. Not done in salt. The fm4tag S2 FR probes over T × λ are a partial proxy. [I]
2. Jet-specific physics augmentations: track dropout, IP smearing in σ, η–φ rotation, pile-up injection. [F] day260929. Implemented in embedding2tagging on 2026-11-04/05 (ConstituentDropout, ConstituentSplit, Smear), not yet trained. [F] e2t DIARY
3. Robustness study (data/MC, MC/MC, different samples): promised in the SIF abstract and on the poster. Not run. [F]/[C]
4. Comparison with RS3L / JetCLR-like contrastive methods. [C] poster; [F] day260529
5. Pretraining → FT/FR for **GN2 in salt**: infrastructure exists (`model.phase`), only from-scratch scans run. Track-level contrastive in salt deferred. [M] salt note; plan_v0 lists it
6. Stability of a track/jet representation vs batch composition at inference (re-chunking test). [F] day251219; e2t plan
7. Physics-aware/domain-aware positive pairs and synthetic "Mismodel" domains. [F] e2t DIARY (kick-off)
8. FT/FR←P1 CE-only with dropout; P1 at low label fractions; downstream from the P_drop checkpoints. [M]
9. Automatic loss balancing (2026-07-13 idea); plateau LR schedule (salt future_developments.md). [F]

---

## 4. BibTeX inventory

### 4a. `/storage3/DSIP/rriva/research/diary/diary.bib` (35 entries; relevant ones)
| key | title | id |
|---|---|---|
| somepalli2021saint | SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training | arXiv:2106.01342 |
| tian2020makes | What Makes for Good Views for Contrastive Learning? (NeurIPS 33) | arXiv:2005.10243 |
| oord2018representation | Representation Learning with Contrastive Predictive Coding | arXiv:1807.03748 |
| pmlr-v9-gutmann10a | Noise-contrastive estimation… (AISTATS 2010, PMLR 9:297–304) | proceedings.mlr.press/v9/gutmann10a |
| chen2024tabdecocomprehensivecontrastiveframework | TabDeco: contrastive framework for decoupled representations in tabular data | arXiv:2411.11148 |
| lee2019settransformerframeworkattentionbased | Set Transformer | arXiv:1810.00825 |
| zaheer2018deepsets | Deep Sets | arXiv:1703.06114 (year should be 2017) |
| Qu_2020 | Jet tagging via particle clouds (PRD 101, 056019) | doi:10.1103/PhysRevD.101.056019 |
| Komiske_2019 | Energy flow networks: deep sets for particle jets (JHEP 01 (2019) 121) | doi:10.1007/JHEP01(2019)121 |
| golling2024masked | Masked particle modeling on sets (MLST 5, 035074) | (arXiv:2401.13537 [I]) |
| vigl2024finetuning | Finetuning foundation models for joint analysis optimization in HEP (MLST 5, 025075) | (arXiv:2401.13536 [I]) |
| shlomi2020graph | Graph neural networks in particle physics (MLST 2, 021001) | — |
| battaglia2018relationalinductivebiasesdeep | Relational inductive biases, deep learning, and graph networks | arXiv:1806.01261 |
| vaswani2023attentionneed | Attention Is All You Need | arXiv:1706.03762 (year should be 2017) |
| atlas2023transformer | Transformer NNs for boosted H→bb/cc in ATLAS (= ATL-PHYS-PUB-2023-021, duplicates salt `GN2X`) | — |
| atlas2020monte | Monte Carlo to Monte Carlo scale factors for flavour tagging efficiency calibration (ATL-PHYS-PUB-2020-009) | — |
| atlas2021efficiency | Efficiency corrections for a boosted Hbb tagger (ATL-PHYS-PUB-2021-035) | — |
| Brianti:2023tut | Pipeline for performance evaluation of flavour tagging dedicated GNN algorithms (JINST 19 C02064) | arXiv:2312.06245, doi:10.1088/1748-0221/19/02/C02064 |
| DiLuca:2023byz | Automated feature selection procedure for particle jet classification (NPB 990, 116182) | doi:10.1016/j.nuclphysb.2023.116182 |
| guth2021search | PhD thesis (Guth): ttH(bb) and DL flavour tagging | — |
| aaboud2017performance | Performance of ATLAS track reconstruction in dense environments (EPJC 77) | — |
| cms2017particle, salam2010towards, murphy2022/2023probabilistic, Cristoforetti:2017naf, Rinnert:2019vxw, ju2020graph, farrell2018novel, di2022deep, aakesson2006atlas, salzburger2007atlas, badgerlectures | background / not central | — |

### 4b. `/storage3/DSIP/rriva/research/salt-ririva/paper/paper.bib` (Salt JOSS paper bib; identical in salt-ririva-dev)
| key | title | id |
|---|---|---|
| GN2 | Transforming jet flavour tagging at ATLAS | arXiv:2505.19689 |
| GN1 | Graph Neural Network Jet Flavour Tagging with the ATLAS Detector | ATL-PHYS-PUB-2022-027 (cds 2811135) |
| GN2X | Transformer NNs for identifying boosted H→bb/cc in ATLAS | ATL-PHYS-PUB-2023-021 (cds 2866601) |
| ATLAS:2019 | ATLAS flavour-tagging algorithms for the LHC Run 2 pp dataset | arXiv:2211.16345, CERN-EP-2022-226 |
| Qu:2020 | Jet tagging via particle clouds | doi:10.1103/physrevd.101.056019 |
| Qu:2022 | Particle Transformer for Jet Tagging (ICML 2022, PMLR 162) | — |
| Bols:2020 | Jet flavour classification using DeepJet (JINST 15 P12012) | doi:10.1088/1748-0221/15/12/p12012 |
| Cagnotta:2022 | ML applications for jet tagging in CMS | doi:10.3390/app122010574 |
| JetClass:2022 | JetClass dataset | doi:10.5281/ZENODO.6619768 |
| Zaheer:2017 | Deep Sets (NIPS'17) | — |
| 2017arXiv170603762V | Attention Is All You Need | arXiv:1706.03762 |
| Guest:2018 | Deep Learning and Its Application to LHC Physics | doi:10.1146/annurev-nucl-101917-021019 |
| ATLAS:2008 / Evans:2008 | ATLAS experiment / LHC machine (JINST 3) | doi:10.1088/1748-0221/3/08/S08003 / S08001 |
| Kingma:2015 | Adam | arXiv:1412.6980 |
| pytorch / lightning | PyTorch (NeurIPS 2019) / PyTorch Lightning | doi:10.5281/zenodo.3828935 |
| Birk:2023 | umami-hep/puma v0.2.4 | doi:10.5281/ZENODO.7806395 |
| umami | Umami toolkit | github |
| LeCun:2015, NIPS:2017 (SHAP), hdf5:2023, onnx, … | software/background | — |

### 4c. Elsewhere in files
- `salt2025`: Salt, JOSS 10(112) 2025, doi:10.21105/joss.07217. [F] salt-ririva/README.md
- puma JOSS paper doi:10.21105/joss.05833 (as attributed by the JetSet2 README). [F]
- JetSet v1: CERN Open Data record 93940. Companion paper arXiv:2505.19689. [F] transforming-jet-flavor/README.md
- JetSet2 + "Carpe Datum" scaling note ATL-SOFT-PUB-2026-002 (cds 2953659). Vigl et al. arXiv:2610.06784 "How to scale your HEP ML models". [F] jetset2-open-data/README.md; embedding2tagging/DIARY.md
- TabICL arXiv:2502.05564 (ICML 2025); TabICLv2 arXiv:2602.11139 (ICML 2026). [F] tabicl/README.md

### 4d. Cited but with NO BibTeX entry yet (to add)
- CMS-DP-2025-081 (poster ref [2]).
- JetSet dataset doi:10.7483/OPENDATA.ATLAS.QG8W.TO8P (poster ref [4]).
- Harris et al., RS3L, arXiv:2403.07066 (poster ref [7]; diary 2026-05-29 links zenodo 10633815).
- JetCLR (Dillon et al., arXiv:2108.04253 [I]).
- "Self-Supervised Learning Strategies for Jet Physics" arXiv:2503.11632 (diary 2026-05-29).
- SPANet (Shmakov et al., SciPost Phys. 12, 178 (2022)) (diary 2026-07-01).
- Wang & Isola 2020, uniformity/alignment (arXiv:2005.10242 [I]).
- Effective rank: Roy & Vetterli 2007 [I].
- SimCLR (arXiv:2002.05709 [I]) and SupCon (Khosla et al., arXiv:2004.11362 [I]): fm4tag uses `MultiViewSupConLoss`.
- QK-norm (Henry et al. 2020, arXiv:2010.04245 [I]; ViT-22B arXiv:2302.05442 [I]).
- FlashAttention-2 (arXiv:2307.08691 [I]).
- IFAE 2026 and SIF 2026 talk (poster refs [5], [6]).
- Duplicates to merge: diary `Qu_2020` = salt `Qu:2020`; `zaheer2018deepsets` = `Zaheer:2017`; `vaswani2023attentionneed` = `2017arXiv170603762V`; `atlas2023transformer` = `GN2X`.

---

## 5. Candidate figures (absolute paths)

| Path | Shows | Paper use |
|---|---|---|
| /storage3/DSIP/rriva/research/diary/figures/gn2.png | GN2 schematic (from the ATLAS GN2 paper), annotated in red: "Global jet embedding: here we can apply the contrastive loss" and "In my framework I can apply also the contrastive loss here" (initial track representation). Also referenced by plan_v0.tex as `images/gn2.png`. | Model section: where JC acts. Redraw, or cite as "adapted from [GN2]"; the red annotations are not publication style. |
| (missing) `images/fm4tag.png` | Referenced by plan_v0.tex as "Our model". **Not found on disk.** | Must be drawn. Basis: the fm4tag README ASCII diagram and docs/encoder_dataflow.md. |
| /storage3/DSIP/rriva/research/fm4tag/README.md (ASCII) and /storage3/DSIP/rriva/research/fm4tag/docs/encoder_dataflow.md | Per-object encoders → TransformerAggregator → MLP head. Per-feature tokens, col/row/rowcol backbone, projector/denoising heads. | Blueprint for the architecture figure. |
| /storage3/DSIP/rriva/research/fm4tag/docs/ssl_tests_plan_preview.jpg | **Table P0/P1 × {FT, SC, FR} × {no JC, JC}** with columns Starts from / Trains / Feature CL / JC / CE (2026-10-16; P2 absent). | **Starting point for the requested training-strategies table.** |
| /storage3/DSIP/rriva/research/fm4tag/docs/slides/report/preview/slide_02.png | "The two pretrainings": P0 vs P1 settings. 7.4M params (P0 5.4M trained), T 0.1, λ 0.4, chunk 512, QK-norm, B 12,288, lr 1e-3, 30 ep, clip 1, 39.68M jets. | Pretraining settings table. |
| /storage3/DSIP/rriva/research/fm4tag/docs/slides/report/preview/slide_03.png | "The eight training methods": SC, FT(P0), FR(P0), FT(P1), FR(P1), SC+JC, FT(P1)+JC, FR(P1)+JC. Seeds, init, trained parts, params, batch, lr, clipping (2026-11-12). | **Near-final version of the training-method table.** |
| /storage3/DSIP/rriva/research/fm4tag/docs/slides/report/ssl_campaign_plots.pptx (78 slides; previews in .../preview/slide_NN.png) | Pretraining curves (5–10), CE/eff-rank/uniformity per method (12–14, 20–22), ROC vs SC/GN2v01 (15–18, 23–26), rejection summaries (32–37), label scan (39–47), pT backups (49–78). | Results figures (SSL pillar). |
| /storage3/DSIP/rriva/research/fm4tag/notebooks/effective_rank.png, uniformity.png (2026-03-31) | Early pretraining: loss + val track/jet effective rank over ~65 epochs. Spike at ep ~64. | Historical / probably IFAE-era; not for the paper. |
| /storage3/DSIP/rriva/research/salt-ririva/paper/salt-arch.png | Salt generic architecture (global feats + constituents → init nets → unified encoder → tasks). | Explains the GN2/salt baseline; cite Salt. |
| /storage3/DSIP/rriva/research/tabicl/docs/figures/architecture.png | TabICL column-then-row architecture (third-party). | Related-work inspiration only. |
| /storage3/DSIP/rriva/research/salt-ririva/slurm/scratch_lambda_scan/output/run_20260928_195621/plots/best_vs_lambda_jets_classification_loss.png, .../run_20261001_091525/plots/... | salt GN2 λ-scan best val CE vs λ. | GN2 pillar (owned by the salt area). |
| Diary 2026-07-01 table (day260701.tex) | Full vs chunked intersample attention timing (b = 256…8192). | Appendix on IA cost, if needed. |

---

## 6. Key verified numbers used above

- JetSet v1 files (h5 metadata): large 168,495,316 jets / 41,102,683 events; medium 25,637,537 /
  6,230,557; small 5,619,475 / 1,365,435. large+medium = 194.1M jets; all three = 199.75M
  (if disjoint); events 48.7M. This matches the poster's "~190M jets" and the README's "~50M tt̄ events".
- Preprocessed datasets (h5 metadata): `251112_112022` train 52.00M / val 5.00M / test 5.00M (used by
  the fm4tag λ scan, verified in a run's .hydra config). `251112_152051` pretrain 39.68M / train
  18.60M / val 1.86M / test 1.86M (SSL campaign). `260720_181751` train 56.00M / val 4.00M / test 2.00M.
- fm4tag λ-scan model: rowcol, per_token ISAB num_inds 16, dim 16, depth 6; **2.2M total params**
  (Lightning summary in rank_0.err). The salt GN2 is 2.2M params [M]. The SSL-campaign model is
  rowcol_concat 7.4–7.5M [F slide 2/3].
