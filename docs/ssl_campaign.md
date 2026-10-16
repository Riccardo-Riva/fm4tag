# SSL campaign — does contrastive pretraining of the SAINT-like encoders help flavour tagging?

Status (2026-10-16): S0 benchmark done (A40). Decided: S1 chunk-size candidates
128 and 512; batch per arm from S0, FR / FR+JC capped at 32768; A40 cards may be
used alongside L40S. S1 submitted.

## 1. The question

Does pretraining the per-object encoders (column + intersample attention, SAINT-like)
with a contrastive loss on the **feature embeddings** improve the downstream tagger?
And does a contrastive loss on the **global jet embedding** (the aggregator output, "JC")
add to that?

Two side questions come with it:

- **Intersample attention: SAINT-like or per-feature?** SAINT attends across jets on the
  concatenated feature vector (`row_mode: concat`). The per-feature variant runs one
  attention per feature token (`row_mode: per_token`). Which gives the better tagger?
- **Jet chunks.** Intersample attention and the contrastive losses now act within chunks
  of `chunk_size` jets. So the batch can fill the GPU memory without the cost per jet
  growing, and `chunk_size`, not the batch, sets the intersample context.

## 2. The tests

### Pretraining (`pretrain` split, labels unused)

| id | loss | aggregator |
|---|---|---|
| **P0** | contrastive on the feature embeddings: per-track embedding (`tracks_embedding`) and global-jet-feature embedding (`jets_embedding`) | not trained |
| **P1** | P0 + contrastive on the global jet embedding (aggregator output) | trained |

Both are contrastive only (no denoising), with intersample attention in the encoders.

### Downstream trainings (`train` split, cross-entropy always on)

| | without JC | with JC (`jet_contrastive = 0.3`) |
|---|---|---|
| **FT**: fine-tune the pretrained model, everything trains | FT ← P0 | FT ← P1, + JC |
| **SC**: whole tagger from scratch | SC | SC + JC |
| **FR**: pretrained encoders frozen, aggregator + head train | FR ← P0 | FR ← P1, + JC |

- **FR freezes the encoders and their projectors** for the whole run, with no gradient
  and no backward pass. The aggregator stays trainable, otherwise the downstream JC
  would have nothing to act on.
- **The aggregator is loaded only if pretraining trained it** (`load_aggregator: auto`).
  So FT ← P0 and FR ← P0 start with a random aggregator, which trains at the head's
  learning rate, as in SC.
- **Caveat on the JC column.** It switches on JC in pretraining *and* in the downstream
  loss at once. If it wins, two extra arms (FT ← P1 and FR ← P1, both without
  downstream JC) would show which of the two helped. They are optional, decided after
  the main results.

## 3. Common setup

| | |
|---|---|
| data | `data_parallel_preprocess_251112_152051`, split by `eventNumber % 100`: pretrain 0-63 (39.68M jets), train 64-93 (18.6M), val 94-96 (1.86M), test 97-99 (1.86M), all disjoint. Normalisation from the train split for every phase. |
| reference tagger | GN2v01 columns of the test file (the salt GN2 only-JT was trained on `112022`, whose train events overlap this test set) |
| model | `rowcol_v0` (per_token) or `rowcol_concat_v0` (concat), whichever wins S1; dim 16, depth 6; aggregator depth 6 |
| intersample attention | all-pairs within each chunk (ISAB removed); flash varlen on GPU |
| `chunk_size` | 128 or 512 (S1 decides; a model hyper-parameter, also used at inference) |
| batch | per arm, the largest multiple of 8·`chunk_size` under ~38 GiB peak (S0): SC/FT 30720 per_token / 26624 concat, +JC 16384 / 14336, P0 20480 / 16384, P1 14336 / 12288; FR and FR+JC capped at 32768. S1 uses 24576 for every cell (fits both row modes) |
| optimiser | AdamW, wd 5.96e-5, cosine to `max_epochs` with 2 % warmup, bf16-mixed, 1 GPU |
| learning rate | tuned per arm (S3); prior 3e-4 at batch 2048, scaled as √B |
| downstream stop | 100 epochs max, early stopping on `val/head/loss_ce` (patience 12), best checkpoint by the same |
| pretraining budget | 30 epochs, `last.ckpt` |
| FT schedule | pretrained parts at `backbone_lr` (= lr; S3 also tries lr/10), held at 0 for the first 2 epochs |
| JC (when on) | weight 0.3, augmentation strength λ = 0.4, temperature 0.1 (λ-scan priors) |
| seeds | spaced list `42 424 4242 42424 …`, never consecutive (the sampler uses seed + epoch) |

## 4. Stages

Each stage ends at a decision taken on its numbers, before the next is launched.

| stage | runs | decides |
|---|---|---|
| **S0** benchmark | one profiling job: chunk 64-1024 at batch 4096, batch 4096-16384 at chunk 256, for SC, SC+JC, P0, P1, FR, FR+JC × both row modes | two candidate chunk sizes (speed) and the batch of each arm (memory) |
| **S1** architecture | SC (CE only) × {per_token, concat} × the two chunk sizes, 2 seeds | **SAINT-like vs per-feature intersample attention**, and the chunk size |
| **S2** pretraining recipe | P0 × temperature {0.1, 0.3} × λ {0.1, 0.4}, 10 epochs, each followed by a short FR probe (15 epochs, fixed lr), plus one probe of frozen *random* encoders as the floor | the pretraining recipe, judged by the probe's val CE (pretraining losses are not comparable across T and λ); reused for P1 |
| **S3** learning rates | each of the 6 arms × 3 learning rates (×2 log steps); FT also backbone_lr = lr/10 | the learning rate of each arm |
| **final** | P0 and P1 × 2 pretraining seeds (full budget); each pretrained arm × 2 pretraining seeds × 2 downstream seeds; SC × 4 seeds | the results |

- **Budget.** Roughly 350 GPU-h of pilots and 500 GPU-h of final runs: about 1.5-2 weeks
  on 4-6 L40S.
- **Stuck seeds.** A seed is "stuck" if its val CE never gets below a threshold within
  20 epochs. These seeds are left out with one rule for every arm, and listed by name.
  The threshold is re-derived from the pilots, because the dataset and model are new.

## 5. What is compared at the end

Every run is identified from its own `.hydra/config.yaml`, never from its directory name:
- `encoder_ckpt`, followed to the pretraining run to read its `jet_contrastive`;
- `finetune.freeze_encoders`;
- `finetune.loss_weights.jet_contrastive`;
- `row_mode` and `chunk_size`.

1. **Pretraining curves:** P0 and P1 validation losses per component, plus uniformity and
   effective rank of the embeddings.
2. **Downstream curves:** validation CE per arm, mean ± band over seeds.
3. **Best validation CE per arm**, with seed spread, grouped without / with JC.
4. **Test ROC vs GN2v01:**
   - one curve per arm, at each model's own optimal (f_c, f_tau);
   - the fractions are optimised on the **validation** split, then applied to test;
   - ratio panels c, light, tau.
5. **c-rejection at 77 % b-efficiency per arm**, constrained to light- and tau-rejection
   ≥ GN2v01's; mean ± seed spread.
6. **Quality of the global jet embedding per arm** (uniformity, effective rank).
7. **Cost per arm** in GPU-h, with the pretraining shared out over the runs that use it.

All plot wording lives in each plot's YAML `texts:`. No plot carries "ATLAS".

## 6. How a stage is launched

Every run composes `src/fm4tag/configs/campaign_ssl.yaml` plus an arm:

```bash
fm4tag --config-name=campaign_ssl arm=pretrain                                   # P0
fm4tag --config-name=campaign_ssl arm=pretrain pretrain.loss_weights.jet_contrastive=1.0   # P1
fm4tag --config-name=campaign_ssl arm=scratch  [finetune.loss_weights.jet_contrastive=0.3] # SC
fm4tag --config-name=campaign_ssl arm=finetune encoder_ckpt=<P ckpt> [...JC]     # FT
fm4tag --config-name=campaign_ssl arm=frozen   encoder_ckpt=<P ckpt> [...JC]     # FR
```

On the cluster, a stage is a cell file (one line per cell: name + overrides) given to
`slurm/ssl_campaign.sh`. It submits one 1-GPU job per cell and runs the seeds
back-to-back inside it:

```bash
CELLS=scripts/configs/campaign/<stage>.cells EST_SECONDS_PER_EPOCH=<measured> \
    bash slurm/ssl_campaign.sh                       # DRY_RUN=1 to check first
```

- **Pretraining checkpoints.** A downstream cell can name its pretraining as
  `encoder_ckpt=@${PRE}/p0/seed_42`. The job resolves that to the pretraining run's
  `last.ckpt` when it starts.
- **Queueing behind the pretraining.** `DEPENDENCY=afterok:<jobids>` holds the downstream
  jobs until the pretraining jobs have finished.
- **Checks at submit time.** Each cell is composed when it is submitted, so a typo fails
  before anything is queued. Each job also reads its data files once at the start to warm
  the node's page cache: cold NFS reads were the loader's bottleneck.
- **A tiny full chain to copy from.** `scripts/configs/campaign/smoke_*.cells` run
  P0/P1 → SC/FT/FR in a few minutes.
