# fm4tag — self-supervised pretraining tests (plan, October 2026)

## Question

Does self-supervised pretraining of the SAINT-like encoders (attention across features
and across jets) with a contrastive loss improve the flavour tagger? Does a contrastive
term on the global jet embedding add to it?

The earlier λ scan used a contrastive term on the jet embedding only during supervised
training, where it acted as a regulariser (lower validation CE). These tests ask whether
contrastive *pretraining* helps as well.

## Tests

**Pretraining** (no labels):
- **P0:** contrastive loss on the per-track and global-jet-feature embeddings.
- **P1:** P0 plus a contrastive loss on the global jet embedding.

**Downstream training** (cross-entropy), each without and with the jet-embedding
contrastive term (JC):

| | without JC | with JC |
|---|---|---|
| fine-tune the pretrained model | FT ← P0 | FT ← P1 |
| train from scratch (baseline) | SC | SC + JC |
| frozen pretrained encoders, train aggregator + head | FR ← P0 | FR ← P1 |

The FR arm measures how good the pretrained representation is by itself. FT against SC
measures the gain from pretraining at equal downstream budget.

**Side question.** Which intersample attention is better: SAINT's, which attends across
jets on the concatenated feature vector, or a per-feature attention across jets?

## Setup

- **Data:** one preprocessing with disjoint splits by event number:
  - 39.7M jets for pretraining;
  - 18.6M labelled jets for downstream training;
  - 1.9M for validation and 1.9M for test.
- **Reference:** GN2v01 on the same test set.
- **Metrics:**
  - validation cross-entropy (mean ± spread over seeds);
  - test ROC curves;
  - c-rejection at the 77 % b-efficiency working point, with light and τ rejection
    kept at least at GN2v01's;
  - quality of the jet embedding (uniformity, effective rank);
  - compute cost.
- **Seeds:** 4 per arm. The pretrained arms use 2 pretraining seeds × 2 fine-tuning seeds,
  so the spread includes pretraining variance.

## Framework changes made for these tests

- **Chunked intersample attention.** Attention across jets, and the contrastive losses,
  now act within chunks of a fixed number of jets, instead of over the whole batch.
  - The chunk size sets the context the model sees.
  - The batch size only sets the optimisation, so it can fill the GPU memory (a cluster
    requirement) without slowing training per jet. Measured: flat cost per jet from 1k
    to 8k jets per batch.
  - The inducing-point approximation (ISAB) is removed.
- **Frozen-encoder training mode**, and correct handling of which parts were actually
  pretrained.
- **Speed:**
  - the aggregator now runs on real tracks only (about 80 % of the track slots were
    padding);
  - jobs pre-load the data files into memory, because cold network-filesystem reads
    were the data-loading bottleneck.

## Plan and timeline

1. **Benchmark:** speed and memory per configuration. This fixes the chunk size candidates
   and the batch sizes. Running now.
2. **Pilots** (short runs, a few seeds):
   - SAINT vs per-feature attention, from scratch;
   - pretraining temperature × augmentation strength, judged by a frozen-encoder probe;
   - learning rate per arm.
3. **Final runs:** the six arms with seeds, then the comparison plots.

Estimated cost is about 850 GPU-hours, roughly 1.5–2 weeks on 4–6 GPUs. A full pipeline
test (pretraining → fine-tune / scratch / frozen) has run successfully at small scale.
