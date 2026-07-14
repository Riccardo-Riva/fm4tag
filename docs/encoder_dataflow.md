# Encoder data flow

Scope: the **per-object encoders only** (not the aggregator, head, or losses).
Each object (the global `jets`, each constituent e.g. `tracks`) is encoded
independently by one encoder. An encoder has three stages:

```
raw features ──► [1] embedding front-end ──► tokens (B, N, dim)
                                              │
                                              ▼
                              [2] transformer backbone   ◄── swappable: col | row | rowcol
                                              │
                                    tokens (B, N, dim)
                                              │
                              ┌───────────────┼────────────────┐
                              ▼               ▼                ▼
                        [3a] projector   [3b] cat_recon   [3b] con_recon
                        (contrastive)    (denoising)      (denoising)
```

`N` = number of tokens = number of features for that object; `dim` = token
embedding width. Only stage **[2]** differs between the three encoder kinds.

Concrete sizes in the default config:

| object | class | N (tokens) | dim |
|---|---|---|---|
| `jets` (global) | `GlobalTransformerEncoder` | 2 (pt, eta) | 64 |
| `tracks` (constituent) | `Encoder` | 19 (9 cat + 10 con) | 64 |

> **What is `B`?** For the global object it is the jets in the batch. For a
> constituent object it is **every valid constituent of every jet in the batch**,
> packed by boolean-indexing the padded `(B_jets, C)` grid (see
> `modules/view_encoding.py`). So `B` is *data-dependent and changes every step* —
> which is why the row step must handle a variable set size natively.

---

## [1] Embedding front-end — raw features → tokens `(B, N, dim)`

The front-end turns each raw feature into one `dim`-wide token. It differs by
object type; the backbone downstream is identical.

### Global object (`GlobalTransformerEncoder`) — continuous only

```
x_cont  (B, F_g)                         # F_g continuous global features
   │  per-feature grouped-Conv1d MLP  (fc1 → ReLU → fc2, groups=F_g)
   ▼
tokens  (B, F_g, feature_dim)            # one token per feature   → N = F_g
```

### Constituent object (`Encoder`) — categorical + continuous

```
x_categ (B, F_cat) int        x_cont (B, F_con) float
   │  + category offsets          │  per-feature grouped-Conv1d MLP
   │  nn.Embedding lookup         │  (cont_fc1 → ReLU → cont_fc2)
   ▼                              ▼
(B, F_cat, dim)               (B, F_con, dim)
   └──────────────┬───────────────┘
                  ▼  concat along the token axis
          tokens (B, N, dim),  N = F_cat + F_con
```

> Embedding is done by `embed_data(...)` *before* `Encoder.forward`, so
> augmentations can act on the embedded tokens (EMBEDDING stage) or on the raw
> values (RAW stage) in between. `Encoder.forward` receives the tokens and runs
> only stages [2].

---

## [2] Transformer backbone — the swappable part

Selected by the `type` of the layer (`encoders=col_v0 | row_v0 | rowcol_v0`).
All three take and return `(B, N, dim)` and stack `depth` identical steps.

Two attention axes are in play:

- **Column (within-sample):** tokens of *one* sample attend to each other —
  over the `N` axis, at width `dim`. Cheap, batch-independent.
- **Row (intersample):** each *sample* attends to *other samples* in the batch —
  over the `B` axis, **independently per token and at width `dim`**, with the
  projections shared across tokens. Batch-composition dependent.

This is the factorisation tabicl uses (`ColEmbedding` attends over samples
per-column; `RowInteraction` attends over columns per-row). The row step never
sees a token's neighbours — cross-feature mixing is entirely the column step's
job, and `rowcol` alternates the two.

### `col` — `ColTransformer`

Column attention only. Operates directly on `(B, N, dim)`.

```
x (B, N, dim)
   └─ depth × [
        x + Attention(over N, within sample)(norm(x))
        x + FeedForward(dim)(norm(x))
      ]
x (B, N, dim)
```

Everything stays at width `dim` → smallest and most stable.

### `row` — `RowTransformer`

Row attention only. Flattens the tokens, mixes across the batch, unflattens.

```
x (B, N, dim)
   └─ depth × RowMixer        # see below
x (B, N, dim)
```

### `rowcol` — `RowColTransformer`

Column then row, per depth step. Reshapes between the two axes each step.

```
x (B, N, dim)
   └─ depth × [
        # --- column (within-sample), at width dim ---
        x + Attention(over N)(norm(x))
        x + FeedForward(dim)(norm(x))
        # --- row (intersample) ---
        RowMixer                                 # see below
      ]
x (B, N, dim)
```

---

## RowMixer — the row (intersample) step

Used by `row` and `rowcol`. Input and output are the token grid `(B, N, dim)`;
no flattening. Row attention transposes to `(N, B, dim)` so the token axis is a
batch dimension and the **sample axis `B` is the attention sequence**:

```
x (B, N, dim)
   │  x + RowAttention|InducedRowAttention (over B, per token)(norm(x))
   │  x + FeedForward(dim)(norm(x))
   ▼
x (B, N, dim)
```

Because every projection has width `dim` (64), not `N·dim` (1216), the row step
is affordable at full width — the old `out_row_dim` down/up bottleneck is gone.

### RowAttention variants (both attend over the `B` axis, per token)

`num_inds` in the layer config picks one:

- `num_inds: null` → **`RowAttention`**: all-pairs — every sample attends to
  every other sample. O(B²).
- `num_inds: m` (default 16) → **`InducedRowAttention`**: ISAB. Two
  cross-attentions through `m` learned inducing points, O(B·m):

```
        inducing points I (N, m, dim)          x (N, B, dim)
                    │                               │
                    └────► MAB₁: I attends to x ◄───┘        # read the batch
                                  │
                          summary (N, m, dim)
                                  │
                    ┌────► MAB₂: x attends to summary        # every sample reads it
                    ▼
                 out (N, B, dim)
```

Why ISAB is the default:

- **Variable `B` is native.** The inducing points are fixed learned queries, so
  the packed constituent count can change every step. No chunking, no random
  permutation, no tail padding — all of which the old `ChunkedRowAttention`
  needed, and which made a sample's output depend on which 512 others happened
  to land in its chunk.
- **Permutation-invariant**, and identical in train and eval.
- **No attention mask is needed**, which keeps the row path eligible for
  FlashAttention (an arbitrary mask forces the SDPA fallback).
- The out-projection of MAB₂ is **zero-initialised**, so under the residual the
  row step starts as the identity (the encoder begins col-only) and learns
  intersample mixing from there.

The inducing points are **per token** (`(N, m, dim)`). tabicl shares one set
across columns because its column count varies per table; here `N` is fixed and
each token is a distinct physics feature with its own embedding subspace.

> Consequence of row attention (any variant): a sample's output depends on which
> other samples share its batch — keep evaluation batch composition fixed when
> comparing numbers.

---

## Final LayerNorm — closing the pre-norm stack

The blocks are pre-norm (`x + f(norm(x))`, see `models/blocks.py`), so nothing
normalises the residual stream on the way *out* of the stack: its scale is free
to grow with depth as the weights train. The heads below are plain Linears and
would inherit that drift, so both `Encoder` and `GlobalTransformerEncoder` end
with a `norm_out = LayerNorm(dim)` — the `ln_f` of a pre-norm transformer.

> tabicl omits this and instead zero-inits *every* block (attention out-proj and
> FFN second linear), so its whole stack starts as the identity. We zero-init
> only the ISAB out-projection, so the final norm is the more robust guard: it
> holds however far training moves the weights, not just at init.

---

## [3] Heads hanging off the encoder output `(B, N, dim)`

Not part of the backbone, but part of the encoder module — three heads read the
same token output:

- **`projector`** (`simple_MLP`): flattens `(B, N·dim)` → `(B, proj_out)` — the
  per-object embedding used by the contrastive loss (and fed to the aggregator).
- **`cat_reconstructor`** (`sep_MLP`, per token): `(B, F_cat, dim)` → categorical
  logits per feature — categorical denoising (cross-entropy).
- **`con_reconstructor`** (`sep_MLP`, per token): `(B, F_con, dim)` → scalar per
  feature — continuous denoising (MSE).