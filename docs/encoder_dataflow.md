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

| object | class | N (tokens) | dim | row_dim = N·dim |
|---|---|---|---|---|
| `jets` (global) | `GlobalTransformerEncoder` | 2 (pt, eta) | 64 | 128 |
| `tracks` (constituent) | `Encoder` | 19 (9 cat + 10 con) | 64 | 1216 |

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
  over the `B` axis, on the flattened `row_dim = N·dim` vector. Expensive,
  batch-composition dependent.

### `col` — `ColTransformer`

Column attention only. Operates directly on `(B, N, dim)`.

```
x (B, N, dim)
   └─ depth × [
        PreNorm(dim) → Attention(over N, within sample)   → + residual
        PreNorm(dim) → FeedForward(dim)                    → + residual
      ]
x (B, N, dim)
```

Everything stays at width `dim` → smallest and most stable.

### `row` — `RowTransformer`

Row attention only. Flattens the tokens, mixes across the batch, unflattens.

```
x (B, N, dim)
   └─ flatten ─► (B, row_dim),  row_dim = N·dim
        └─ depth × RowMixer        # see below
   └─ unflatten ─► (B, N, dim)
```

### `rowcol` — `RowColTransformer`

Column then row, per depth step. Reshapes between the two axes each step.

```
x (B, N, dim)
   └─ depth × [
        # --- column (within-sample), at width dim ---
        PreNorm(dim) → Attention(over N)        → + residual
        PreNorm(dim) → FeedForward(dim)         → + residual
        flatten ─► (B, row_dim)
        # --- row (intersample) ---
        RowMixer                                 # see below
        unflatten ─► (B, N, dim)
      ]
x (B, N, dim)
```

---

## RowMixer — the row (intersample) step, with optional bottleneck

Used by `row` and `rowcol`. Input/output are the flattened `(B, row_dim)`.
`row_dim = N·dim` is large (1216 for tracks), so running attention **and** the
GEGLU feed-forward at `row_dim` is what made this block ~95% of the model. The
`out_row_dim` knob down-projects the whole row step into a bottleneck.

### Bottleneck path — `out_row_dim < row_dim` (default: `out_row_dim = 128`)

```
x (B, row_dim)
   │  down: Linear(row_dim → out_row_dim)
   ▼
z (B, out_row_dim)
   │  PreNorm → RowAttention(across B, intersample) → + residual   ┐ all at
   │  PreNorm → FeedForward(out_row_dim)            → + residual   ┘ out_row_dim
   ▼
z (B, out_row_dim)
   │  up: Linear(out_row_dim → row_dim)      # weight+bias zero-init
   ▼
x + up(z)      (B, row_dim)                  # single residual; starts as identity
```

- The up-projection is **zero-initialised**, so the row step begins as the
  identity (the encoder starts col-only) and learns intersample mixing
  adapter-style — a stabiliser for the otherwise-diverging row path.
- Attention/FFN cost now scales with `out_row_dim`, not `row_dim`
  (tracks `RowColTransformer`: 22.9M → 2.09M params).

### Full-width path — `out_row_dim = None` (or `≥ row_dim`)

```
x (B, row_dim)
   │  PreNorm(row_dim) → RowAttention(across B) → + residual
   │  PreNorm(row_dim) → FeedForward(row_dim)   → + residual
   ▼
x (B, row_dim)
```

### RowAttention variants (both operate over the `B` axis)

- `chunk_size = null` → **`RowAttention`**: every sample attends to every other
  sample in the batch.
- `chunk_size = k` → **`ChunkedRowAttention`**: the batch is split into disjoint
  groups of `k` (randomly permuted in train, contiguous in eval) and attention
  is computed within each group. Padding rows (when `k ∤ B`) are masked out.

> Consequence of row attention: a sample's output depends on which other
> samples share its batch/chunk — keep evaluation batch composition fixed when
> comparing numbers.

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