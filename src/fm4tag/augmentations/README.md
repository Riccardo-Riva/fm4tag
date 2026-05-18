# Augmentation framework — integration notes

Drop-in replacement for the augmentations folder. The eight files go under
`src/fm4tag/augmentations/`.

## Design

Each augmentation is an `nn.Module` that declares one of three pipeline
positions via a class attribute `stage`:

| Stage | When it runs | Dict shape |
|---|---|---|
| `PRE_FLATTEN` | on `(B, C, F)` + valid mask, before flatten-by-valid | `{categorical (B,C,F_cat), continuous (B,C,F_con), valid (B,C)}` |
| `RAW` | on flat raw features, before `embed_data` | `{categorical (N,F_cat) long, continuous (N,F_con) float}` |
| `EMBEDDING` | on embedded tokens, after `embed_data`, before transformer | `{categorical (N,F_cat,dim), continuous (N,F_con,dim)}` |

The `Compose` object groups a list of augmentations into the three stages
and exposes `apply_pre_flatten`, `apply_raw`, `apply_embedding` so the
pretraining module can call them at the right point.

## Wiring into `_compute_loss_for_constituent`

The current pipeline in `pretrain_module.py` looks like (simplified):

```python
const = batch['constituents'][obj_name]
valids_flat = rearrange(const['valid'], 'b c -> (b c)')
x_categ = rearrange(const['categorical'], 'b c f -> (b c) f')[valids_flat]
x_cont  = rearrange(const['continuous'],  'b c f -> (b c) f')[valids_flat]

if 'cutmix' in cfg_pt.aug:
    x_categ_2, x_cont_2 = add_noise(x_categ, x_cont, lam=cfg_pt.aug_lambda)

x_cat_enc_2, x_con_enc_2 = embed_data(x_categ_2, x_cont_2, encoder)

if 'mixup' in cfg_pt.aug:
    x_cat_enc_2, x_con_enc_2 = mixup_data(x_cat_enc_2, x_con_enc_2, lam=...)

X_2 = encoder(x_cat_enc_2, x_con_enc_2)
```

With the new framework (single view), per-view `compose` built once at
module construction from `cfg.pretrain.views[k].augmentations`:

```python
# 1. Pre-flatten stage — operates on (B, C, F) + valid
pre = compose.apply_pre_flatten({
    'categorical': const['categorical'],
    'continuous':  const['continuous'],
    'valid':       const['valid'],
})
valids_flat = rearrange(pre['valid'], 'b c -> (b c)')
x_categ = rearrange(pre['categorical'], 'b c f -> (b c) f')[valids_flat]
x_cont  = rearrange(pre['continuous'],  'b c f -> (b c) f')[valids_flat]

# 2. Raw stage — operates on (N_valid, F)
raw = compose.apply_raw({'categorical': x_categ, 'continuous': x_cont})

# 3. Embed (unchanged)
x_cat_enc, x_con_enc = embed_data(raw['categorical'], raw['continuous'], encoder)

# 4. Embedding stage
emb = compose.apply_embedding({'categorical': x_cat_enc, 'continuous': x_con_enc})

# 5. Transformer
X = encoder(emb['categorical'], emb['continuous'])
```

For the **global object**, only the raw and embedding stages run (no
`valid` mask) — same dict format, `categorical=None`.

## Multi-view extension

For `K` views, build `K` `Compose` objects from `cfg.pretrain.views`:

```python
self.composes = nn.ModuleList([
    build_from_config(view_cfg.get('augmentations', []))
    for view_cfg in cfg.pretrain.views
])
```

In the training step, loop over the composes and stack the resulting
`X_k` tensors along a new dim — or concatenate along dim 0 — then feed
into the multi-view contrastive loss. Note that for memory efficiency
you can concatenate raw inputs across views and run the encoder once
on a `(K*N_valid, ...)` batch.

## Config example

```yaml
pretrain:
  views:
    - augmentations: []                              # clean view
    - augmentations:
        - {name: track_dropout, drop_prob: 0.15}
        - {name: cutmix, lam: 0.7}
        - {name: mixup, lam: 0.8}
    - augmentations:
        - {name: scarf, corrupt_frac: 0.4}
        - {name: gaussian_noise, space: embedding, sigma: 0.05}
```

## Notes on the existing functional helpers

`embed_data` from `augmentations/augmentations.py` is **kept** — it is not
really an augmentation, it's the encoder's embedding step. The new
`Augmentation` classes layer on top of it.

`add_noise` and `mixup_data` are functionally superseded by `CutMix` and
`Mixup`. You can either:

- Delete them from `augmentations.py` along with their imports in the
  pretrain module (cleaner, one-time migration).
- Keep them around as functional wrappers for backward compatibility with
  any old configs that use `cfg.pretrain.aug = ['cutmix', 'mixup']`.

## Registry & extensibility

To add a new augmentation, drop a file next to the others:

```python
from .base import Augmentation, Stage, register

@register('my_aug')
class MyAug(Augmentation):
    stage = Stage.RAW
    def __init__(self, my_param=1.0):
        super().__init__()
        self.my_param = my_param
    def forward(self, data):
        ...
```

then import it from `__init__.py` so the decorator runs. The YAML name
`my_aug` is now usable in `augmentations:` lists.

## Tested

A smoke test (`/home/claude/smoke_test.py` during development) verified:

- Registry population from decorators
- `build_from_config` correctly groups augmentations by stage
- Each augmentation preserves expected shapes & dtypes
- `min_valid` floor in `TrackDropout` is respected
- Global-object path (no categorical, no valid) works
- Empty config returns a pass-through `Compose`
- K=3 views with mixed-stage augmentations builds cleanly

The integration into `pretrain_module.py` itself is a separate step (part
of the multi-view contrastive refactor).
