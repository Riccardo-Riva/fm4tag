# Tests

## Running the DDP tests

```bash
# All DDP tests, verbose output
uv run pytest tests/ddp/ -v

# With a 30-second per-test timeout (catches deadlocks)
uv run pytest tests/ddp/ -v --timeout=30

# Only DDP-marked tests
uv run pytest tests/ddp/ -v -m ddp

# Single test by name
uv run pytest tests/ddp/test_embedding_gather.py::test_basic_size_matched_gather -v
```

The tests use CPU and the `gloo` backend — no GPU required.  Each DDP test
spawns 2 child processes via `torch.multiprocessing.spawn`.  Expect ~5–10 s
per test for process setup overhead.

## What each test verifies and which real-world bug it catches

| Test | What it checks | Bug it would catch |
|------|---------------|-------------------|
| `test_basic_size_matched_gather` | After the trim-to-n_min gather, output shape equals `(2·n_min, D)`, rank slots are in the correct order, and both ranks hold the identical tensor. | Wrong trim index or off-by-one in n_min would produce mismatched shapes or corrupted embeddings silently. |
| `test_zero_rank_skips` | When one rank has zero samples, `gather_embeddings_sized` returns `None` on **both** ranks without issuing a second `all_gather`. Enforced with a 30 s timeout. | The original per-rank `if z_local is None: continue` guard would let the rank with data fall through to `all_gather` while the empty rank exited — the surviving rank would block forever (NCCL watchdog hang). |
| `test_multi_object_asymmetric` | With 3 objects and one rank missing data for only the third object, all ranks iterate in the same encoder-key order and the missing-object n_local=0 collective does not deadlock. | Using `store.keys()` instead of `encoders.keys()` would cause the rank with missing data to skip the collective, hanging the other rank. |
| `test_pretrain_module_e2e` | `PretrainModule.on_validation_epoch_end` logs uniformity and effective_rank values that match those computed by manually concatenating the per-rank embeddings. | A silent bug in the gather (wrong flatten dimension, wrong trim) would log metrics computed on a subset or wrongly ordered tensor — metrics would appear plausible but differ from single-GPU training. |
| `test_sanity_check_skip` | When `trainer.sanity_checking=True`, the epoch-end hook clears both buffers and emits no log calls. | Without the early-return guard, the sanity-check pass would try to gather embeddings before training starts; with only one validation batch the buffer might be tiny and produce misleading early-epoch metrics. |
