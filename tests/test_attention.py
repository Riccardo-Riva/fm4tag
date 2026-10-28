"""Tests for the intersample (row) attention.

:class:`RowAttention` takes the token grid ``(B, N, dim)`` and attends all-pairs
over the **sample** axis ``B``, independently per token.
"""

from __future__ import annotations

import pytest
import torch

from fm4tag.models.attention import (
    LeanRMSNorm,
    RowAttention,
    RowGroups,
    grouped_attention,
)
from fm4tag.models.backbones.transformers import RowMixer

B, N, DIM = 10, 5, 16


def _row_attn(heads: int = 4, qk_norm: bool = False) -> RowAttention:
    torch.manual_seed(0)
    attn = RowAttention(dim=DIM, heads=heads, dim_row_head=8, qk_norm=qk_norm).eval()
    # ``to_out`` ships zero-initialised (identity-at-init under the residual),
    # which would make every behavioural test below trivially true.
    torch.nn.init.normal_(attn.to_out.weight, std=0.1)
    torch.nn.init.normal_(attn.to_out.bias, std=0.1)
    return attn


def test_row_attention_starts_as_zero():
    torch.manual_seed(0)
    attn = RowAttention(dim=DIM, heads=4, dim_row_head=8).eval()
    with torch.no_grad():
        out = attn(torch.randn(B, N, DIM))
    assert torch.count_nonzero(out) == 0


# ── shape / mask handling ────────────────────────────────────────────────────


def test_shape_roundtrip():
    attn = _row_attn()
    out = attn(torch.randn(B, N, DIM))
    assert out.shape == (B, N, DIM)
    assert torch.isfinite(out).all()


def test_mask_shape_broadcasts():
    """Regression: the per-sample (B,) mask must work when B != heads."""
    attn = _row_attn()
    x = torch.randn(B, N, DIM)
    mask = torch.ones(B, dtype=torch.bool)
    mask[7:] = False
    out = attn(x, mask=mask)
    assert out.shape == (B, N, DIM)
    assert torch.isfinite(out).all()


def test_masked_samples_are_ignored_as_keys():
    """Garbage in masked-out samples must not change the valid samples' outputs."""
    attn = _row_attn()
    x = torch.randn(B, N, DIM)
    mask = torch.ones(B, dtype=torch.bool)
    mask[7:] = False

    x_garbage = x.clone()
    x_garbage[7:] = 1e4

    with torch.no_grad():
        out = attn(x, mask=mask)
        out_garbage = attn(x_garbage, mask=mask)

    assert torch.allclose(out[:7], out_garbage[:7], atol=1e-4)


def test_all_valid_mask_matches_no_mask():
    attn = _row_attn()
    x = torch.randn(B, N, DIM)
    with torch.no_grad():
        out_none = attn(x)
        out_all = attn(x, mask=torch.ones(B, dtype=torch.bool))
    assert torch.allclose(out_none, out_all, atol=1e-6)


# ── the tabicl factorisation: row attention acts per token ───────────────────


def test_token_channels_are_independent():
    """Output at token ``n`` must depend only on the input at token ``n``.

    Row attention runs over the sample axis independently per token — mixing
    across tokens is the column attention's job.  Perturbing a different token
    must therefore leave this token's output untouched.
    """
    attn = _row_attn()
    x = torch.randn(B, N, DIM)
    x_perturbed = x.clone()
    x_perturbed[:, 3, :] += 5.0  # disturb token 3 only

    with torch.no_grad():
        out = attn(x)
        out_perturbed = attn(x_perturbed)

    others = [n for n in range(N) if n != 3]
    assert torch.allclose(out[:, others], out_perturbed[:, others], atol=1e-5)
    assert not torch.allclose(out[:, 3], out_perturbed[:, 3], atol=1e-3)


# ── permutation equivariance over the sample axis ────────────────────────────


def test_permutation_equivariant_over_samples():
    """Reordering the batch must reorder the outputs identically.

    This is what the old chunked row attention could not offer: it drew a fresh
    random permutation every training step, so a sample's output depended on
    which 512 others happened to land in its chunk.
    """
    attn = _row_attn()
    x = torch.randn(B, N, DIM)
    perm = torch.randperm(B)

    with torch.no_grad():
        out = attn(x)
        out_perm = attn(x[perm])

    assert torch.allclose(out[perm], out_perm, atol=1e-5)


def test_deterministic_in_train_mode():
    """No stochastic chunking: train mode must be as deterministic as eval."""
    attn = _row_attn()
    attn.train()
    x = torch.randn(B, N, DIM)
    with torch.no_grad():
        assert torch.equal(attn(x), attn(x))


# ── RowMixer: per_token vs concat row_mode ───────────────────────────────────


def _row_mixer(row_mode: str, zero_init: bool = False) -> RowMixer:
    torch.manual_seed(0)
    mixer = RowMixer(
        DIM,
        N,
        heads=2,
        dim_row_head=8,
        ff_mult=1,
        attn_dropout=0.0,
        ff_dropout=0.0,
        row_mode=row_mode,
    ).eval()
    if not zero_init:
        torch.nn.init.normal_(mixer.attn.fn.to_out.weight, std=0.1)
    return mixer


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_row_mixer_shape_roundtrips(row_mode):
    """Both modes return the (B, N, dim) grid."""
    mixer = _row_mixer(row_mode)
    out = mixer(torch.randn(B, N, DIM))
    assert out.shape == (B, N, DIM)
    assert torch.isfinite(out).all()


def test_row_mixer_rejects_unknown_mode():
    with pytest.raises(ValueError, match='row_mode'):
        _row_mixer('sideways')


def test_concat_row_attention_runs_at_row_dim():
    """In concat mode the row attention operates on the flattened N*dim vector."""
    mixer = _row_mixer('concat')
    assert mixer.attn.fn.to_qkv.in_features == N * DIM


def test_per_token_row_attention_runs_per_token():
    mixer = _row_mixer('per_token')
    assert mixer.attn.fn.to_qkv.in_features == DIM


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_row_mixer_attn_sublayer_is_identity_at_init(row_mode):
    """Zero-init out-projection ⇒ the row-attention sublayer starts as a no-op."""
    mixer = _row_mixer(row_mode, zero_init=True)
    x = torch.randn(B, N, DIM)
    if row_mode == 'concat':
        x = x.reshape(B, 1, N * DIM)
    with torch.no_grad():
        assert torch.allclose(mixer.attn(x), x, atol=1e-6)


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_row_mixer_permutation_equivariant_over_samples(row_mode):
    """Reordering the batch reorders the outputs identically, in either mode."""
    mixer = _row_mixer(row_mode)
    x = torch.randn(B, N, DIM)
    perm = torch.randperm(B)
    with torch.no_grad():
        out, out_perm = mixer(x)[perm], mixer(x[perm])
    assert torch.allclose(out, out_perm, atol=1e-5)


def test_concat_mixes_across_tokens_but_per_token_does_not():
    """The two modes differ exactly in whether the row step couples feature tokens.

    per_token: perturbing token 3 leaves the other tokens' outputs untouched.
    concat:    the row attention sees the whole flattened vector, so it does not.
    """
    x = torch.randn(B, N, DIM)
    x_pert = x.clone()
    x_pert[:, 3, :] += 5.0
    others = [n for n in range(N) if n != 3]

    per_token = _row_mixer('per_token')
    with torch.no_grad():
        a, b = per_token(x), per_token(x_pert)
    assert torch.allclose(a[:, others], b[:, others], atol=1e-5)

    concat = _row_mixer('concat')
    torch.nn.init.normal_(concat.ff.fn.net[0].weight, std=0.1)
    with torch.no_grad():
        a, b = concat(x), concat(x_pert)
    assert not torch.allclose(a[:, others], b[:, others], atol=1e-3)


# ── chunked (grouped) row attention ─────────────────────────────────────────

# Contiguous chunks of the B=10 axis, including an empty one.
_BOUNDS = [0, 3, 3, 7, 10]


def _groups(bounds=_BOUNDS) -> RowGroups:
    return RowGroups.from_boundaries(torch.tensor(bounds))


def _per_group(fn, x, bounds=_BOUNDS):
    return torch.cat([fn(x[s:e]) for s, e in zip(bounds[:-1], bounds[1:]) if e > s])


def test_row_groups_max_seqlen():
    assert _groups().max_seqlen == 4
    assert _groups().cu_seqlens.dtype == torch.int32


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_grouped_row_mixer_matches_per_chunk_calls(row_mode):
    """Chunked attention == running the block separately on each chunk."""
    mixer = _row_mixer(row_mode)
    x = torch.randn(B, N, DIM)
    with torch.no_grad():
        out = mixer(x, groups=_groups())
        ref = _per_group(mixer, x)
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_groups_isolate_chunks(row_mode):
    """Changing one chunk's samples must not move any other chunk's outputs."""
    mixer = _row_mixer(row_mode)
    x = torch.randn(B, N, DIM)
    x_pert = x.clone()
    x_pert[7:] += 5.0  # the last chunk only
    with torch.no_grad():
        out, out_pert = mixer(x, groups=_groups()), mixer(x_pert, groups=_groups())
    assert torch.equal(out[:7], out_pert[:7])
    assert not torch.allclose(out[7:], out_pert[7:], atol=1e-3)


def test_single_group_equals_ungrouped():
    attn = _row_attn()
    x = torch.randn(B, N, DIM)
    with torch.no_grad():
        out = attn(x, groups=_groups([0, B]))
        ref = attn(x)
    assert torch.allclose(out, ref, atol=1e-5)


def test_grouped_permutation_equivariant_within_chunk():
    attn = _row_attn()
    x = torch.randn(B, N, DIM)
    perm = torch.arange(B)
    perm[3:7] = perm[3:7][torch.tensor([2, 0, 3, 1])]
    with torch.no_grad():
        out, out_perm = attn(x, groups=_groups()), attn(x[perm], groups=_groups())
    assert torch.allclose(out[perm], out_perm, atol=1e-5)


def test_grouped_rejects_mask():
    with pytest.raises(ValueError, match='mask or groups'):
        _row_attn()(
            torch.randn(B, N, DIM),
            mask=torch.ones(B, dtype=torch.bool),
            groups=_groups(),
        )


def test_grouped_attention_handles_empty_input():
    q = torch.randn(0, 2, 8)
    assert grouped_attention(q, q, q, _groups([0, 0])).shape == (0, 2, 8)


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_encoder_grouped_matches_per_chunk_calls(row_mode):
    """Whole stack (col + row, depth 2): chunks never see each other."""
    from fm4tag.models.backbones import Encoder

    torch.manual_seed(0)
    encoder = Encoder(
        categories=[3, 4],
        num_continuous=2,
        dim=DIM,
        layers=[
            {
                'type': 'rowcol',
                'depth': 2,
                'col_heads': 2,
                'row_heads': 2,
                'dim_head': 8,
                'dim_row_head': 8,
                'row_mode': row_mode,
            }
        ],
    ).eval()
    for m in encoder.modules():
        if isinstance(m, RowAttention):
            torch.nn.init.normal_(m.to_out.weight, std=0.1)
    x_cat, x_con = torch.randn(B, 2, DIM), torch.randn(B, 2, DIM)
    with torch.no_grad():
        out = encoder(x_cat, x_con, groups=_groups())
        bounds = [(s, e) for s, e in zip(_BOUNDS[:-1], _BOUNDS[1:]) if e > s]
        ref = torch.cat([encoder(x_cat[s:e], x_con[s:e]) for s, e in bounds])
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
@pytest.mark.parametrize('heads,d', [(38, 8), (8, 32), (4, 64)])
@pytest.mark.parametrize('max_pad', [0, 512])
def test_varlen_matches_loop_on_gpu(heads, d, max_pad):
    """bf16 flash varlen vs the exact fp32 per-chunk loop, forward and backward.

    ``max_pad`` over-estimates ``max_seqlen`` (the aggregator passes the slot
    count as a sync-free upper bound): the kernel must not care.
    """
    torch.manual_seed(0)
    bounds = [0, 300, 300, 1100, 1137]
    exact = RowGroups.from_boundaries(torch.tensor(bounds, device='cuda'))
    groups = RowGroups(exact.cu_seqlens, exact.max_seqlen + max_pad)
    q, k, v = (
        torch.randn(1137, heads, d, device='cuda', requires_grad=True) for _ in range(3)
    )
    out = grouped_attention(*(t.bfloat16() for t in (q, k, v)), groups)
    out.float().square().sum().backward()
    grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    ref = grouped_attention(q, k, v, groups)
    ref.square().sum().backward()
    assert torch.allclose(out.float(), ref, atol=3e-2, rtol=3e-2)
    for g, t in zip(grads, (q, k, v)):
        assert torch.allclose(g, t.grad, atol=5e-2, rtol=5e-2)


# ── pre-norm stack must be closed by a final LayerNorm ───────────────────────


@pytest.mark.parametrize('depth', [1, 6, 12])
def test_encoder_output_is_normalised_regardless_of_depth(depth):
    """The pre-norm residual stream is unnormalised on the way out of the stack.

    ``Encoder.norm_out`` (the ``ln_f`` of a pre-norm transformer) is what keeps
    the encoder output — and therefore the projector and reconstructor inputs —
    at a fixed scale however deep the stack is and however far the weights drift.
    """
    from fm4tag.models.backbones import Encoder

    torch.manual_seed(0)
    encoder = Encoder(
        categories=[3, 4],
        num_continuous=2,
        dim=DIM,
        layers=[
            {
                'type': 'rowcol',
                'depth': depth,
                'col_heads': 2,
                'row_heads': 2,
                'dim_head': 8,
                'dim_row_head': 8,
            }
        ],
    ).eval()

    # Feed a stream that is already far off unit scale.
    x_cat = torch.randn(32, 2, DIM) * 10.0
    x_con = torch.randn(32, 2, DIM) * 10.0
    with torch.no_grad():
        out = encoder(x_cat, x_con)

    assert out.shape == (32, 4, DIM)
    per_token_std = out.std(dim=-1)
    assert torch.allclose(per_token_std, torch.ones_like(per_token_std), atol=0.1)


# ── the CUDA grid limit on the kernel batch ─────────────────────────────────


def test_column_attention_slices_beyond_kernel_batch_limit(monkeypatch):
    import fm4tag.models.attention as attention_mod
    from fm4tag.models.attention import Attention

    torch.manual_seed(0)
    attn = Attention(DIM, heads=2, dim_head=8).eval()
    x = torch.randn(10, N, DIM)
    mask = torch.rand(10, N) > 0.2
    mask[:, 0] = True
    with torch.no_grad():
        ref, ref_masked = attn(x), attn(x, mask=mask)
        monkeypatch.setattr(attention_mod, 'MAX_KERNEL_BATCH', 3)
        out, out_masked = attn(x), attn(x, mask=mask)
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(out_masked, ref_masked, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
def test_column_attention_over_70k_tracks_on_gpu():
    """More packed tracks than a CUDA grid dimension takes (65535)."""
    from fm4tag.models.attention import Attention

    torch.manual_seed(0)
    attn = Attention(DIM, heads=2, dim_head=8).cuda().eval()
    x = torch.randn(70_000, 19, DIM, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = attn(x)
        head = attn(x[:1000])
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:1000].float(), head.float(), atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
def test_varlen_slices_beyond_kernel_batch_limit_on_gpu(monkeypatch):
    import fm4tag.models.attention as attention_mod

    torch.manual_seed(0)
    bounds = [0, 30, 30, 75, 120, 121, 160]
    groups = RowGroups.from_boundaries(torch.tensor(bounds, device='cuda'))
    q, k, v = (
        torch.randn(160, 4, 32, device='cuda', dtype=torch.bfloat16) for _ in range(3)
    )
    ref = grouped_attention(q, k, v, groups)
    monkeypatch.setattr(attention_mod, 'MAX_KERNEL_BATCH', 2)
    out = grouped_attention(q, k, v, groups)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2)


def test_varlen_slices_respect_both_launch_limits(monkeypatch):
    import fm4tag.models.attention as attention_mod
    from fm4tag.models.attention import _varlen_dq_accum_numel, _varlen_slices

    torch.manual_seed(0)
    lens = torch.randint(0, 41, (500,))
    bounds = [0, *lens.cumsum(0).tolist()]
    assert _varlen_slices(bounds, 4, 64) == [(0, 500)]

    monkeypatch.setattr(attention_mod, 'MAX_KERNEL_BATCH', 120)
    monkeypatch.setattr(attention_mod, 'MAX_VARLEN_DQ_ACCUM', 40 * 168 * 256)
    slices = _varlen_slices(bounds, 4, 64)
    assert slices[0][0] == 0 and slices[-1][1] == 500
    assert all(a < b for a, b in slices)
    assert all(b0 == a1 for (_, b0), (a1, _) in zip(slices, slices[1:]))
    for a, b in slices:
        assert b - a <= 120
        assert (
            _varlen_dq_accum_numel(bounds[b] - bounds[a], b - a, 4, 64)
            <= 40 * 168 * 256
        )
    assert len(slices) > 500 // 120 + 1  # the dq cap, not the group cap, binds


def test_varlen_slices_refuse_one_oversized_group(monkeypatch):
    import fm4tag.models.attention as attention_mod
    from fm4tag.models.attention import _varlen_slices

    monkeypatch.setattr(attention_mod, 'MAX_VARLEN_DQ_ACCUM', 1000)
    with pytest.raises(ValueError, match='too large'):
        _varlen_slices([0, 10, 5000], 1, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
def test_varlen_dq_cap_slices_forward_and_backward_on_gpu(monkeypatch):
    """Slicing for the dq-buffer cap changes neither the output nor the gradients."""
    import fm4tag.models.attention as attention_mod

    torch.manual_seed(0)
    lens = torch.randint(0, 41, (300,))
    groups = RowGroups.from_boundaries(
        torch.nn.functional.pad(lens.cumsum(0), (1, 0)).cuda()
    )
    t = int(lens.sum())
    qkv = torch.randn(3, t, 4, 64, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(t, 4, 64, device='cuda')

    def run():
        q, k, v = (x.clone().requires_grad_() for x in qkv)
        out = grouped_attention(q, k, v, groups)
        (out.float() * w).sum().backward()
        return out.float(), q.grad.float(), k.grad.float(), v.grad.float()

    ref = run()
    monkeypatch.setattr(
        attention_mod, 'MAX_VARLEN_DQ_ACCUM', (t + 128 * 300) * 256 // 3
    )
    got = run()
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, atol=1e-2)


# ── qk_norm / logit_stats ───────────────────────────────────────────────────


def _grown(qk_norm: bool) -> RowAttention:
    """``to_qkv`` scaled up 1000x, like the unstable pretrainings' first layers."""
    attn = _row_attn(qk_norm=qk_norm)
    with torch.no_grad():
        attn.to_qkv.weight.mul_(1000.0)
    return attn


def test_qk_norm_bounds_the_logits():
    x = torch.randn(B, N, DIM)
    top, entropy = _grown(qk_norm=False).logit_stats(x, _groups())
    assert top > 1e4 and entropy < 0.1  # one-hot attention
    top, entropy = _grown(qk_norm=True).logit_stats(x, _groups())
    # Unit-RMS q and k with unit gains: |q·k|/sqrt(d) <= sqrt(d).
    assert top <= 8**0.5 + 1e-4 and entropy > 0.5


def test_qk_norm_off_adds_no_parameters():
    """Old checkpoints (no q_norm/k_norm) still load into qk_norm=False."""
    assert not any('norm' in k for k in _row_attn().state_dict())
    assert {'q_norm.weight', 'k_norm.weight'} <= set(
        _row_attn(qk_norm=True).state_dict()
    )


def test_qk_norm_grouped_matches_per_chunk_calls_and_ungrouped():
    attn = _row_attn(qk_norm=True)
    x = torch.randn(B, N, DIM)
    with torch.no_grad():
        out = attn(x, groups=_groups())
        ref = _per_group(attn, x)
        whole = attn(x, groups=_groups([0, B]))
        ungrouped = attn(x)
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.allclose(whole, ungrouped, atol=1e-5)


@pytest.mark.parametrize('qk_norm', [False, True])
def test_logit_stats_matches_brute_force(qk_norm):
    attn = _grown(qk_norm) if qk_norm else _row_attn()
    x = torch.randn(B, N, DIM)
    top, entropy = attn.logit_stats(x, _groups(), max_groups=10)
    with torch.no_grad():
        q, k, _ = attn.project(x)  # (B, N*h, d)
    tops, ents = [], []
    for s, e in zip(_BOUNDS[:-1], _BOUNDS[1:]):
        if e - s < 2:
            continue
        for h in range(q.shape[1]):
            logits = q[s:e, h] @ k[s:e, h].T / q.shape[-1] ** 0.5
            p = logits.softmax(-1)
            tops.append(logits.abs().max())
            ents.append(-(p * p.log()).sum(-1) / torch.log(torch.tensor(e - s)))
    assert top == pytest.approx(max(tops).item(), rel=1e-5)
    # Mean over every (query, head) of every group.
    assert entropy == pytest.approx(torch.cat(ents).mean().item(), rel=1e-5)


def test_logit_stats_of_uniform_attention():
    attn = _row_attn()
    torch.nn.init.zeros_(attn.to_qkv.weight)
    top, entropy = attn.logit_stats(torch.randn(B, N, DIM), _groups())
    assert top == 0.0 and entropy == pytest.approx(1.0)


# ── LeanRMSNorm ──────────────────────────────────────────────────────────────


def _norms(dim=8):
    torch.manual_seed(0)
    ref = torch.nn.RMSNorm(dim)
    torch.nn.init.normal_(ref.weight, 1.0, 0.3)
    lean = LeanRMSNorm(dim)
    lean.load_state_dict(ref.state_dict())  # same key: weight
    return ref, lean


def test_lean_rms_norm_matches_torch():
    ref, lean = _norms()
    x = (torch.randn(6, 5, 8) * 3).requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    g = torch.randn(6, 5, 8)
    (ref(x) * g).sum().backward()
    (lean(x2) * g).sum().backward()
    assert torch.allclose(lean(x2), ref(x), atol=1e-6)
    assert torch.allclose(x2.grad, x.grad, atol=1e-5)
    assert torch.allclose(lean.weight.grad, ref.weight.grad, atol=1e-5)


def test_lean_rms_norm_gradcheck():
    from fm4tag.models.attention import _RMSNormFn

    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, w: _RMSNormFn.apply(x, w, 1e-6), (x, w))


def test_lean_rms_norm_bf16_matches_fp32_statistics_and_saves_no_fp32_copy():
    ref, lean = _norms()
    x = torch.randn(64, 4, 8).bfloat16().requires_grad_()
    saved = []

    def pack(t):
        saved.append((t.dtype, t.numel()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = lean(x)
    assert y.dtype == torch.bfloat16
    assert torch.equal(y, ref(x.float()).bfloat16())
    # Only x itself at full size; fp32 tensors are the gain and the per-row rstd.
    assert (torch.bfloat16, x.numel()) in saved
    assert all(n < x.numel() for dt, n in saved if dt == torch.float32)
