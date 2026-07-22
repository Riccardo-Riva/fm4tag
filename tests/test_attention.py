"""Tests for the intersample (row) attention variants.

Both variants take the token grid ``(B, N, dim)`` and attend over the **sample**
axis ``B``, independently per token.  :class:`RowAttention` does so all-pairs;
:class:`InducedRowAttention` routes through learned inducing points (ISAB).
"""

from __future__ import annotations

import pytest
import torch

from fm4tag.models.attention import InducedRowAttention, RowAttention
from fm4tag.models.backbones.transformers import RowMixer

B, N, DIM = 10, 5, 16


def _row_attn(heads: int = 4) -> RowAttention:
    torch.manual_seed(0)
    return RowAttention(dim=DIM, heads=heads, dim_row_head=8).eval()


def _induced(heads: int = 4, num_inds: int = 6) -> InducedRowAttention:
    torch.manual_seed(0)
    attn = InducedRowAttention(
        dim=DIM, nfeats=N, heads=heads, dim_row_head=8, num_inds=num_inds
    ).eval()
    # ``attn_out.to_out`` ships zero-initialised (identity-at-init under the
    # surrounding residual), which would make every behavioural test below
    # trivially true.  Give it real weights.
    torch.nn.init.normal_(attn.attn_out.to_out.weight, std=0.1)
    torch.nn.init.normal_(attn.attn_out.to_out.bias, std=0.1)
    return attn


def _both():
    return [_row_attn(), _induced()]


# ── shape / mask handling ────────────────────────────────────────────────────


@pytest.mark.parametrize('attn', _both())
def test_shape_roundtrip(attn):
    out = attn(torch.randn(B, N, DIM))
    assert out.shape == (B, N, DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('attn', _both())
def test_mask_shape_broadcasts(attn):
    """Regression: the per-sample (B,) mask must work when B != heads."""
    x = torch.randn(B, N, DIM)
    mask = torch.ones(B, dtype=torch.bool)
    mask[7:] = False
    out = attn(x, mask=mask)
    assert out.shape == (B, N, DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('attn', _both())
def test_masked_samples_are_ignored_as_keys(attn):
    """Garbage in masked-out samples must not change the valid samples' outputs."""
    x = torch.randn(B, N, DIM)
    mask = torch.ones(B, dtype=torch.bool)
    mask[7:] = False

    x_garbage = x.clone()
    x_garbage[7:] = 1e4

    with torch.no_grad():
        out = attn(x, mask=mask)
        out_garbage = attn(x_garbage, mask=mask)

    assert torch.allclose(out[:7], out_garbage[:7], atol=1e-4)


@pytest.mark.parametrize('attn', _both())
def test_all_valid_mask_matches_no_mask(attn):
    x = torch.randn(B, N, DIM)
    with torch.no_grad():
        out_none = attn(x)
        out_all = attn(x, mask=torch.ones(B, dtype=torch.bool))
    assert torch.allclose(out_none, out_all, atol=1e-6)


# ── the tabicl factorisation: row attention acts per token ───────────────────


@pytest.mark.parametrize('attn', _both())
def test_token_channels_are_independent(attn):
    """Output at token ``n`` must depend only on the input at token ``n``.

    Row attention runs over the sample axis independently per token — mixing
    across tokens is the column attention's job.  Perturbing a different token
    must therefore leave this token's output untouched.
    """
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


@pytest.mark.parametrize('attn', _both())
def test_permutation_equivariant_over_samples(attn):
    """Reordering the batch must reorder the outputs identically.

    This is what the old chunked row attention could not offer: it drew a fresh
    random permutation every training step, so a sample's output depended on
    which 512 others happened to land in its chunk.
    """
    x = torch.randn(B, N, DIM)
    perm = torch.randperm(B)

    with torch.no_grad():
        out = attn(x)
        out_perm = attn(x[perm])

    assert torch.allclose(out[perm], out_perm, atol=1e-5)


@pytest.mark.parametrize('attn', _both())
def test_deterministic_in_train_mode(attn):
    """No stochastic chunking: train mode must be as deterministic as eval."""
    attn.train()
    x = torch.randn(B, N, DIM)
    with torch.no_grad():
        assert torch.equal(attn(x), attn(x))


# ── ISAB-specific ────────────────────────────────────────────────────────────


def test_induced_starts_as_identity():
    """As shipped, the zero-init out-projection makes the block a no-op.

    Under ``PreNormResidual`` in ``RowMixer`` that means the row step starts as
    the identity and learns intersample mixing from there.
    """
    torch.manual_seed(0)
    attn = InducedRowAttention(dim=DIM, nfeats=N, heads=4, dim_row_head=8).eval()
    with torch.no_grad():
        out = attn(torch.randn(B, N, DIM))
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize('batch', [1, 7, 64])
def test_induced_handles_variable_batch_size(batch):
    """A variable B — the packed constituent count — needs no padding or chunking."""
    attn = _induced()
    with torch.no_grad():
        out = attn(torch.randn(batch, N, DIM))
    assert out.shape == (batch, N, DIM)
    assert torch.isfinite(out).all()


def test_induced_cost_is_independent_of_batch_size():
    """The inducing points, not B, set the attention's key length."""
    attn = _induced(num_inds=6)
    assert attn.inds.shape == (N, 6, DIM)


def test_induced_gradients_reach_inducing_points():
    attn = _induced()
    out = attn(torch.randn(B, N, DIM))
    out.sum().backward()
    assert attn.inds.grad is not None
    assert torch.count_nonzero(attn.inds.grad) > 0


# ── RowMixer: per_token vs concat row_mode ───────────────────────────────────


def _row_mixer(row_mode: str, num_inds: int | None = 6) -> RowMixer:
    torch.manual_seed(0)
    return RowMixer(
        DIM,
        N,
        heads=2,
        dim_row_head=8,
        ff_mult=1,
        attn_dropout=0.0,
        ff_dropout=0.0,
        num_inds=num_inds,
        row_mode=row_mode,
    ).eval()


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
@pytest.mark.parametrize('num_inds', [6, None])
def test_row_mixer_shape_roundtrips(row_mode, num_inds):
    """Both modes and both num_inds settings return the (B, N, dim) grid."""
    mixer = _row_mixer(row_mode, num_inds)
    out = mixer(torch.randn(B, N, DIM))
    assert out.shape == (B, N, DIM)
    assert torch.isfinite(out).all()


def test_row_mixer_rejects_unknown_mode():
    with pytest.raises(ValueError, match='row_mode'):
        _row_mixer('sideways')


def test_concat_row_attention_runs_at_row_dim():
    """In concat mode the row attention operates on the flattened N*dim vector."""
    mixer = _row_mixer('concat', num_inds=6)
    # attn.fn is the InducedRowAttention; its inducing points are a single set
    # (nfeats == 1) of width row_dim = N*dim, not N sets of width dim.
    assert mixer.attn.fn.inds.shape == (1, 6, N * DIM)


def test_per_token_row_attention_runs_per_token():
    mixer = _row_mixer('per_token', num_inds=6)
    assert mixer.attn.fn.inds.shape == (N, 6, DIM)


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_row_mixer_attn_sublayer_is_identity_at_init(row_mode):
    """Zero-init ISAB out-projection ⇒ the row-attention sublayer starts as a no-op."""
    mixer = _row_mixer(row_mode, num_inds=6)
    x = torch.randn(B, N, DIM)
    if row_mode == 'concat':
        x = x.reshape(B, 1, N * DIM)
    with torch.no_grad():
        assert torch.allclose(mixer.attn(x), x, atol=1e-6)


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_row_mixer_permutation_equivariant_over_samples(row_mode):
    """Reordering the batch reorders the outputs identically, in either mode."""
    mixer = _row_mixer(row_mode, num_inds=6)
    # Un-zero the ISAB out-projection so the block is not a trivial identity.
    torch.nn.init.normal_(mixer.attn.fn.attn_out.to_out.weight, std=0.1)
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

    per_token = _row_mixer('per_token', num_inds=6)
    torch.nn.init.normal_(per_token.attn.fn.attn_out.to_out.weight, std=0.1)
    with torch.no_grad():
        a, b = per_token(x), per_token(x_pert)
    assert torch.allclose(a[:, others], b[:, others], atol=1e-5)

    concat = _row_mixer('concat', num_inds=6)
    torch.nn.init.normal_(concat.attn.fn.attn_out.to_out.weight, std=0.1)
    torch.nn.init.normal_(concat.ff.fn.net[0].weight, std=0.1)
    with torch.no_grad():
        a, b = concat(x), concat(x_pert)
    assert not torch.allclose(a[:, others], b[:, others], atol=1e-3)


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
                'num_inds': 4,
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
