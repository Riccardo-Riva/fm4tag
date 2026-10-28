import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.varlen import varlen_attn
from einops import rearrange

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False

_FLASH_ATTN_DTYPES = (torch.float16, torch.bfloat16)

# CUDA caps grid dims y/z at 65535 and the fused attention kernels put the batch
# (or, for varlen, the number of sequences) there: a launch beyond it fails
# with "invalid configuration argument".  The column attention batches every
# packed track (~118k at 16k jets), so such calls are split into slices.
MAX_KERNEL_BATCH = 65535

# flash's varlen backward accumulates dq in an fp32 buffer of (T + 128·K) rows
# of H·round32(D) floats (T packed samples, K sequences) and indexes it with
# 32-bit offsets: from 2**31 elements on it writes out of bounds — an illegal
# memory access, or silently corrupted gradients (the frozen arm at 94208 jets
# went NaN in its first epoch).  Measured on an A40, torch 2.10: H·D = 4·64
# faults between 62000 and 65024 aggregator sequences, 8·64 between 30000 and
# 34000, exactly where this count crosses 2**31.
MAX_VARLEN_DQ_ACCUM = 2**31 - 1


def sdpa(q, k, v, attn_mask=None, dropout_p=0.0):
    """Scaled dot-product attention, routed through FlashAttention-2 when possible.

    ``q``/``k``/``v``: ``(..., h, seq, d)`` — any number of leading batch-like
    dims (one for :class:`Attention`, ``n`` tokens for :class:`RowAttention`).
    Falls back to ``F.scaled_dot_product_attention`` whenever an explicit
    ``attn_mask`` is given (flash-attn doesn't support arbitrary attention
    masks), when ``flash-attn`` isn't installed, or off-CUDA/unsupported dtype.

    Do not route the small column attentions (``seq=19``, ``head_dim=8``) onto
    the MATH backend to dodge the padding the fused kernels apply: measured
    2026-08-17, it does cut their backward ~20%, but triples their forward
    (unaligned GEMMs + fp32 scores) for a net 2-3% loss and +0.4 GiB.

    A leading batch above :data:`MAX_KERNEL_BATCH` is attended in slices.
    """
    if q.dim() >= 4 and q.shape[0] > MAX_KERNEL_BATCH:
        n = MAX_KERNEL_BATCH
        per_sample_mask = attn_mask is not None and attn_mask.shape[0] == q.shape[0]
        return torch.cat(
            [
                sdpa(
                    q[i : i + n],
                    k[i : i + n],
                    v[i : i + n],
                    attn_mask=attn_mask[i : i + n] if per_sample_mask else attn_mask,
                    dropout_p=dropout_p,
                )
                for i in range(0, q.shape[0], n)
            ]
        )
    if (
        HAS_FLASH_ATTN
        and attn_mask is None
        and q.is_cuda
        and q.dtype in _FLASH_ATTN_DTYPES
    ):
        orig_shape = q.shape
        qf = q.reshape(-1, *q.shape[-3:]).transpose(1, 2).contiguous()
        kf = k.reshape(-1, *k.shape[-3:]).transpose(1, 2).contiguous()
        vf = v.reshape(-1, *v.shape[-3:]).transpose(1, 2).contiguous()
        out = flash_attn_func(qf, kf, vf, dropout_p=dropout_p)
        return out.transpose(1, 2).reshape(orig_shape)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )


class Attention(nn.Module):
    """Scaled dot-product self-attention over the token dimension of a sample.

    Attends across the ``n`` tokens of each sample independently.  The optional
    mask is a per-token key-padding mask, so padded positions can be excluded
    individually (used e.g. for cross-constituent attention within a jet).

    With ``groups`` the input is instead a packed ``(T, dim)`` set of tokens
    and each group (one sample's tokens, e.g. one jet's valid tracks) attends
    within itself — no padding, no mask (flash varlen on GPU).

    Args:
        x:      ``(b, n, dim)`` — batch of token sequences, or ``(T, dim)``
                with ``groups``.
        mask:   ``(b, n)`` bool — ``True`` = valid token, ``False`` = padding.
        groups: Optional :class:`RowGroups` over the packed ``T`` axis.
    """

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x, mask=None, groups=None):
        h = self.heads

        if groups is not None:
            if mask is not None:
                raise ValueError('Attention: pass either mask or groups, not both.')
            t = x.shape[0]
            q, k, v = (u.reshape(t, h, -1) for u in self.to_qkv(x).chunk(3, dim=-1))
            dropout_p = self.dropout if self.training else 0.0
            out = grouped_attention(q, k, v, groups, dropout_p=dropout_p)
            return self.to_out(out.reshape(t, -1))

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :]
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))

        out = sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


@dataclass(frozen=True)
class RowGroups:
    """Contiguous groups of samples that attend only among themselves.

    ``cu_seqlens`` ``(K+1,)`` int32 holds the group boundaries on the sample
    axis (``0 = cu[0] <= ... <= cu[K] = n_samples``, empty groups allowed);
    ``max_seqlen`` is the longest group, which flash's varlen kernel needs on
    the host.  Build it once per batch and reuse it across views and layers.
    """

    cu_seqlens: torch.Tensor
    max_seqlen: int

    @classmethod
    def from_boundaries(cls, cu_seqlens: torch.Tensor) -> 'RowGroups':
        cu = cu_seqlens.to(torch.int32)
        longest = int((cu[1:] - cu[:-1]).max()) if cu.numel() > 1 else 0  # host sync
        return cls(cu, longest)


def _use_varlen(q: torch.Tensor, dropout_p: float) -> bool:
    # varlen_attn: CUDA, fp16/bf16, head_dim % 8 == 0 and <= 256, no dropout.
    d = q.shape[-1]
    return (
        q.is_cuda
        and q.dtype in _FLASH_ATTN_DTYPES
        and d % 8 == 0
        and d <= 256
        and dropout_p == 0.0
    )


def _varlen_dq_accum_numel(
    n_samples: int, n_groups: int, heads: int, head_dim: int
) -> int:
    return (n_samples + 128 * n_groups) * heads * (-(-head_dim // 32) * 32)


def _varlen_slices(
    bounds: list[int], heads: int, head_dim: int
) -> list[tuple[int, int]]:
    """Consecutive group ranges ``[a, b)`` that one varlen launch each can take.

    Every range has at most :data:`MAX_KERNEL_BATCH` groups and a dq
    accumulator of at most :data:`MAX_VARLEN_DQ_ACCUM` elements.
    """
    n_groups = len(bounds) - 1
    slices = []
    a = 0
    while a < n_groups:
        b = min(a + MAX_KERNEL_BATCH, n_groups)
        while (
            numel := _varlen_dq_accum_numel(
                bounds[b] - bounds[a], b - a, heads, head_dim
            )
        ) > MAX_VARLEN_DQ_ACCUM:
            if b - a == 1:
                raise ValueError(
                    f'an attention group of {bounds[b] - bounds[a]} samples is '
                    'too large for one varlen launch'
                )
            b = a + max(1, (b - a) * MAX_VARLEN_DQ_ACCUM // numel)
        slices.append((a, b))
        a = b
    return slices


def grouped_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    groups: RowGroups,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Self-attention within each group of the packed sample axis.

    ``q``/``k``/``v``: ``(T, H, D)`` — ``T`` packed samples, ``H`` independent
    heads.  Sample ``i`` attends only to the samples of its own group.  Runs
    flash's varlen kernel when eligible, otherwise an exact loop over the
    groups (CPU tests, fp32 offline inference, attention dropout).
    """
    if q.shape[0] == 0:
        return q.clone()
    cu = groups.cu_seqlens
    if _use_varlen(q, dropout_p):
        n_groups = cu.numel() - 1
        h, d = q.shape[1], q.shape[2]
        if (
            n_groups <= MAX_KERNEL_BATCH
            and _varlen_dq_accum_numel(q.shape[0], n_groups, h, d)
            <= MAX_VARLEN_DQ_ACCUM
        ):
            return varlen_attn(q, k, v, cu, cu, groups.max_seqlen, groups.max_seqlen)
        # More sequences, or a larger dq buffer, than one launch takes: slices
        # of whole groups.
        bounds = cu.tolist()
        outs = []
        for a, b in _varlen_slices(bounds, h, d):
            lo, hi = bounds[a], bounds[b]
            sub = RowGroups(cu[a : b + 1] - lo, groups.max_seqlen)
            outs.append(grouped_attention(q[lo:hi], k[lo:hi], v[lo:hi], sub, dropout_p))
        return torch.cat(outs, dim=0)

    bounds = cu.tolist()
    outs = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end == start:
            continue
        # (L, H, D) → (H, L, D): heads as batch, the group as the sequence.
        o = sdpa(
            q[start:end].transpose(0, 1),
            k[start:end].transpose(0, 1),
            v[start:end].transpose(0, 1),
            dropout_p=dropout_p,
        )
        outs.append(o.transpose(0, 1))
    return torch.cat(outs, dim=0)


class _RMSNormFn(torch.autograd.Function):
    """RMSNorm over the last dim: fp32 statistics, output in the input's dtype.

    Saves only the input (in its own dtype) and the per-row inverse RMS, and
    recomputes the rest in backward.  ``nn.RMSNorm`` on an upcast input saves
    an fp32 copy instead: for bf16 q and k that cost P1 ~2 GiB and its batch.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        dt = torch.promote_types(x.dtype, torch.float32)
        xf = x.to(dt)
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        ctx.save_for_backward(x, weight, rstd)
        return (xf * rstd * weight.to(dt)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, weight, rstd = ctx.saved_tensors
        dt = rstd.dtype
        xhat = x.to(dt) * rstd
        g = grad.to(dt)
        grad_w = (g * xhat).reshape(-1, x.shape[-1]).sum(0)
        g_xhat = g * weight.to(dt)
        grad_x = rstd * (g_xhat - xhat * (g_xhat * xhat).mean(-1, keepdim=True))
        return grad_x.to(x.dtype), grad_w.to(weight.dtype), None


class LeanRMSNorm(nn.Module):
    """``nn.RMSNorm`` (learnable gain, fp32 eps) with :class:`_RMSNormFn`'s memory."""

    def __init__(self, dim: int, eps: float | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = torch.finfo(torch.float32).eps if eps is None else eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNormFn.apply(x, self.weight, self.eps)


class RowAttention(nn.Module):
    """All-pairs intersample (row) attention over the batch axis.

    Every sample attends to every other sample in the batch, independently for
    each of the ``N`` token positions and at width ``dim`` — the token axis is
    a batch dimension, so one attention operator is shared across tokens.  Cost
    is O(B²) in the batch size.

    Consequence: outputs (also at inference) depend on batch composition.

    ``to_out`` is zero-initialised, so under the surrounding pre-norm residual
    the row step starts as the identity and learns intersample mixing from
    there.

    With ``groups`` (:class:`RowGroups`) the batch axis is split into
    contiguous chunks and each sample attends only within its own chunk; the
    ``N`` tokens × ``heads`` become independent heads of one varlen call.

    ``qk_norm`` RMS-normalises queries and keys per head (learnable gain)
    before the dot product, which bounds the logits.  Without it, contrastive
    pretraining grew ``to_qkv`` until the logits reached 1e5-1e7: one-hot
    attention, and gradient blowups once bf16 rounding flipped the argmax.

    Args:
        x:      ``(B, N, dim)`` — token grid.
        mask:   ``(B,)`` bool — ``True`` = valid sample, ``False`` = padding.
                Only without ``groups``.
        groups: Optional :class:`RowGroups` over the ``B`` axis.
    """

    def __init__(self, dim, heads=8, dim_row_head=16, dropout=0.0, qk_norm=False):
        super().__init__()
        inner_dim = dim_row_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)
        self.dropout = dropout
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = LeanRMSNorm(dim_row_head)
            self.k_norm = LeanRMSNorm(dim_row_head)

    def project(self, x):
        """``(B, N, dim)`` → ``q``, ``k``, ``v`` of shape ``(B, N*h, d)``.

        The ``N`` tokens and ``h`` heads are all independent heads attending
        over the sample axis.
        """
        b, n, _ = x.shape
        q, k, v = (
            t.reshape(b, n * self.heads, -1) for t in self.to_qkv(x).chunk(3, dim=-1)
        )
        if self.qk_norm:
            # fp32 statistics, output (and saved input) in the kernel's dtype.
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k, v

    @torch.no_grad()
    def logit_stats(self, x, groups=None, max_groups=4, head_slice=8):
        """Largest ``|q·k|/sqrt(d)`` and mean normalised attention entropy.

        Entropy is divided by ``log(group size)``: 1 = uniform, 0 = one-hot;
        the mean is over every (query, head) of the first ``max_groups``
        groups (the whole batch as one group without ``groups``), computed
        ``head_slice`` heads at a time.
        """
        q, k, _ = self.project(x)
        n_samples = q.shape[0]
        bounds = [0, n_samples] if groups is None else groups.cu_seqlens.tolist()
        top, ent_sum, n_ent = 0.0, 0.0, 0
        # fp32 logits even inside a bf16-autocast forward.
        with torch.autocast(q.device.type, enabled=False):
            for s, e in list(zip(bounds[:-1], bounds[1:]))[:max_groups]:
                if e - s < 2:
                    continue
                for h0 in range(0, q.shape[1], head_slice):
                    qs = q[s:e, h0 : h0 + head_slice].float()
                    ks = k[s:e, h0 : h0 + head_slice].float()
                    logits = torch.einsum('lhd,mhd->hlm', qs, ks) * qs.shape[-1] ** -0.5
                    top = max(top, logits.abs().amax().item())
                    p = logits.softmax(-1)
                    ent = -(p * p.clamp_min(1e-30).log()).sum(-1) / math.log(e - s)
                    ent_sum += ent.sum().item()
                    n_ent += ent.numel()
        return top, (ent_sum / n_ent if n_ent else float('nan'))

    def forward(self, x, mask=None, groups=None):
        h = self.heads
        dropout_p = self.dropout if self.training else 0.0
        b, n, _ = x.shape
        q, k, v = self.project(x)

        if groups is not None:
            if mask is not None:
                raise ValueError('RowAttention: pass either mask or groups, not both.')
            out = grouped_attention(q, k, v, groups, dropout_p=dropout_p)
            return self.to_out(out.reshape(b, n, -1))

        # (B, N*h, d) → (N, h, B, d): tokens become a batch dim, the sample axis
        # B becomes the attention sequence.
        q, k, v = (t.reshape(b, n, h, -1).permute(1, 2, 0, 3) for t in (q, k, v))

        attn_mask = None
        if mask is not None:
            # Key-padding mask over the sample axis: (B,) → (1, 1, 1, B).
            attn_mask = torch.where(mask, 0.0, float('-inf'))[None, None, None, :]

        out = sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        out = rearrange(out, 'n h b d -> b n (h d)')
        return self.to_out(out)
