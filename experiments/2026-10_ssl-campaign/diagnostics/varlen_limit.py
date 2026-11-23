"""Largest number of sequences one varlen_attn backward handles (aggregator-like shapes).

One configuration per process (a fault poisons the CUDA context):

  python varlen_limit.py <n_groups> <heads> <head_dim> [mean_len]

Prints max |dq| difference between one launch and the same groups in two halves, or FAULT.
"""
import sys

import torch
from torch.nn.attention.varlen import varlen_attn

n, h, d = (int(a) for a in sys.argv[1:4])
mean = float(sys.argv[4]) if len(sys.argv) > 4 else 7.2
g = torch.Generator().manual_seed(0)
lens = torch.poisson(torch.full((n,), mean), generator=g).clamp(0, 40).long()
cu = torch.nn.functional.pad(lens.cumsum(0), (1, 0)).int().cuda()
T = int(cu[-1])
q, k, v = (torch.randn(T, h, d, device='cuda', dtype=torch.bfloat16, generator=None) for _ in range(3))
w = torch.randn(T, h, d, device='cuda')
rows = (T + 128 * n) * h * ((d + 31) // 32 * 32)


def grads(slices):
    qs, ks, vs = (t.detach().clone().requires_grad_() for t in (q, k, v))
    outs = []
    for a, b in slices:
        lo, hi = int(cu[a]), int(cu[b])
        outs.append(varlen_attn(qs[lo:hi], ks[lo:hi], vs[lo:hi], cu[a:b + 1] - lo, cu[a:b + 1] - lo, 40, 40))
    (torch.cat(outs).float() * w).sum().backward()
    return qs.grad.float()


try:
    ref = grads([(0, n // 2), (n // 2, n)])
    full = grads([(0, n)])
    torch.cuda.synchronize()
    diff = (full - ref).abs().max().item()
    print(f'n={n} h={h} d={d} T={T} (T+128n)*h*d_r={rows / 2**31:.3f}*2^31  max|ddq|={diff:.3g}  finite={torch.isfinite(full).all().item()}')
except Exception as e:  # noqa: BLE001
    print(f'n={n} h={h} d={d} T={T} (T+128n)*h*d_r={rows / 2**31:.3f}*2^31  FAULT {type(e).__name__}')
