import torch.nn.functional as F
from torch import nn


class PreNormResidual(nn.Module):
    """Pre-norm sublayer with the residual on the *un-normalised* input::

        x + fn(norm(x))

    The ordering matters.  This used to be spelled ``PreNorm(dim, Residual(fn))``,
    which expands to ``fn(norm(x)) + norm(x)`` — the skip connection runs through
    the LayerNorm, so the stack has no clean identity path and the residual stream
    is renormalised at every sublayer.  (The bug is inherited from SAINT; tabicl,
    and pre-norm transformers generally, use ``x + fn(norm(x))``.)  Keeping the
    composition in one module stops it being re-inverted at a call site.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)
