#!/usr/bin/env python
"""Lightning DDP device-count invariance check for ``MultiViewSupConLoss``.

Runs the **same** global batch through a toy contrastive ``LightningModule``
under CPU/gloo DDP with ``devices = 1, 2, 3, 4`` and asserts that both the
loss **value** and the back-propagated **gradient** are identical regardless
of the device count.

This is the practical, end-to-end counterpart to
``tests/ddp/test_supcon_gradient.py``: it exercises the differentiable
all-gather through Lightning's real DDP machinery (auto ``DistributedSampler``
sharding + gradient averaging) rather than a hand-rolled DDP wrapper.

Why the invariance must hold
----------------------------
* The toy dataset is exactly **one** global batch of ``GLOBAL_BS=12`` samples
  (divisible by 1/2/3/4), ``shuffle=False``, ``drop_last=False``, a single
  optimiser step (``limit_train_batches=1``, ``max_epochs=1``).  Lightning's
  ``DistributedSampler`` shards those 12 samples across the ``n`` devices and
  the differentiable gather inside ``MultiViewSupConLoss`` reassembles them, so
  every device count contrasts the identical 12 samples.
* The encoder is seeded in ``__init__`` (so it is identical across configs;
  ``ddp_spawn`` pickles it from the main process) and the ``V`` views are
  deterministic (``x * scale``), so there is **no RNG** anywhere in the forward
  pass.  ``lr=0`` keeps the single step a no-op (we capture grads before it).

Exit code 0 if every device count agrees with the ``devices=1`` reference
(``atol=1e-5``, ``rtol=1e-4``), otherwise 1.
"""

from __future__ import annotations

import os
import sys
import tempfile

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running as a plain script from the repo root even if fm4tag is not
# installed (it normally is, as an editable install).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fm4tag.losses import MultiViewSupConLoss  # noqa: E402


GLOBAL_BS = 12  # divisible by 1, 2, 3, 4 -> no DistributedSampler padding
F_IN = 6
D_OUT = 5
V = 3
TEMP = 0.1
VIEW_SCALES = [1.0, 1.3, 1.6]
DEVICE_COUNTS = [1, 2, 3, 4]
DATA_SEED = 7
ENC_SEED = 123


def _make_dataset() -> TensorDataset:
    """One fixed global batch, identical across configs and ranks."""
    g = torch.Generator().manual_seed(DATA_SEED)
    x = torch.randn(GLOBAL_BS, F_IN, generator=g)
    return TensorDataset(x)


class ToyContrastive(L.LightningModule):
    """Minimal contrastive module: linear encoder + multi-view SupCon."""

    def __init__(self, dump_path: str) -> None:
        super().__init__()
        # Seeded here so the initial weights are identical across every config
        # (ddp_spawn pickles this module from the main process to the workers).
        torch.manual_seed(ENC_SEED)
        self.enc = nn.Linear(F_IN, D_OUT, bias=False)
        self.loss_fn = MultiViewSupConLoss(
            temperature=TEMP, loss_type='out', include_pos_in_denom=True
        )
        self.dump_path = dump_path
        self._last_loss: torch.Tensor | None = None

    def _embed(self, x: torch.Tensor) -> list[torch.Tensor]:
        """V deterministic views via one forward pass (no RNG)."""
        n = x.size(0)
        x_views = torch.cat([x * s for s in VIEW_SCALES], dim=0)  # (V*n, F)
        h = self.enc(x_views)  # (V*n, D)
        return list(h.split(n, dim=0))  # V x (n, D)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        loss = self.loss_fn(self._embed(x))
        self._last_loss = loss.detach()
        return loss

    def on_after_backward(self) -> None:
        # Grads are DDP-averaged (and identical across ranks) by this hook.
        # all_gather is a collective: call it on EVERY rank before guarding on
        # rank 0.  Mean over the per-rank local losses == the full-batch loss.
        full_loss = self.all_gather(self._last_loss).float().mean()
        grad = self.enc.weight.grad.detach().flatten().clone()
        if self.trainer.is_global_zero:
            torch.save({'loss': full_loss.cpu(), 'grad': grad.cpu()}, self.dump_path)

    def configure_optimizers(self):
        # lr=0: the single optimiser step is a no-op (we capture grads before it).
        return torch.optim.SGD(self.parameters(), lr=0.0)


def _run_one(n: int, dump_path: str) -> None:
    model = ToyContrastive(dump_path)
    loader = DataLoader(
        _make_dataset(), batch_size=GLOBAL_BS, shuffle=False, drop_last=False
    )
    trainer = L.Trainer(
        accelerator='cpu',
        devices=n,
        strategy='ddp_spawn' if n > 1 else 'auto',
        max_epochs=1,
        limit_train_batches=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)


def main() -> int:
    tmpdir = tempfile.mkdtemp(prefix='ddp_loss_invariance_')
    dumps: dict[int, dict] = {}
    for n in DEVICE_COUNTS:
        path = os.path.join(tmpdir, f'dump_{n}.pt')
        _run_one(n, path)
        dumps[n] = torch.load(path)

    ref = dumps[DEVICE_COUNTS[0]]
    ref_loss, ref_grad = ref['loss'], ref['grad']

    header = (
        f'{"devices":>8} | {"loss":>14} | {"max|d-loss|":>12} | {"max|d-grad|":>12}'
    )
    print(header)
    print('-' * len(header))
    ok = True
    for n in DEVICE_COUNTS:
        loss, grad = dumps[n]['loss'], dumps[n]['grad']
        dloss = (loss - ref_loss).abs().max().item()
        dgrad = (grad - ref_grad).abs().max().item()
        loss_ok = torch.allclose(loss, ref_loss, atol=1e-5, rtol=1e-4)
        grad_ok = torch.allclose(grad, ref_grad, atol=1e-5, rtol=1e-4)
        ok = ok and loss_ok and grad_ok
        flag = '' if (loss_ok and grad_ok) else '   <-- MISMATCH'
        print(f'{n:>8} | {loss.item():>14.8f} | {dloss:>12.3e} | {dgrad:>12.3e}{flag}')

    print('-' * len(header))
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
