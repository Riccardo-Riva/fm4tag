"""GradNormMonitorCallback logs the gradient norm before clipping."""

from __future__ import annotations

import csv

import lightning as L
import pytest
import torch
from lightning.pytorch.loggers import CSVLogger

from fm4tag.callbacks import GradNormMonitorCallback


class _Tiny(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        return self.net(x).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _fit(tmp_path, clip):
    torch.manual_seed(0)
    data = torch.utils.data.TensorDataset(torch.randn(48, 4) * 10)
    trainer = L.Trainer(
        accelerator='cpu',
        max_steps=6,
        log_every_n_steps=2,
        gradient_clip_val=clip,
        logger=CSVLogger(tmp_path),
        callbacks=[GradNormMonitorCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(_Tiny(), torch.utils.data.DataLoader(data, batch_size=8))
    with open(next(tmp_path.rglob('metrics.csv'))) as f:
        return [r for r in csv.DictReader(f) if r.get('train/grad_norm')]


def test_logs_on_lightning_write_steps_before_clipping(tmp_path):
    rows = _fit(tmp_path, clip=1e-6)
    assert [int(r['step']) for r in rows] == [1, 3, 5]
    for r in rows:
        # A clipped norm would be <= 1e-6: this is the norm before clipping.
        assert float(r['train/grad_norm']) > 1e-3
        assert float(r['train/grad_norm_max']) >= float(r['train/grad_norm'])
        assert float(r['train/grad_clip_frac']) == pytest.approx(1.0)


def test_no_clip_fraction_without_clipping(tmp_path):
    rows = _fit(tmp_path, clip=None)
    assert rows and all(not r.get('train/grad_clip_frac') for r in rows)
