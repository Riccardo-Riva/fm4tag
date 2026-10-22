"""RowAttentionMonitor logs each row-attention layer's max logit and entropy."""

from __future__ import annotations

import csv

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger

from fm4tag.callbacks import RowAttentionMonitor
from fm4tag.models.attention import RowGroups
from fm4tag.models.backbones.transformers import RowTransformer


class _Tracks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                RowTransformer(
                    8, 2, depth=2, row_heads=2, dim_row_head=8, row_qk_norm=True
                )
            ]
        )


class _Tiny(L.LightningModule):
    """Named like the real modules: ``encoders.<obj>.layers.<l>.blocks.<b>``."""

    def __init__(self):
        super().__init__()
        self.encoders = torch.nn.ModuleDict({'tracks': _Tracks()})

    def _loss(self, batch):
        (x,) = batch
        groups = RowGroups.from_boundaries(torch.tensor([0, 4, x.shape[0]]))
        for layer in self.encoders['tracks'].layers:
            x = layer(x, groups=groups)
        return x.pow(2).mean()

    def training_step(self, batch, batch_idx):
        return self._loss(batch)

    def validation_step(self, batch, batch_idx):
        self.log('val/loss', self._loss(batch))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_logs_max_logit_and_entropy_per_layer(tmp_path):
    torch.manual_seed(0)
    data = torch.utils.data.TensorDataset(torch.randn(24, 2, 8))
    loader = torch.utils.data.DataLoader(data, batch_size=8)
    trainer = L.Trainer(
        accelerator='cpu',
        max_epochs=2,
        logger=CSVLogger(tmp_path),
        callbacks=[RowAttentionMonitor()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(_Tiny(), loader, loader)
    with open(next(tmp_path.rglob('metrics.csv'))) as f:
        rows = [r for r in csv.DictReader(f) if r.get('val/loss')]
    assert len(rows) == 2  # one per epoch, none from the sanity check
    for r in rows:
        for block in ('L0b0', 'L0b1'):
            top = float(r[f'val/row_attn/tracks/{block}/max_logit'])
            entropy = float(r[f'val/row_attn/tracks/{block}/entropy'])
            assert 0.0 <= top <= 8**0.5 + 1e-4  # qk_norm bound, unit gains
            assert 0.0 < entropy <= 1.0 + 1e-6
