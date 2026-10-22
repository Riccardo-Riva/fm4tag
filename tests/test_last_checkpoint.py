"""``last.ckpt`` follows training to the end, past the last val/loss improvement."""

from __future__ import annotations

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from fm4tag.configs import __path__ as cfg_path
from fm4tag.utils import instantiate_callbacks


class _Worsening(L.LightningModule):
    """val/loss is best at epoch 0 and gets worse every epoch after it."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        return self.net(x).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        self.log('val/loss', float(self.current_epoch))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_last_ckpt_is_the_final_epoch(tmp_path):
    with initialize_config_dir(config_dir=str(cfg_path[0]), version_base=None):
        cfg = compose(config_name='campaign_ssl', overrides=['arm=pretrain'])
    with open_dict(cfg):
        for key in ('model_checkpoint', 'last_checkpoint'):
            cfg.callbacks[key].dirpath = str(tmp_path)
        callbacks = OmegaConf.masked_copy(
            cfg.callbacks, ['model_checkpoint', 'last_checkpoint']
        )
    data = torch.utils.data.TensorDataset(torch.randn(16, 4))
    loader = torch.utils.data.DataLoader(data, batch_size=8)
    trainer = L.Trainer(
        accelerator='cpu',
        max_epochs=5,
        logger=False,
        callbacks=instantiate_callbacks(callbacks),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(_Worsening(), loader, loader)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert 'last.ckpt' in files and not any(f.startswith('last-v') for f in files)
    assert torch.load(tmp_path / 'last.ckpt', weights_only=False)['epoch'] == 4
    # The val/loss top-3 still holds epochs 0-2, and stays the trainer's first
    # checkpoint callback (hpo.py reads its best_model_path).
    assert len([f for f in files if f.startswith('epoch')]) == 3
    assert trainer.checkpoint_callback.monitor == 'val/loss'
    assert trainer.checkpoint_callback.best_model_path.startswith(
        str(tmp_path / 'epoch000')
    )
