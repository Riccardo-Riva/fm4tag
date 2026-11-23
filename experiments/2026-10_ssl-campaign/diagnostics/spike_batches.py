"""Extremes of the continuous track features in the dumped spike batches vs ordinary batches."""
import glob
import sys

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load(glob.glob('experiments/2026-10_ssl-campaign/diagnostics/outputs/repro/p0_s42/outputs/*/*/.hydra/config.yaml')[0])
names = list(cfg.variables.tracks.continuous) if 'continuous' in cfg.variables.tracks else None


def stats(batch):
    t = batch['constituents']['tracks']
    x = t['continuous'][t['valid']]  # (N_valid, F)
    return x.abs().amax(0), (x.abs() > 20).sum(0), x.shape[0]


rows = []
for f in sorted(glob.glob('experiments/2026-10_ssl-campaign/diagnostics/outputs/repro/*/dump/spike_*.pt')):
    d = torch.load(f, map_location='cpu', weights_only=False)
    rows.append((f.split('/')[-3] + ' ' + f.split('/')[-1], *stats(d['batch'])))

cfg.datamodule.batch_size = 12288
cfg.datamodule.num_workers = 0
dm = instantiate(cfg.datamodule, _convert_='all', train_dataset_path=cfg.pretrain_dataset_path)
dm.setup('fit')
it = iter(dm.train_dataloader())
for i in range(int(sys.argv[1])):
    rows.append((f'ordinary train batch {i}', *stats(next(it))))

print('features:', names)
for name, amax, n20, n in rows:
    print(f'{name:38s} N {n:6d} max|x| ' + ' '.join(f'{v:7.1f}' for v in amax.tolist()) + f'   tracks with any |x|>20: {int(n20.sum())}')
