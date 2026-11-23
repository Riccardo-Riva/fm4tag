"""Projector-output norms and their gradients: healthy vs collapsed checkpoint of one run."""
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fm4tag.utils import build_aggregator, build_encoders

run = Path(sys.argv[1])
ckpts = sys.argv[2:]
cfg = OmegaConf.load(next(run.glob('outputs/*/*/.hydra/config.yaml')))
cfg.datamodule.batch_size = 1024
cfg.datamodule.num_workers = 0
torch.manual_seed(0)
dm = instantiate(cfg.datamodule, _convert_='all', train_dataset_path=cfg.pretrain_dataset_path)
dm.setup('fit')
batch = next(iter(dm.val_dataloader()))

for name in ckpts:
    torch.manual_seed(0)  # same augmentation draws for both checkpoints
    enc = build_encoders(cfg)
    agg = build_aggregator(cfg, enc)
    views = [instantiate(v) for v in cfg.get('views', [])]
    m = instantiate(cfg.pretrain, encoders=enc, aggregator=agg, views=views, _convert_='all')
    ck = torch.load(next(run.glob(f'*/version_*/checkpoints/{name}')), map_location='cpu', weights_only=False)
    m.load_state_dict(ck['state_dict'])
    m.train()
    captured = []
    orig = m.contrastive_loss.forward
    def hook(z_list, groups=None, _orig=orig):
        for z in z_list:
            z.retain_grad()
        captured.append(z_list)
        return _orig(z_list, groups=groups)
    m.contrastive_loss.forward = hook
    loss, logs = m._compute_loss(batch)
    loss.backward()
    print(f'\n=== {name} (epoch {ck["epoch"]}): loss {loss.item():.3f}')
    for obj, z_list in zip(['jets', 'tracks'], captured):
        z = torch.cat([v.detach() for v in z_list])
        g = torch.cat([v.grad for v in z_list])
        n, gn = z.norm(dim=-1), g.norm(dim=-1)
        q = torch.quantile(n, torch.tensor([0.0, 0.001, 0.01, 0.5, 0.99]))
        small = n < 0.05 * n.median()
        print(f'  {obj}: N {len(n)}  |z| min {q[0]:.3g}  0.1% {q[1]:.3g}  1% {q[2]:.3g}  median {q[3]:.3g}  99% {q[4]:.3g}')
        print(f'    |dL/dz| median {gn.median():.3g}  max {gn.max():.3g}; |z| < 5% of median: {int(small.sum())} rows'
              f' carrying {(gn[small] ** 2).sum().sqrt() / (gn ** 2).sum().sqrt() * 100:.1f}% of the grad norm')
        top = gn.argsort(descending=True)[:3]
        print(f'    largest |dL/dz| rows: ' + ', '.join(f'|g| {gn[i]:.3g} at |z| {n[i]:.3g}' for i in top))
    tot = torch.nn.utils.get_total_norm([p.grad for p in m.parameters() if p.grad is not None])
    parts = {}
    for pn, p in m.named_parameters():
        if p.grad is not None:
            key = '.'.join(pn.split('.')[:3])
            parts[key] = parts.get(key, 0.0) + p.grad.float().pow(2).sum().item()
    print(f'  total grad norm {tot:.3g}; top parts: ' + ', '.join(f'{k} {v ** 0.5:.3g}' for k, v in sorted(parts.items(), key=lambda kv: -kv[1])[:5]))
