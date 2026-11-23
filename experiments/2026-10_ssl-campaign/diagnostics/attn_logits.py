"""Max scaled attention logit and mean entropy of every RowAttention, per checkpoint, on one val batch."""
import glob
import math
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fm4tag.models.attention import RowAttention
from fm4tag.utils import build_aggregator, build_encoders

cfg = OmegaConf.load(glob.glob('experiments/2026-10_ssl-campaign/diagnostics/outputs/repro/p1_s42/outputs/*/*/.hydra/config.yaml')[0])
cfg.datamodule.batch_size = 1024
cfg.datamodule.num_workers = 0
dm = instantiate(cfg.datamodule, _convert_='all', train_dataset_path=cfg.pretrain_dataset_path)
dm.setup('fit')
batch = next(iter(dm.val_dataloader()))

enc = build_encoders(cfg)
agg = build_aggregator(cfg, enc)
m = instantiate(cfg.pretrain, encoders=enc, aggregator=agg, views=[instantiate(v) for v in cfg.views], _convert_='all')
m.eval()

stats = {}


def hook(name):
    def fn(mod, args, kwargs):
        x, groups = args[0], kwargs.get('groups')
        b, n, _ = x.shape
        q, k, _v = (t.reshape(b, n * mod.heads, -1) for t in mod.to_qkv(x).chunk(3, dim=-1))
        d = q.shape[-1]
        cu = groups.cu_seqlens.tolist()
        mx, ent = 0.0, []
        for s, e in zip(cu[:-1], cu[1:]):
            if e - s < 2:
                continue
            lg = torch.einsum('lhd,mhd->hlm', q[s:e], k[s:e]) / math.sqrt(d)  # (H, L, L)
            mx = max(mx, lg.abs().max().item())
            p = lg.softmax(-1)
            ent.append((-(p * p.clamp_min(1e-30).log()).sum(-1) / math.log(e - s)).mean().item())
        stats[name] = (mx, sum(ent) / len(ent))
    return fn


for name, mod in m.named_modules():
    if isinstance(mod, RowAttention) and name.startswith('encoders.tracks'):
        mod.register_forward_pre_hook(hook(name.replace('encoders.tracks.layers.', 'L').replace('.blocks.', 'b').replace('.attn.fn', '')), with_kwargs=True)

import csv
out = csv.writer(open(sys.argv[1], 'w', newline=''))
out.writerow(['label', 'layer', 'max_logit', 'entropy', 'qkv_norm'])
for label, path in [a.split('=', 1) for a in sys.argv[2:]]:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = {('encoders.' + k[len('backbone.'):] if k.startswith('backbone.') else k): v for k, v in ck['state_dict'].items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k.startswith('encoders.tracks')]
    unexpected = [k for k in unexpected if k.startswith('encoders.tracks')]
    stats.clear()
    with torch.no_grad():
        m(batch)
    qkv = {k: v.norm().item() for k, v in sd.items() if k.startswith('encoders.tracks') and k.endswith('to_qkv.weight') and '.2.attn' in k}
    for layer, (top, ent) in stats.items():
        w = sd.get('encoders.tracks.layers.0.blocks.' + layer.split('b')[1].split('.')[0] + '.2.attn.fn.to_qkv.weight')
        out.writerow([label, layer, f'{top:.6g}', f'{ent:.4f}', f'{w.norm().item():.2f}' if w is not None else ''])
    print(f'{label:16s} ' + '  '.join(f'{k}: max {v[0]:6.1f} H {v[1]:.2f}' for k, v in stats.items())
          + f'  | |W_qkv| row L0b0 {qkv.get("encoders.tracks.layers.0.blocks.0.2.attn.fn.to_qkv.weight", float("nan")):.1f}'
          + (f'  (missing {len(missing)}, unexpected {len(unexpected)})' if missing or unexpected else ''), flush=True)
