"""Where does the frozen arm go non-finite at batch 94208 (fine at 32768)?

Rebuilds the failed S3 run's model from its .hydra config (P0 seed-42 last.ckpt, frozen encoders), then for each
batch size runs a few real training steps (bf16 autocast, AdamW at the run's lr) and prints the first modules whose
forward output is non-finite, the non-finite gradients and the loss.

  python fr_nan_diag.py <run config.yaml> <B> [<B> ...]
"""

import os
import sys

import torch
from hydra.utils import instantiate
from lightning.fabric.utilities.apply_func import move_data_to_device
from omegaconf import OmegaConf

from fm4tag.runner.run import _load_aggregator
from fm4tag.utils import (
    build_aggregator,
    build_encoders,
    build_head,
    load_pretrained_encoders,
)

bad = []


def hook(name):
    def fn(mod, args, out):
        outs = out if isinstance(out, (tuple, list)) else [out]
        for o in outs:
            if (
                torch.is_tensor(o)
                and o.is_floating_point()
                and not torch.isfinite(o).all()
            ):
                bad.append(
                    f'{name} {tuple(o.shape)} nonfinite={(~torch.isfinite(o)).sum().item()}'
                )
                return

    return fn


def main():
    if os.environ.get('ANOMALY'):  # name the forward op of a failing backward
        torch.autograd.set_detect_anomaly(True)
    cfg = OmegaConf.load(sys.argv[1])
    torch.manual_seed(0)
    enc = build_encoders(cfg)
    agg = build_aggregator(cfg, enc)
    head = build_head(cfg, agg)
    parts = ['encoders'] if load_pretrained_encoders(enc, cfg.encoder_ckpt) else []
    assert not _load_aggregator(cfg.get('load_aggregator', 'auto'), cfg.encoder_ckpt)
    n_classes = len(cfg.variables[cfg.global_object].unique_labels)
    m = instantiate(
        cfg.finetune,
        encoders=enc,
        aggregator=agg,
        head=head,
        views=[instantiate(v) for v in cfg.views],
        n_classes=n_classes,
        pretrained_parts=parts,
        _convert_='all',
    ).cuda()
    init = {k: v.detach().clone() for k, v in m.state_dict().items()}

    for name, mod in m.named_modules():
        if name:
            mod.register_forward_hook(hook(name))

    for B in map(int, sys.argv[2:]):
        m.load_state_dict(init)
        m.train()
        opt = torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad], lr=cfg.optimizer.lr
        )
        c = cfg.datamodule.copy()
        c.batch_size, c.num_workers = B, 2
        dm = instantiate(c, _convert_='all', train_dataset_path=cfg.train_dataset_path)
        dm.setup('fit')
        it = iter(dm.train_dataloader())
        for step in range(4):
            batch = move_data_to_device(next(it), 'cuda')
            n_tracks = sum(
                int(v['valid'].sum()) for v in batch['constituents'].values()
            )
            bad.clear()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, logits, _ = m._compute_loss(batch)
            fwd = list(bad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gbad = [
                n
                for n, p in m.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()
            ]
            opt.step()
            pbad = [n for n, p in m.named_parameters() if not torch.isfinite(p).all()]
            print(
                f'B={B} step={step} tracks={n_tracks} loss={loss.item():.4f} '
                f'logits_finite={torch.isfinite(logits).all().item()} peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB',
                flush=True,
            )
            for line in fwd[:8]:
                print('   fwd nonfinite:', line)
            if gbad:
                print(f'   grad nonfinite ({len(gbad)}):', gbad[:6])
            if pbad:
                print(f'   param nonfinite ({len(pbad)}):', pbad[:6])
        del it, dm
        torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':  # the data loader spawns its workers
    main()
