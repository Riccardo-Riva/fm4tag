"""Diagnostic callback: per-parameter gradient norms around loss spikes.

Every step, before clipping: the total gradient norm, the loss and the norm of
each parameter's gradient. A step whose total norm exceeds ``factor`` x the
running median is a spike: its top parameters go to ``spikes.jsonl``, and for
the first ``n_dumps`` spikes the batch and the model state BEFORE the update
are saved, for offline replay. Training stops at global step ``stop_step``
(not trainer.max_steps, which would change the cosine schedule).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import lightning as L
import torch


class SpikeDump(L.Callback):
    def __init__(self, out_dir: str, stop_step: int, factor: float = 20.0, n_dumps: int = 3):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop_step = stop_step
        self.factor = factor
        self.n_dumps = n_dumps
        self.recent: deque[float] = deque(maxlen=200)
        self.batch = None
        self.dumped = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch = batch
        self.rng = {'cpu': torch.random.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        names, grads = zip(*[(n, p.grad) for n, p in pl_module.named_parameters() if p.grad is not None])
        norms = torch.stack([g.detach().float().norm() for g in grads]).cpu()
        total = float(norms.pow(2).sum().sqrt())
        step = trainer.global_step
        logs = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith('train/') and v.numel() == 1}
        with open(self.out / 'steps.jsonl', 'a') as f:
            f.write(json.dumps({'step': step, 'grad_norm': total, **logs}) + '\n')
        med = float(torch.tensor(list(self.recent)).median()) if len(self.recent) >= 20 else None
        if med is not None and total > self.factor * med:
            top = norms.argsort(descending=True)[:15]
            rec = {'step': step, 'grad_norm': total, 'median': med,
                   'top': [[names[i], float(norms[i])] for i in top]}
            with open(self.out / 'spikes.jsonl', 'a') as f:
                f.write(json.dumps(rec) + '\n')
            if self.dumped < self.n_dumps:
                torch.save({'step': step, 'batch': self.batch, 'rng': self.rng,
                            'state_dict': {k: v.detach().cpu() for k, v in pl_module.state_dict().items()}},
                           self.out / f'spike_{step}.pt')
                self.dumped += 1
        self.recent.append(total)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step >= self.stop_step:
            trainer.should_stop = True
