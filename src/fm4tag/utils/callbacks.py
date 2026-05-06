"""Custom Lightning callbacks used across pretrain and finetune CLIs."""

from __future__ import annotations

import os

import torch
from lightning.pytorch.callbacks import Callback, TQDMProgressBar


class MemoryMonitorCallback(Callback):
    """Log CPU RSS and GPU VRAM usage each training step and validation epoch.

    Metrics logged (all in MiB):
    * ``mem/cpu_rss_MiB``      — resident set size of the main process
    * ``mem/gpu_alloc_MiB``    — GPU memory currently allocated
    * ``mem/gpu_reserved_MiB`` — GPU memory reserved (cached) by the allocator

    Enable from config::

        callbacks:
          memory_monitor:
            enabled: true
            log_every_n_steps: 100
    """

    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        self._log_every_n_steps = log_every_n_steps
        import psutil
        self._proc = psutil.Process(os.getpid())

    def _mem_stats(self) -> dict[str, float]:
        rss_mib = self._proc.memory_info().rss / 1024**2
        stats = {'mem/cpu_rss_MiB': rss_mib}
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            stats['mem/gpu_alloc_MiB'] = torch.cuda.memory_allocated(dev) / 1024**2
            stats['mem/gpu_reserved_MiB'] = torch.cuda.memory_reserved(dev) / 1024**2
        return stats

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self._log_every_n_steps == 0:
            pl_module.log_dict(self._mem_stats(), on_step=True, on_epoch=False)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        pl_module.log_dict(self._mem_stats(), on_step=False, on_epoch=True, sync_dist=True)


class PrecisionProgressBar(TQDMProgressBar):
    """TQDMProgressBar that formats floats with 4 decimal places.

    Lightning's default ``Tqdm.format_num`` uses ``.3g`` (3 significant
    figures), which for loss values in the range 1–9 only preserves 2
    decimal digits.  Pre-converting floats to strings here bypasses that.
    """

    def get_metrics(self, trainer, pl_module):  # type: ignore[override]
        metrics = super().get_metrics(trainer, pl_module)
        return {
            k: f'{v:.4f}' if isinstance(v, float) else v
            for k, v in metrics.items()
            if not k.endswith('_step')
        }
