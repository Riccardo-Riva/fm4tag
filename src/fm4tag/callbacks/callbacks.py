import os

import psutil
import torch
from lightning.pytorch.callbacks import Callback, TQDMProgressBar


class MemoryMonitorCallback(Callback):
    """Log CPU RSS and GPU VRAM usage each training step and validation epoch.

    Metrics logged (all in MiB):

    * ``mem/cpu_rss_MiB``  — resident set size of the main process (CPU RAM)
    * ``mem/gpu_alloc_MiB`` — GPU memory currently allocated by PyTorch tensors
    * ``mem/gpu_reserved_MiB`` — GPU memory reserved (cached) by the allocator
    * ``mem/gpu_peak_MiB``      — peak allocated since the start of the run

    Enable from config::

        callbacks:
          memory_monitor:
            enabled: true
            log_every_n_steps: 100   # optional, defaults to trainer.log_every_n_steps
    """

    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        self._log_every_n_steps = log_every_n_steps
        self._proc = psutil.Process(os.getpid())

    def _mem_stats(self) -> dict[str, float]:
        rss_mib = self._proc.memory_info().rss / 1024**2
        stats = {'mem/cpu_rss_MiB': rss_mib}
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            stats['mem/gpu_alloc_MiB'] = torch.cuda.memory_allocated(dev) / 1024**2
            stats['mem/gpu_reserved_MiB'] = torch.cuda.memory_reserved(dev) / 1024**2
            stats['mem/gpu_peak_MiB'] = torch.cuda.max_memory_allocated(dev) / 1024**2
        return stats

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self._log_every_n_steps == 0:
            pl_module.log_dict(self._mem_stats(), on_step=True, on_epoch=False)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        pl_module.log_dict(
            self._mem_stats(), on_step=False, on_epoch=True, sync_dist=True
        )


class GradNormMonitorCallback(Callback):
    """Log the global L2 norm of the gradients, before any clipping.

    The norm is computed every optimizer step (on device, no host sync) and
    logged on the steps Lightning writes its logs (``trainer.log_every_n_steps``),
    so each row covers the steps since the previous one:

    * ``train/grad_norm``      — the norm at the logged step
    * ``train/grad_norm_max``  — the largest norm since the previous log, so a
      short spike between two logs is not missed
    * ``train/grad_clip_frac`` — fraction of those steps whose norm exceeded
      ``trainer.gradient_clip_val`` (only when clipping by norm is on)
    """

    def __init__(self) -> None:
        super().__init__()
        self._max = None
        self._clipped = None
        self._steps = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        # Lightning runs this hook before it clips the gradients.
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        if not grads:
            return
        norm = torch.nn.utils.get_total_norm(grads).float()
        self._max = norm if self._max is None else torch.maximum(self._max, norm)
        clip = trainer.gradient_clip_val
        by_norm = trainer.gradient_clip_algorithm in (None, 'norm')
        if clip and by_norm:
            hit = (norm > clip).float()
            self._clipped = hit if self._clipped is None else self._clipped + hit
        self._steps += 1
        # The step Lightning writes: (batches stepped + 1) % log_every_n_steps.
        if (trainer.global_step + 1) % trainer.log_every_n_steps:
            return
        stats = {'train/grad_norm': norm, 'train/grad_norm_max': self._max}
        if self._clipped is not None:
            stats['train/grad_clip_frac'] = self._clipped / self._steps
        pl_module.log_dict(stats, on_step=True, on_epoch=False)
        self._max, self._clipped, self._steps = None, None, 0


class _PrecisionProgressBar(TQDMProgressBar):
    """TQDMProgressBar that formats floats with 4 decimal places.

    Lightning's default ``Tqdm.format_num`` uses ``.3g`` (3 significant
    figures), which for loss values in the range 1-9 only preserves 2
    decimal digits, then pads with a trailing zero to a fixed width.
    Pre-converting floats to strings here bypasses that pipeline.
    """

    def get_metrics(self, trainer, pl_module):  # type: ignore[override]
        metrics = super().get_metrics(trainer, pl_module)
        return {
            k: f'{v:.4f}' if isinstance(v, float) else v
            for k, v in metrics.items()
            if not k.endswith('_step')
        }
