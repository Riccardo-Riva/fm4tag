#!/bin/bash
#SBATCH --job-name=gpu_health
#SBATCH --partition=gpu-A40
#SBATCH --nodelist=node82
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag/slurm/logs/gpu_health_%j.out
echo "host $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.total,temperature.gpu --format=csv
/storage3/DSIP/rriva/research/fm4tag/.venv/bin/python - <<'PY'
import torch
print("torch sees", torch.cuda.device_count(), "devices")
for i in range(torch.cuda.device_count()):
    try:
        x = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.bfloat16)
        y = (x @ x).float().sum().item()
        free, tot = torch.cuda.mem_get_info(i)
        print(f"cuda:{i} {torch.cuda.get_device_properties(i).pci_bus_id if hasattr(torch.cuda.get_device_properties(i),'pci_bus_id') else ''} ok matmul, free {free/2**30:.1f}/{tot/2**30:.1f} GiB")
    except Exception as e:
        print(f"cuda:{i} FAILED: {e!r}")
PY
