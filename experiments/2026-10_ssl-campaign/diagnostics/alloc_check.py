import torch  # imported first, as in fm4tag
import os

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
x = torch.empty(1 << 20, device='cuda')
segs = torch.cuda.memory._snapshot()['segments']
print(
    'after import, set before first CUDA call -> is_expandable:',
    {s.get('is_expandable') for s in segs},
)
