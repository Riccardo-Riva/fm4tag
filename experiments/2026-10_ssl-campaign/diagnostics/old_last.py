import sys, tempfile
from pathlib import Path
import lightning as L, torch
sys.path.insert(0, 'tests')
from test_last_checkpoint import _Worsening
d = Path(tempfile.mkdtemp())
cb = L.pytorch.callbacks.ModelCheckpoint(dirpath=d, monitor='val/loss', save_top_k=3, mode='min', save_last=True)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(16, 4)), batch_size=8)
L.Trainer(accelerator='cpu', max_epochs=5, logger=False, callbacks=[cb], enable_progress_bar=False, enable_model_summary=False).fit(_Worsening(), loader, loader)
print('old save_last: last.ckpt epoch', torch.load(d / 'last.ckpt', weights_only=False)['epoch'], 'of 4')
