"""Run the Lightning DDP device-count invariance script as a subprocess.

``scripts/ddp_loss_invariance.py`` spins up real Lightning DDP process groups
for ``devices = 1, 2, 3, 4``; running that in-process under pytest would clash
with pytest's own process/`ddp_spawn` management, so we invoke it as an
isolated subprocess and simply assert it exits 0 (the script does the
loss/gradient comparisons itself).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / 'scripts' / 'ddp_loss_invariance.py'


@pytest.mark.ddp
@pytest.mark.timeout(180)
def test_loss_invariance_script():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'ddp_loss_invariance.py failed (exit code {result.returncode})\n'
        f'--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}'
    )
