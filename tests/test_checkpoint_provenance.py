"""``load_aggregator: auto`` loads the aggregator only if pretraining trained it."""

from __future__ import annotations

import pytest
import torch

from fm4tag.runner.run import _load_aggregator
from fm4tag.utils import checkpoint_trained_aggregator


def _ckpt(tmp_path, loss_weights):
    path = tmp_path / 'pretrain.ckpt'
    hparams = {} if loss_weights is None else {'loss_weights': loss_weights}
    torch.save({'state_dict': {}, 'hyper_parameters': hparams}, path)
    return str(path)


@pytest.mark.parametrize(
    'weights,trained',
    [
        ({'contrastive': 1.0, 'jet_contrastive': 0.0}, False),
        ({'contrastive': 1.0, 'jet_contrastive': 0.5}, True),
        (None, None),
    ],
)
def test_checkpoint_trained_aggregator(tmp_path, weights, trained):
    assert checkpoint_trained_aggregator(_ckpt(tmp_path, weights)) is trained


def test_auto_skips_an_untrained_aggregator(tmp_path):
    path = _ckpt(tmp_path, {'contrastive': 1.0, 'jet_contrastive': 0.0})
    assert _load_aggregator('auto', path) is False
    with pytest.warns(UserWarning, match='random init'):
        assert _load_aggregator(True, path) is True
    assert _load_aggregator(False, path) is False


def test_auto_loads_a_trained_aggregator(tmp_path):
    path = _ckpt(tmp_path, {'contrastive': 1.0, 'jet_contrastive': 1.0})
    assert _load_aggregator('auto', path) is True


def test_auto_loads_when_unknown_but_warns(tmp_path):
    with pytest.warns(UserWarning, match='does not record'):
        assert _load_aggregator('auto', _ckpt(tmp_path, None)) is True


def test_bad_mode_raises(tmp_path):
    with pytest.raises(ValueError, match='load_aggregator'):
        _load_aggregator('sometimes', _ckpt(tmp_path, None))
