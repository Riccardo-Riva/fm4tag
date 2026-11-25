"""Tests for fm4tag.models.MultiStreamClassifierHead: the MLP's layer widths."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from fm4tag.models import MultiStreamClassifierHead

_JET_DIM = 12
_Y_DIM = 4


def _widths(head: MultiStreamClassifierHead) -> list[int]:
    linears = [m for m in head.cls_mlp.mlp if isinstance(m, nn.Linear)]
    return [linears[0].in_features] + [m.out_features for m in linears]


@pytest.mark.parametrize(
    ('mlp_hidden', 'widths'),
    [
        (None, [_JET_DIM, _JET_DIM // 2, _Y_DIM]),
        (8, [_JET_DIM, 8, _Y_DIM]),
        ([8, 6, 5], [_JET_DIM, 8, 6, 5, _Y_DIM]),
        # what hydra passes for a YAML list
        (OmegaConf.create([8, 6, 5]), [_JET_DIM, 8, 6, 5, _Y_DIM]),
    ],
)
def test_mlp_widths(mlp_hidden, widths):
    head = MultiStreamClassifierHead(
        jet_dim=_JET_DIM, y_dim=_Y_DIM, mlp_hidden=mlp_hidden
    )
    assert _widths(head) == widths
    assert head(torch.randn(3, _JET_DIM)).shape == (3, _Y_DIM)


def test_dropout_after_every_hidden_layer_only():
    head = MultiStreamClassifierHead(
        jet_dim=_JET_DIM, y_dim=_Y_DIM, mlp_hidden=[8, 6, 5], mlp_dropout=0.1
    )
    kinds = [type(m).__name__ for m in head.cls_mlp.mlp]
    assert kinds == ['Linear', 'ReLU', 'Dropout'] * 3 + ['Linear']


def test_deep_head_overfits_a_tiny_batch():
    torch.manual_seed(0)
    head = MultiStreamClassifierHead(
        jet_dim=_JET_DIM, y_dim=_Y_DIM, mlp_hidden=[16, 8, 8]
    )
    x = torch.randn(16, _JET_DIM)
    y = torch.randint(0, _Y_DIM, (16,))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(head(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
