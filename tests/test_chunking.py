"""Jet-chunked intersample attention, end to end.

With ``chunk_size = C`` the batch is cut into chunks of ``C`` consecutive jets
and the row attention of every encoder runs within a chunk only: among the
chunk's jets (global encoder) and among all valid tracks of the chunk's jets
(track encoder).  So a jet's prediction must not depend on any other chunk.
"""

from __future__ import annotations

import pytest
import torch

from fm4tag.models.attention import RowAttention
from fm4tag.modules.view_encoding import constituent_row_groups, jet_row_groups
from fm4tag.runner.run import _check_chunk_size

B, C_SLOTS = 12, 40


# ── boundaries ───────────────────────────────────────────────────────────────


def test_jet_row_groups():
    g = jet_row_groups(10, 4, 'cpu')
    assert g.cu_seqlens.tolist() == [0, 4, 8, 10]
    assert g.max_seqlen == 4
    assert jet_row_groups(10, None, 'cpu') is None


def test_constituent_row_groups_follow_the_jet_major_packing():
    counts = [2, 0, 3, 1, 2]
    valids = torch.zeros(5, 3, dtype=torch.bool)
    for j, n in enumerate(counts):
        valids[j, :n] = True
    # packed offsets [0, 2, 2, 5, 6, 8]; jet chunks [0, 2, 4, 5]
    g = constituent_row_groups(valids, 2)
    assert g.cu_seqlens.tolist() == [0, 2, 6, 8]
    assert g.max_seqlen == 4
    assert constituent_row_groups(valids, None) is None


def test_batch_size_must_split_into_whole_chunks():
    _check_chunk_size(None, 1000)
    _check_chunk_size(256, 1024)
    with pytest.raises(ValueError, match='multiple of chunk_size'):
        _check_chunk_size(300, 1024)


# ── modules ──────────────────────────────────────────────────────────────────


def _compose(overrides):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    from fm4tag.configs import __path__ as cfg_path

    with initialize_config_dir(config_dir=str(cfg_path[0]), version_base=None):
        return hydra_compose(config_name='default', overrides=overrides)


def _batch(cfg):
    torch.manual_seed(1)
    valid = torch.rand(B, C_SLOTS) < 0.18
    valid[:, 0] = True
    return {
        'label': torch.randint(0, 4, (B,)),
        'global': torch.randn(B, len(cfg.variables.jets.inputs)),
        'constituents': {
            'tracks': {
                'categorical': torch.stack(
                    [
                        torch.randint(0, len(c), (B, C_SLOTS))
                        for c in cfg.variables.tracks.inputs.cat_classes.values()
                    ],
                    dim=-1,
                ),
                'continuous': torch.randn(
                    B, C_SLOTS, len(cfg.variables.tracks.inputs.continuous)
                ),
                'valid': valid,
            }
        },
    }


def _build(phase, chunk_size, row_mode='per_token'):
    from hydra.utils import instantiate as hydra_instantiate

    from fm4tag.utils.model_builders import (
        build_aggregator,
        build_encoders,
        build_head,
    )

    enc = 'rowcol_v0' if row_mode == 'per_token' else 'rowcol_concat_v0'
    cfg = _compose([f'phase={phase}', f'encoders={enc}', f'chunk_size={chunk_size}'])
    torch.manual_seed(0)
    encoders = build_encoders(cfg)
    aggregator = build_aggregator(cfg, encoders)
    # Row attention ships zero-initialised (identity at init), which would make
    # the isolation checks trivially true; give it real weights.
    for m in encoders.modules():
        if isinstance(m, RowAttention):
            torch.nn.init.normal_(m.to_out.weight, std=0.1)
    views = [hydra_instantiate(v) for v in cfg.get('views', [])]
    kwargs = dict(
        encoders=encoders, aggregator=aggregator, views=views, _convert_='all'
    )
    if phase == 'finetune':
        node = cfg.finetune
        kwargs['head'] = build_head(cfg, aggregator)
        kwargs['n_classes'] = len(cfg.variables[cfg.global_object].unique_labels)
    else:
        node = cfg.pretrain
    return hydra_instantiate(node, **kwargs).eval(), _batch(cfg)


def _logits(module, batch):
    with torch.no_grad():
        return module.head(module(batch)['aggregator'])


def _perturb_last_chunk(batch, first_jet):
    b = {**batch, 'global': batch['global'].clone()}
    tr = {k: v.clone() for k, v in batch['constituents']['tracks'].items()}
    b['global'][first_jet:] += 3.0
    tr['continuous'][first_jet:] += 3.0
    b['constituents'] = {'tracks': tr}
    return b


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_chunks_isolate_jets_end_to_end(row_mode):
    module, batch = _build('finetune', 4, row_mode)
    out = _logits(module, batch)
    out_pert = _logits(module, _perturb_last_chunk(batch, 8))
    assert torch.equal(out[:8], out_pert[:8])
    assert not torch.allclose(out[8:], out_pert[8:], atol=1e-4)


def test_without_chunks_every_jet_sees_the_whole_batch():
    module, batch = _build('finetune', 'null')
    out = _logits(module, batch)
    out_pert = _logits(module, _perturb_last_chunk(batch, 8))
    assert not torch.allclose(out[:8], out_pert[:8], atol=1e-6)


@pytest.mark.parametrize('row_mode', ['per_token', 'concat'])
def test_one_chunk_equals_no_chunking(row_mode):
    chunked, batch = _build('finetune', B, row_mode)
    plain, _ = _build('finetune', 'null', row_mode)
    assert torch.allclose(_logits(chunked, batch), _logits(plain, batch), atol=1e-5)


def test_chunk_size_is_saved_in_hparams():
    module, _ = _build('finetune', 4)
    assert module.hparams.chunk_size == 4


@pytest.mark.parametrize('phase', ['pretrain', 'finetune'])
def test_chunked_training_step_backpropagates(phase):
    module, batch = _build(phase, 4)
    module.train()
    out = module._compute_loss(batch)
    loss = out[0]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in grads)
