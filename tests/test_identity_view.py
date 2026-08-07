"""Identity views must reuse the clean pass, bit-for-bit.

With ``jet_contrastive > 0`` the finetune step used to run three full encoder
passes: the clean one for the CE head plus one per view — and the default second
view is ``Identity``, which reproduces the clean pass exactly.  Reusing it drops
a third of the model compute (measured 3226 -> 4752 jets/s at batch 1024, and
8069 -> 5436 MiB), so what needs guarding is that the reuse is *exact* and that
the augmented view is still genuinely computed.
"""

from __future__ import annotations

import pytest
import torch

from fm4tag.augmentations import Compose, Identity, Mixup


# ---------------------------------------------------------------------------
# Compose.is_identity
# ---------------------------------------------------------------------------


def test_compose_of_identities_is_identity():
    assert Compose([Identity()]).is_identity
    assert Compose([Identity(), Identity()]).is_identity


def test_empty_compose_is_identity():
    assert Compose([]).is_identity


def test_compose_with_a_real_augmentation_is_not_identity():
    assert not Compose([Mixup(lam=0.1)]).is_identity
    assert not Compose([Identity(), Mixup(lam=0.1)]).is_identity


def test_identity_is_not_inferred_from_parameters():
    # lam=0 may happen to be a no-op, but only literal Identity counts —
    # inferring it from parameters would silently skip a real augmentation
    # if its semantics ever changed.
    assert not Compose([Mixup(lam=0.0)]).is_identity


# ---------------------------------------------------------------------------
# FinetuneModule._encode_jet_views
# ---------------------------------------------------------------------------


@pytest.fixture
def finetune_setup():
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from hydra.utils import instantiate as hydra_instantiate

    from fm4tag.configs import __path__ as cfg_path
    from fm4tag.utils.model_builders import (
        build_aggregator,
        build_encoders,
        build_head,
    )

    with initialize_config_dir(config_dir=str(cfg_path[0]), version_base=None):
        cfg = hydra_compose(
            config_name='default',
            overrides=['phase=finetune', 'action=fit', 'encoder_ckpt=null'],
        )

    torch.manual_seed(0)
    encoders = build_encoders(cfg)
    aggregator = build_aggregator(cfg, encoders)
    head = build_head(cfg, aggregator)
    views = [hydra_instantiate(v) for v in cfg.get('views', [])]

    module = hydra_instantiate(
        cfg.finetune,
        encoders=encoders,
        aggregator=aggregator,
        head=head,
        views=views,
        n_classes=len(cfg.variables[cfg.global_object].unique_labels),
        _convert_='all',
    )

    B, C = 32, 40
    torch.manual_seed(1)
    valid = torch.rand(B, C) < 0.18
    valid[:, 0] = True  # no empty jets
    batch = {
        'label': torch.randint(0, 4, (B,)),
        'global': torch.randn(B, len(cfg.variables.jets.inputs)),
        'constituents': {
            'tracks': {
                'categorical': torch.stack(
                    [
                        torch.randint(0, len(c), (B, C))
                        for c in cfg.variables.tracks.inputs.cat_classes.values()
                    ],
                    dim=-1,
                ),
                'continuous': torch.randn(
                    B, C, len(cfg.variables.tracks.inputs.continuous)
                ),
                'valid': valid,
            }
        },
    }
    return module, batch


def test_default_config_has_exactly_one_identity_view(finetune_setup):
    module, _ = finetune_setup
    assert sum(v.is_identity for v in module.views) == 1


def test_identity_view_reuses_the_clean_pass_exactly(finetune_setup):
    module, batch = finetune_setup
    module.eval()

    with torch.no_grad():
        out = module(batch)
        torch.manual_seed(7)
        recomputed = module._encode_jet_views(batch)  # z_clean=None: recompute all
        torch.manual_seed(7)
        reused = module._encode_jet_views(batch, z_clean=out['aggregator'])

    assert len(recomputed) == len(reused)
    for i, view in enumerate(module.views):
        assert torch.equal(recomputed[i], reused[i]), f'view[{i}] changed'
        if view.is_identity:
            assert reused[i] is out['aggregator'], (
                f'view[{i}] is identity but was recomputed'
            )


def test_augmented_view_still_differs_from_the_clean_pass(finetune_setup):
    """Otherwise the contrastive loss would have nothing to separate."""
    module, batch = finetune_setup
    module.eval()

    with torch.no_grad():
        out = module(batch)
        z_views = module._encode_jet_views(batch, z_clean=out['aggregator'])

    augmented = [i for i, v in enumerate(module.views) if not v.is_identity]
    assert augmented, 'expected at least one augmented view'
    for i in augmented:
        assert not torch.allclose(z_views[i], out['aggregator'])


def test_gradients_still_reach_the_backbone(finetune_setup):
    module, batch = finetune_setup
    module.train()

    out = module(batch)
    z_views = module._encode_jet_views(batch, z_clean=out['aggregator'])
    sum(z.pow(2).mean() for z in z_views).backward()

    with_grad = [
        p for p in module.backbone.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert with_grad, 'reusing the clean tensor detached the backbone'
