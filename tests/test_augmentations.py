"""Tests for augmentations and AugmentationPipeline.

These tests are pure-tensor — no HDF5 file or model needed.
All augmentation classes are nn.Module; the callable interface is via __call__.
"""

import torch
import pytest
from fm4tag.augmentations import (
    AugmentationPipeline,
    CategoricalShift,
    ContinuousDilation,
    ContinuousFeatureDilation,
    CutMix,
    Mixup,
    MultiViewAugmentation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(N=32, F_cat=3, F_con=4, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    x_categ = torch.randint(0, 5, (N, F_cat), generator=g)
    x_cont  = torch.randn(N, F_con, generator=g)
    return x_categ, x_cont


def make_embedded(N=32, F=4, dim=8, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(N, F, dim, generator=g)


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

class TestCutMix:
    def test_is_nn_module(self):
        assert isinstance(CutMix(0.1), torch.nn.Module)

    def test_output_shape_matches_input(self):
        x_categ, x_cont = make_batch()
        aug = CutMix(lam=0.3)
        out_categ, out_cont = aug(x_categ, x_cont)
        assert out_categ.shape == x_categ.shape
        assert out_cont.shape  == x_cont.shape

    def test_output_dtype_preserved(self):
        x_categ, x_cont = make_batch()
        aug = CutMix(lam=0.3)
        out_categ, out_cont = aug(x_categ, x_cont)
        assert out_categ.dtype == x_categ.dtype
        assert out_cont.dtype  == x_cont.dtype

    def test_some_elements_are_changed(self):
        x_categ, x_cont = make_batch(N=256)
        aug = CutMix(lam=0.9)
        out_categ, out_cont = aug(x_categ, x_cont)
        assert not torch.equal(out_categ, x_categ)
        assert not torch.equal(out_cont,  x_cont)

    def test_zero_lam_is_identity(self):
        x_categ, x_cont = make_batch()
        aug = CutMix(lam=0.0)
        out_categ, out_cont = aug(x_categ, x_cont)
        assert torch.equal(out_categ, x_categ)
        assert torch.equal(out_cont,  x_cont)

    def test_none_categ_for_global_object(self):
        _, x_cont = make_batch()
        aug = CutMix(lam=0.3)
        out_categ, out_cont = aug(None, x_cont)
        assert out_categ is None
        assert out_cont.shape == x_cont.shape

    def test_output_values_come_from_input(self):
        x_categ, x_cont = make_batch(N=64)
        aug = CutMix(lam=0.5)
        out_categ, _ = aug(x_categ, x_cont)
        valid_values = set(x_categ.flatten().tolist())
        assert set(out_categ.flatten().tolist()).issubset(valid_values)

    def test_ignores_obj_name_kwarg(self):
        x_categ, x_cont = make_batch()
        aug = CutMix(lam=0.0)
        out_categ, out_cont = aug(x_categ, x_cont, obj_name='tracks')
        assert torch.equal(out_categ, x_categ)


# ---------------------------------------------------------------------------
# ContinuousDilation
# ---------------------------------------------------------------------------

class TestContinuousDilation:
    def test_is_nn_module(self):
        assert isinstance(ContinuousDilation(0.9), torch.nn.Module)

    def test_scales_continuous_by_alpha(self):
        x_categ, x_cont = make_batch()
        aug = ContinuousDilation(alpha=2.0)
        out_categ, out_cont = aug(x_categ, x_cont)
        assert torch.allclose(out_cont, x_cont * 2.0)

    def test_categorical_unchanged(self):
        x_categ, x_cont = make_batch()
        aug = ContinuousDilation(alpha=0.5)
        out_categ, _ = aug(x_categ, x_cont)
        assert torch.equal(out_categ, x_categ)

    def test_none_categ_passes_through(self):
        _, x_cont = make_batch()
        aug = ContinuousDilation(alpha=0.5)
        out_categ, out_cont = aug(None, x_cont)
        assert out_categ is None
        assert torch.allclose(out_cont, x_cont * 0.5)

    def test_alpha_one_is_identity(self):
        x_categ, x_cont = make_batch()
        aug = ContinuousDilation(alpha=1.0)
        _, out_cont = aug(x_categ, x_cont)
        assert torch.allclose(out_cont, x_cont)


# ---------------------------------------------------------------------------
# ContinuousFeatureDilation
# ---------------------------------------------------------------------------

class TestContinuousFeatureDilation:
    def test_is_nn_module(self):
        assert isinstance(ContinuousFeatureDilation(['a'], 0.9), torch.nn.Module)

    def test_scales_selected_features(self):
        _, x_cont = make_batch(F_con=4)
        aug = ContinuousFeatureDilation(features=['f1', 'f3'], alpha=2.0)
        aug.setup(obj_name='obj', continuous_features=['f0', 'f1', 'f2', 'f3'])
        _, out_cont = aug(None, x_cont.clone(), obj_name='obj')
        assert torch.allclose(out_cont[:, 0], x_cont[:, 0])
        assert torch.allclose(out_cont[:, 1], x_cont[:, 1] * 2.0)
        assert torch.allclose(out_cont[:, 2], x_cont[:, 2])
        assert torch.allclose(out_cont[:, 3], x_cont[:, 3] * 2.0)

    def test_missing_features_are_skipped(self):
        _, x_cont = make_batch(F_con=4)
        aug = ContinuousFeatureDilation(features=['not_here'], alpha=99.0)
        aug.setup(obj_name='obj', continuous_features=['f0', 'f1', 'f2', 'f3'])
        _, out_cont = aug(None, x_cont, obj_name='obj')
        assert torch.allclose(out_cont, x_cont)

    def test_unknown_obj_name_is_noop(self):
        _, x_cont = make_batch(F_con=4)
        aug = ContinuousFeatureDilation(features=['f0'], alpha=2.0)
        aug.setup(obj_name='tracks', continuous_features=['f0', 'f1'])
        _, out_cont = aug(None, x_cont, obj_name='jets')
        assert torch.allclose(out_cont, x_cont)

    def test_per_object_setup(self):
        _, x_cont = make_batch(F_con=4)
        aug = ContinuousFeatureDilation(features=['a'], alpha=3.0)
        aug.setup(obj_name='obj1', continuous_features=['x', 'a', 'y', 'z'])
        aug.setup(obj_name='obj2', continuous_features=['a', 'b', 'c', 'd'])
        _, out1 = aug(None, x_cont.clone(), obj_name='obj1')
        _, out2 = aug(None, x_cont.clone(), obj_name='obj2')
        assert torch.allclose(out1[:, 1], x_cont[:, 1] * 3.0)  # 'a' at idx 1
        assert torch.allclose(out1[:, 0], x_cont[:, 0])
        assert torch.allclose(out2[:, 0], x_cont[:, 0] * 3.0)  # 'a' at idx 0
        assert torch.allclose(out2[:, 1], x_cont[:, 1])


# ---------------------------------------------------------------------------
# CategoricalShift
# ---------------------------------------------------------------------------

class TestCategoricalShift:
    def test_is_nn_module(self):
        assert isinstance(CategoricalShift(0.5), torch.nn.Module)

    def test_none_categ_is_noop(self):
        _, x_cont = make_batch()
        aug = CategoricalShift(p=1.0)
        out_categ, out_cont = aug(None, x_cont, obj_name='obj')
        assert out_categ is None
        assert torch.equal(out_cont, x_cont)

    def test_unknown_obj_name_is_noop(self):
        x_categ, x_cont = make_batch()
        aug = CategoricalShift(p=1.0)
        out_categ, _ = aug(x_categ, x_cont, obj_name='unknown')
        assert torch.equal(out_categ, x_categ)

    def test_output_within_valid_range(self):
        N, F_cat = 128, 3
        n_classes = [5, 10, 3]
        x_categ = torch.zeros(N, F_cat, dtype=torch.long)
        x_categ[:, 0] = 0
        x_categ[:, 1] = 9
        x_categ[:, 2] = 1
        aug = CategoricalShift(p=1.0)
        aug.setup(obj_name='obj', n_classes=n_classes)
        out_categ, _ = aug(x_categ, torch.zeros(N, 4), obj_name='obj')
        for i, n in enumerate(n_classes):
            assert out_categ[:, i].min() >= 0
            assert out_categ[:, i].max() <= n - 1

    def test_output_shape_preserved(self):
        x_categ, x_cont = make_batch(N=32, F_cat=3)
        aug = CategoricalShift(p=0.5)
        aug.setup(obj_name='obj', n_classes=[5, 5, 5])
        out_categ, out_cont = aug(x_categ, x_cont, obj_name='obj')
        assert out_categ.shape == x_categ.shape
        assert out_cont.shape  == x_cont.shape

    def test_p_zero_is_identity(self):
        x_categ, x_cont = make_batch()
        aug = CategoricalShift(p=0.0)
        aug.setup(obj_name='obj', n_classes=[5, 5, 5])
        out_categ, _ = aug(x_categ, x_cont, obj_name='obj')
        assert torch.equal(out_categ, x_categ)

    def test_dtype_preserved(self):
        x_categ, x_cont = make_batch()
        aug = CategoricalShift(p=0.5)
        aug.setup(obj_name='obj', n_classes=[5, 5, 5])
        out_categ, _ = aug(x_categ, x_cont, obj_name='obj')
        assert out_categ.dtype == x_categ.dtype


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

class TestMixup:
    def test_is_nn_module(self):
        assert isinstance(Mixup(0.1), torch.nn.Module)

    def test_output_shape_single_tensor(self):
        x = make_embedded()
        (out,) = Mixup(lam=0.1)(x)
        assert out.shape == x.shape

    def test_output_shape_two_tensors(self):
        x1, x2 = make_embedded(seed=0), make_embedded(seed=1)
        out1, out2 = Mixup(lam=0.1)(x1, x2)
        assert out1.shape == x1.shape
        assert out2.shape == x2.shape

    def test_same_permutation_for_all_tensors(self):
        x = make_embedded()
        out1, out2 = Mixup(lam=0.5)(x, x.clone())
        assert torch.allclose(out1, out2)

    def test_lam_one_is_identity(self):
        x = make_embedded()
        (out,) = Mixup(lam=1.0)(x)
        assert torch.allclose(out, x)

    def test_output_dtype_preserved(self):
        x = make_embedded()
        (out,) = Mixup(lam=0.3)(x)
        assert out.dtype == x.dtype


# ---------------------------------------------------------------------------
# AugmentationPipeline
# ---------------------------------------------------------------------------

class TestAugmentationPipeline:
    def test_is_nn_module(self):
        assert isinstance(AugmentationPipeline(), torch.nn.Module)

    def test_raw_latent_are_module_lists(self):
        pipeline = AugmentationPipeline(raw=[CutMix(0.1)], latent=[Mixup(0.1)])
        assert isinstance(pipeline.raw, torch.nn.ModuleList)
        assert isinstance(pipeline.latent, torch.nn.ModuleList)

    def test_empty_pipeline_is_identity_raw(self):
        x_categ, x_cont = make_batch()
        out_categ, out_cont = AugmentationPipeline().apply_raw(x_categ, x_cont)
        assert torch.equal(out_categ, x_categ)
        assert torch.equal(out_cont,  x_cont)

    def test_empty_pipeline_is_identity_latent(self):
        x1, x2 = make_embedded(seed=0), make_embedded(seed=1)
        out1, out2 = AugmentationPipeline().apply_latent(x1, x2)
        assert torch.equal(out1, x1)
        assert torch.equal(out2, x2)

    def test_raw_stage_runs_in_order(self):
        x_categ, x_cont = make_batch()
        pipeline = AugmentationPipeline(raw=[CutMix(0.0), CutMix(0.0)])
        out_categ, out_cont = pipeline.apply_raw(x_categ, x_cont)
        assert torch.equal(out_categ, x_categ)
        assert torch.equal(out_cont,  x_cont)

    def test_cutmix_then_mixup_pipeline(self):
        x_categ, x_cont = make_batch(N=16)
        x_cat_emb = make_embedded(N=16, seed=0)
        x_con_emb = make_embedded(N=16, seed=1)
        pipeline = AugmentationPipeline(raw=[CutMix(lam=0.3)], latent=[Mixup(lam=0.3)])
        aug_categ, aug_cont = pipeline.apply_raw(x_categ, x_cont)
        assert aug_categ.shape == x_categ.shape
        assert aug_cont.shape  == x_cont.shape
        aug_cat_emb, aug_con_emb = pipeline.apply_latent(x_cat_emb, x_con_emb)
        assert aug_cat_emb.shape == x_cat_emb.shape
        assert aug_con_emb.shape == x_con_emb.shape

    def test_latent_single_tensor_global_case(self):
        X = make_embedded(N=16)
        pipeline = AugmentationPipeline(latent=[Mixup(lam=0.2)])
        (out,) = pipeline.apply_latent(X)
        assert out.shape == X.shape

    def test_none_categ_propagates_through_raw(self):
        _, x_cont = make_batch()
        pipeline = AugmentationPipeline(raw=[CutMix(lam=0.3)])
        out_categ, out_cont = pipeline.apply_raw(None, x_cont)
        assert out_categ is None
        assert out_cont.shape == x_cont.shape

    def test_obj_name_forwarded_to_augmentations(self):
        _, x_cont = make_batch(F_con=4)
        aug = ContinuousFeatureDilation(features=['f0'], alpha=2.0)
        aug.setup(obj_name='tracks', continuous_features=['f0', 'f1', 'f2', 'f3'])
        pipeline = AugmentationPipeline(raw=[aug])
        _, out = pipeline.apply_raw(None, x_cont.clone(), obj_name='tracks')
        assert torch.allclose(out[:, 0], x_cont[:, 0] * 2.0)
        assert torch.allclose(out[:, 1], x_cont[:, 1])


# ---------------------------------------------------------------------------
# MultiViewAugmentation
# ---------------------------------------------------------------------------

class TestMultiViewAugmentation:
    def test_is_nn_module(self):
        assert isinstance(MultiViewAugmentation([AugmentationPipeline()]), torch.nn.Module)

    def test_n_views_property(self):
        aug = MultiViewAugmentation([
            AugmentationPipeline(raw=[CutMix(0.1)]),
            AugmentationPipeline(raw=[ContinuousDilation(0.9)]),
        ])
        assert aug.n_views == 2

    def test_forward_returns_list_of_pairs(self):
        x_categ, x_cont = make_batch(N=16)
        aug = MultiViewAugmentation([
            AugmentationPipeline(raw=[CutMix(0.1)]),
            AugmentationPipeline(raw=[ContinuousDilation(0.9)]),
        ])
        views = aug(x_categ, x_cont)
        assert len(views) == 2
        for x_c, x_n in views:
            assert x_c.shape == x_categ.shape
            assert x_n.shape == x_cont.shape

    def test_empty_pipelines_list(self):
        x_categ, x_cont = make_batch()
        views = MultiViewAugmentation([])(x_categ, x_cont)
        assert views == []

    def test_views_differ_from_each_other(self):
        x_categ, x_cont = make_batch(N=64)
        aug = MultiViewAugmentation([
            AugmentationPipeline(raw=[CutMix(lam=0.9)]),
            AugmentationPipeline(raw=[ContinuousDilation(alpha=2.0)]),
        ])
        views = aug(x_categ, x_cont)
        _, cont0 = views[0]
        _, cont1 = views[1]
        assert not torch.allclose(cont0, cont1)

    def test_pipelines_registered_as_submodules(self):
        p1 = AugmentationPipeline(raw=[CutMix(0.1)])
        p2 = AugmentationPipeline(raw=[ContinuousDilation(0.5)])
        aug = MultiViewAugmentation([p1, p2])
        child_names = [name for name, _ in aug.named_modules()]
        assert any('pipelines' in n for n in child_names)
