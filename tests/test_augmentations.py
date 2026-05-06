"""Tests for augmentations and AugmentationPipeline.

These tests are pure-tensor — no HDF5 file or model needed.
"""

import torch
import pytest
from fm4tag.augmentations import AugmentationPipeline, CutMix, Mixup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(N=32, F_cat=3, F_con=4, seed=0):
    """Return a small batch of raw (pre-embedding) tensors."""
    g = torch.Generator()
    g.manual_seed(seed)
    x_categ = torch.randint(0, 5, (N, F_cat), generator=g)
    x_cont  = torch.randn(N, F_con, generator=g)
    return x_categ, x_cont


def make_embedded(N=32, F=4, dim=8, seed=0):
    """Return a small batch of embedded (post-embedding) tensors."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(N, F, dim, generator=g)


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

class TestCutMix:
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
        # With lam=0.9 almost everything is corrupted.
        x_categ, x_cont = make_batch(N=256)
        aug = CutMix(lam=0.9)
        out_categ, out_cont = aug(x_categ, x_cont)
        # At least some entries should differ from the original.
        assert not torch.equal(out_categ, x_categ)
        assert not torch.equal(out_cont,  x_cont)

    def test_zero_lam_is_identity(self):
        # lam=0 means no corruption — output must equal input.
        x_categ, x_cont = make_batch()
        aug = CutMix(lam=0.0)
        out_categ, out_cont = aug(x_categ, x_cont)
        assert torch.equal(out_categ, x_categ)
        assert torch.equal(out_cont,  x_cont)

    def test_none_categ_for_global_object(self):
        # Global object has no categorical features — x_categ is None.
        _, x_cont = make_batch()
        aug = CutMix(lam=0.3)
        out_categ, out_cont = aug(None, x_cont)
        assert out_categ is None
        assert out_cont.shape == x_cont.shape

    def test_output_values_come_from_input(self):
        # Every output element must be one of the values present in the input
        # (either from the same row or from a permuted row — never invented).
        x_categ, x_cont = make_batch(N=64)
        aug = CutMix(lam=0.5)
        out_categ, _ = aug(x_categ, x_cont)
        # Each output value must be a value that appeared in x_categ.
        valid_values = set(x_categ.flatten().tolist())
        assert set(out_categ.flatten().tolist()).issubset(valid_values)


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

class TestMixup:
    def test_output_shape_single_tensor(self):
        x = make_embedded()
        aug = Mixup(lam=0.1)
        (out,) = aug(x)
        assert out.shape == x.shape

    def test_output_shape_two_tensors(self):
        x1 = make_embedded(seed=0)
        x2 = make_embedded(seed=1)
        aug = Mixup(lam=0.1)
        out1, out2 = aug(x1, x2)
        assert out1.shape == x1.shape
        assert out2.shape == x2.shape

    def test_same_permutation_for_all_tensors(self):
        # Both outputs are mixed with the same index, so if x1 == x2
        # then out1 must equal out2.
        x = make_embedded()
        aug = Mixup(lam=0.5)
        out1, out2 = aug(x, x.clone())
        assert torch.allclose(out1, out2)

    def test_output_is_convex_combination(self):
        # With lam=1.0 the output must equal the original (x[i] mixed with
        # x[perm[i]], weight 1.0 on self → out = 1*x + 0*x[perm] = x).
        x = make_embedded()
        aug = Mixup(lam=1.0)
        (out,) = aug(x)
        assert torch.allclose(out, x)

    def test_output_dtype_preserved(self):
        x = make_embedded()
        aug = Mixup(lam=0.3)
        (out,) = aug(x)
        assert out.dtype == x.dtype


# ---------------------------------------------------------------------------
# AugmentationPipeline
# ---------------------------------------------------------------------------

class TestAugmentationPipeline:
    def test_empty_pipeline_is_identity_raw(self):
        x_categ, x_cont = make_batch()
        pipeline = AugmentationPipeline()
        out_categ, out_cont = pipeline.apply_raw(x_categ, x_cont)
        assert torch.equal(out_categ, x_categ)
        assert torch.equal(out_cont,  x_cont)

    def test_empty_pipeline_is_identity_latent(self):
        x1 = make_embedded(seed=0)
        x2 = make_embedded(seed=1)
        pipeline = AugmentationPipeline()
        out1, out2 = pipeline.apply_latent(x1, x2)
        assert torch.equal(out1, x1)
        assert torch.equal(out2, x2)

    def test_raw_stage_runs_in_order(self):
        # Two CutMix passes with lam=0.0 are both identity — output == input.
        x_categ, x_cont = make_batch()
        pipeline = AugmentationPipeline(raw=[CutMix(0.0), CutMix(0.0)])
        out_categ, out_cont = pipeline.apply_raw(x_categ, x_cont)
        assert torch.equal(out_categ, x_categ)
        assert torch.equal(out_cont,  x_cont)

    def test_cutmix_then_mixup_pipeline(self):
        # Full two-stage pipeline: raw CutMix + latent Mixup.
        x_categ, x_cont = make_batch(N=16)
        x_cat_emb = make_embedded(N=16, seed=0)
        x_con_emb = make_embedded(N=16, seed=1)

        pipeline = AugmentationPipeline(
            raw=[CutMix(lam=0.3)],
            latent=[Mixup(lam=0.3)],
        )
        # Raw stage changes the inputs
        aug_categ, aug_cont = pipeline.apply_raw(x_categ, x_cont)
        assert aug_categ.shape == x_categ.shape
        assert aug_cont.shape  == x_cont.shape

        # Latent stage returns same-shaped tensors
        aug_cat_emb, aug_con_emb = pipeline.apply_latent(x_cat_emb, x_con_emb)
        assert aug_cat_emb.shape == x_cat_emb.shape
        assert aug_con_emb.shape == x_con_emb.shape

    def test_latent_single_tensor_global_case(self):
        # Global object passes a single tensor through the latent pipeline.
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
