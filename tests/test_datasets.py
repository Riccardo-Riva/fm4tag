"""Tests for DatasetCatCon and cat_con_collate_fn.

These tests use the ``hdf5_file`` fixture from conftest.py, which creates
a minimal synthetic HDF5 file once per session.  No real data is required.
"""

import torch
import pytest
from torch.utils.data import DataLoader

from fm4tag.datasets import DatasetCatCon, cat_con_collate_fn
from tests.conftest import N_JETS, N_TRACKS, N_CLASSES, VARIABLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(hdf5_file, cfg, norm_dict=None):
    return DatasetCatCon(
        file_path=hdf5_file,
        variables=cfg.variables,
        global_object=cfg.global_object,
        constituent_objects=list(cfg.constituent_objects),
        norm_dict=norm_dict,
    )


# ---------------------------------------------------------------------------
# DatasetCatCon — length and item shapes
# ---------------------------------------------------------------------------

class TestDatasetCatCon:
    def test_len(self, hdf5_file, cfg):
        ds = make_dataset(hdf5_file, cfg)
        assert len(ds) == N_JETS

    def test_getitem_returns_dict_keys(self, hdf5_file, cfg):
        ds = make_dataset(hdf5_file, cfg)
        sample = ds[0]
        assert set(sample.keys()) == {'label', 'global', 'constituents'}

    def test_label_shape_and_dtype(self, hdf5_file, cfg):
        ds = make_dataset(hdf5_file, cfg)
        sample = ds[0]
        assert sample['label'].shape == ()
        assert sample['label'].dtype == torch.long

    def test_global_shape_and_dtype(self, hdf5_file, cfg):
        ds = make_dataset(hdf5_file, cfg)
        sample = ds[0]
        n_global_feats = len(VARIABLES['jets']['inputs'])  # 2
        assert sample['global'].shape == (n_global_feats,)
        assert sample['global'].dtype == torch.float32

    def test_constituent_dict_keys(self, hdf5_file, cfg):
        ds = make_dataset(hdf5_file, cfg)
        sample = ds[0]
        assert 'tracks' in sample['constituents']
        assert set(sample['constituents']['tracks'].keys()) == {
            'categorical', 'continuous', 'valid'
        }

    def test_constituent_shapes(self, hdf5_file, cfg):
        ds = make_dataset(hdf5_file, cfg)
        sample = ds[0]
        tracks = sample['constituents']['tracks']
        F_cat = len(VARIABLES['tracks']['inputs']['categorical'])  # 2
        F_con = len(VARIABLES['tracks']['inputs']['continuous'])   # 2
        assert tracks['categorical'].shape == (N_TRACKS, F_cat)
        assert tracks['continuous'].shape  == (N_TRACKS, F_con)
        assert tracks['valid'].shape       == (N_TRACKS,)

    def test_constituent_dtypes(self, hdf5_file, cfg):
        ds = make_dataset(hdf5_file, cfg)
        tracks = ds[0]['constituents']['tracks']
        assert tracks['categorical'].dtype == torch.int64
        assert tracks['continuous'].dtype  == torch.float32
        assert tracks['valid'].dtype       == torch.bool

    def test_label_in_range(self, hdf5_file, cfg):
        ds = make_dataset(hdf5_file, cfg)
        for i in range(len(ds)):
            label = ds[i]['label'].item()
            assert 0 <= label < N_CLASSES

    def test_valid_mask_has_at_least_one_true(self, hdf5_file, cfg):
        # The fixture guarantees each jet has at least 1 valid track.
        ds = make_dataset(hdf5_file, cfg)
        for i in range(len(ds)):
            assert ds[i]['constituents']['tracks']['valid'].any()

    def test_file_not_found_raises(self, cfg):
        with pytest.raises(FileNotFoundError):
            DatasetCatCon(
                file_path='/nonexistent/path.h5',
                variables=cfg.variables,
                global_object=cfg.global_object,
            )


# ---------------------------------------------------------------------------
# DatasetCatCon — normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_identity_norm_dict_leaves_values_unchanged(self, hdf5_file, cfg, norm_dict):
        # mean=0, std=1 → output == raw values
        ds_no_norm   = make_dataset(hdf5_file, cfg, norm_dict=None)
        ds_with_norm = make_dataset(hdf5_file, cfg, norm_dict=norm_dict)
        raw  = ds_no_norm[0]['global']
        normed = ds_with_norm[0]['global']
        assert torch.allclose(raw, normed)

    def test_norm_dict_shifts_mean(self, hdf5_file, cfg):
        # Subtract mean=1000 from both global features → values should be ~-1000
        shifted_norm = {
            'jets': {
                'pt':  {'mean': 1000.0, 'std': 1.0},
                'eta': {'mean': 1000.0, 'std': 1.0},
            },
        }
        ds_raw    = make_dataset(hdf5_file, cfg, norm_dict=None)
        ds_normed = make_dataset(hdf5_file, cfg, norm_dict=shifted_norm)
        raw    = ds_raw[0]['global']
        normed = ds_normed[0]['global']
        assert torch.allclose(normed, raw - 1000.0, atol=1e-4)

    def test_norm_dict_scales_std(self, hdf5_file, cfg):
        # Divide by std=2 → values should be halved
        scaled_norm = {
            'jets': {
                'pt':  {'mean': 0.0, 'std': 2.0},
                'eta': {'mean': 0.0, 'std': 2.0},
            },
        }
        ds_raw    = make_dataset(hdf5_file, cfg, norm_dict=None)
        ds_normed = make_dataset(hdf5_file, cfg, norm_dict=scaled_norm)
        raw    = ds_raw[0]['global']
        normed = ds_normed[0]['global']
        assert torch.allclose(normed, raw / 2.0, atol=1e-5)


# ---------------------------------------------------------------------------
# cat_con_collate_fn
# ---------------------------------------------------------------------------

class TestCollate:
    def _samples(self, hdf5_file, cfg, n=4):
        ds = make_dataset(hdf5_file, cfg)
        return [ds[i] for i in range(n)]

    def test_batch_label_shape(self, hdf5_file, cfg):
        B = 4
        batch = cat_con_collate_fn(self._samples(hdf5_file, cfg, n=B))
        assert batch['label'].shape == (B,)
        assert batch['label'].dtype == torch.long

    def test_batch_global_shape(self, hdf5_file, cfg):
        B = 4
        F_g = len(VARIABLES['jets']['inputs'])
        batch = cat_con_collate_fn(self._samples(hdf5_file, cfg, n=B))
        assert batch['global'].shape == (B, F_g)
        assert batch['global'].dtype == torch.float32

    def test_batch_constituent_shapes(self, hdf5_file, cfg):
        B = 4
        F_cat = len(VARIABLES['tracks']['inputs']['categorical'])
        F_con = len(VARIABLES['tracks']['inputs']['continuous'])
        batch = cat_con_collate_fn(self._samples(hdf5_file, cfg, n=B))
        tracks = batch['constituents']['tracks']
        assert tracks['categorical'].shape == (B, N_TRACKS, F_cat)
        assert tracks['continuous'].shape  == (B, N_TRACKS, F_con)
        assert tracks['valid'].shape       == (B, N_TRACKS)

    def test_batch_constituent_dtypes(self, hdf5_file, cfg):
        batch = cat_con_collate_fn(self._samples(hdf5_file, cfg, n=2))
        tracks = batch['constituents']['tracks']
        assert tracks['categorical'].dtype == torch.long
        assert tracks['continuous'].dtype  == torch.float32
        assert tracks['valid'].dtype       == torch.bool

    def test_dataloader_integration(self, hdf5_file, cfg):
        # Run one full batch through a DataLoader to catch any collation bugs.
        ds = make_dataset(hdf5_file, cfg)
        loader = DataLoader(ds, batch_size=4, collate_fn=cat_con_collate_fn)
        batch = next(iter(loader))
        assert 'label' in batch
        assert 'global' in batch
        assert 'constituents' in batch
