"""Shared pytest fixtures.

The ``hdf5_file`` fixture creates a minimal synthetic HDF5 file in a
temporary directory so dataset tests never touch real data files.

HDF5 schema mirrors what DatasetCatCon expects:

    file["jets"]   — structured array (N,)   with global features + label
    file["tracks"] — structured array (N, C) with cat/con features + valid
"""

import numpy as np
import pytest
import h5py
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

VARIABLES = {
    'jets': {
        'inputs': ['pt', 'eta'],
        'label': 'label',
        'unique_labels': [0, 1, 2],
    },
    'tracks': {
        'inputs': {
            'continuous': ['d0', 'z0'],
            'categorical': ['nPixHits', 'nSCTHits'],
            'cat_classes': {
                'nPixHits':  list(range(5)),   # 5 classes
                'nSCTHits':  list(range(10)),  # 10 classes
            },
        },
    },
}

N_JETS    = 20    # number of jet samples in the fake file
N_TRACKS  = 8    # max tracks per jet (padded)
N_CLASSES = 3    # number of label classes


@pytest.fixture(scope='session')
def cfg():
    """Minimal OmegaConf config matching VARIABLES above."""
    return OmegaConf.create({
        'global_object': 'jets',
        'constituent_objects': ['tracks'],
        'variables': VARIABLES,
    })


@pytest.fixture(scope='session')
def norm_dict():
    """Trivial norm dict: mean=0, std=1 → normalisation is a no-op."""
    return {
        'jets':   {'pt':  {'mean': 0.0, 'std': 1.0}, 'eta': {'mean': 0.0, 'std': 1.0}},
        'tracks': {'d0':  {'mean': 0.0, 'std': 1.0}, 'z0':  {'mean': 0.0, 'std': 1.0}},
    }


# ---------------------------------------------------------------------------
# HDF5 file fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def hdf5_file(tmp_path_factory):
    """Write a minimal synthetic HDF5 file and return its path.

    The file is created once per test session in a temporary directory
    managed by pytest and deleted automatically when the session ends.
    """
    path = tmp_path_factory.mktemp('data') / 'test.h5'

    rng = np.random.default_rng(42)

    # Global object: structured dtype with two float features + int label
    jets_dtype = np.dtype([
        ('pt',    np.float32),
        ('eta',   np.float32),
        ('label', np.int64),
    ])
    jets_data = np.empty(N_JETS, dtype=jets_dtype)
    jets_data['pt']    = rng.normal(size=N_JETS).astype(np.float32)
    jets_data['eta']   = rng.normal(size=N_JETS).astype(np.float32)
    jets_data['label'] = rng.integers(0, N_CLASSES, size=N_JETS)

    # Constituent object: structured dtype with cat + con + valid, shape (N, C)
    tracks_dtype = np.dtype([
        ('d0',       np.float32),
        ('z0',       np.float32),
        ('nPixHits', np.int64),
        ('nSCTHits', np.int64),
        ('valid',    bool),
    ])
    tracks_data = np.empty((N_JETS, N_TRACKS), dtype=tracks_dtype)
    tracks_data['d0']       = rng.normal(size=(N_JETS, N_TRACKS)).astype(np.float32)
    tracks_data['z0']       = rng.normal(size=(N_JETS, N_TRACKS)).astype(np.float32)
    tracks_data['nPixHits'] = rng.integers(0, 5,  size=(N_JETS, N_TRACKS))
    tracks_data['nSCTHits'] = rng.integers(0, 10, size=(N_JETS, N_TRACKS))
    # Each jet has between 1 and N_TRACKS valid tracks; rest are padding
    for i in range(N_JETS):
        n_valid = rng.integers(1, N_TRACKS + 1)
        mask = np.zeros(N_TRACKS, dtype=bool)
        mask[:n_valid] = True
        tracks_data['valid'][i] = mask

    with h5py.File(path, 'w') as f:
        f.create_dataset('jets',   data=jets_data)
        f.create_dataset('tracks', data=tracks_data)

    return str(path)
