from numpy.lib.recfunctions import structured_to_unstructured as s2u

import h5py
import os
import torch
from torch.utils.data import Dataset

from ..utils import resolve_object_inputs


def cat_con_collate_fn(batch: list[dict]) -> dict:
    """Collate a list of samples from DatasetCatCon into a batched dict.

    The default PyTorch collate_fn handles nested dicts but does not enforce
    dtypes. This function stacks tensors and casts each field to its expected
    dtype so model code never needs to cast internally.

    Expected input shape per sample (from DatasetCatCon.__getitem__):
        label       : ()
        global      : { categorical: (F_gcat,), continuous: (F_gcon,) }
        constituents: { obj_name: { categorical: (N, F_cat),
                                    continuous:  (N, F_con),
                                    valid:       (N,) } }

    Output shapes (batch size B):
        label       : (B,)          long
        global      : { categorical: (B, F_gcat)  long
                        continuous:  (B, F_gcon)  float32 }
        constituents: { obj_name: { categorical: (B, N, F_cat)  long
                                    continuous:  (B, N, F_con)  float32
                                    valid:       (B, N)         bool } }
    """
    labels = torch.stack([s['label'] for s in batch])  # (B,)
    globals = {
        'categorical': torch.stack(
            [s['global']['categorical'] for s in batch]
        ).long(),  # (B, F_gcat)
        'continuous': torch.stack(
            [s['global']['continuous'] for s in batch]
        ).float(),  # (B, F_gcon)
    }

    object_names = batch[0]['constituents'].keys()
    constituents = {}
    for name in object_names:
        constituents[name] = {
            'categorical': torch.stack(
                [s['constituents'][name]['categorical'] for s in batch]
            ).long(),  # (B, N, F_cat)
            'continuous': torch.stack(
                [s['constituents'][name]['continuous'] for s in batch]
            ).float(),  # (B, N, F_con)
            'valid': torch.stack(
                [s['constituents'][name]['valid'] for s in batch]
            ).bool(),  # (B, N)
        }

    return {'label': labels, 'global': globals, 'constituents': constituents}


class DatasetCatCon(Dataset):
    def __init__(
        self,
        file_path,
        variables,
        global_object,
        constituent_objects=None,
        norm_dict=None,
        class_dict=None,
    ):
        super().__init__()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} not found.')

        self.file_path = file_path
        self.variables = variables
        self.global_object = global_object
        self.constituent_objects = constituent_objects or []
        self.label_name = variables[self.global_object]['label']
        self.class_dict = class_dict

        # Resolve the global object's feature split once (supports both the
        # legacy flat list — all continuous — and the tracks-style dict schema).
        self._global_continuous, self._global_categorical, _ = resolve_object_inputs(
            variables[self.global_object].inputs
        )

        # Pre-build per-object mean/std tensors once so __getitem__ does no
        # dict lookups or tensor allocations for normalization.
        self._build_norm_tensors(norm_dict)

        with h5py.File(self.file_path, 'r') as file:
            print(f'\nDatasetCatCon: {self.file_path}')
            self.len = file[self.global_object].shape[0]
            print(f'  samples : {self.len}')

        self.file = None  # lazy file opening: one handle opened per worker

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _build_norm_tensors(self, norm_dict: dict | None) -> None:
        """Convert norm_dict to pre-allocated tensors for fast per-sample use.

        norm_dict expected format::

            { "tracks": { "d0": {"mean": 0.0, "std": 1.0}, ... }, ... }

        The tensors are stored as:
            self._norm[obj_name]["mean"]  shape (F_con,)
            self._norm[obj_name]["std"]   shape (F_con,)
        """
        self._norm: dict[str, dict[str, torch.Tensor]] = {}
        if norm_dict is None:
            return

        # Normalisation applies to continuous features only.  resolve_object_inputs
        # handles both the global object (flat list or dict) and constituents.
        all_objects = [self.global_object] + list(self.constituent_objects)
        for obj_name in all_objects:
            if obj_name not in norm_dict:
                continue
            obj_norm = norm_dict[obj_name]
            features, _, _ = resolve_object_inputs(self.variables[obj_name].inputs)
            if not features:
                continue
            self._norm[obj_name] = {
                'mean': torch.tensor(
                    [obj_norm[f]['mean'] for f in features], dtype=torch.float32
                ),
                # Pre-clamped so __getitem__ can divide directly without per-sample ops.
                'std': torch.tensor(
                    [obj_norm[f]['std'] for f in features], dtype=torch.float32
                ).clamp(min=1e-8),
            }

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.len

    def _open_file(self) -> None:
        """Open the HDF5 file once per worker process and cache dataset handles."""
        self.file = h5py.File(self.file_path, 'r', swmr=True, libver='latest')
        self.g_dset = self.file[self.global_object]
        self.c_dsets = {name: self.file[name] for name in self.constituent_objects}

    def __getitem__(self, idx: int) -> dict:
        if self.file is None:
            self._open_file()

        # Single read per object — h5py dataset[idx] is a disk access each time.
        g_record = self.g_dset[idx]
        label = torch.tensor(g_record[self.label_name], dtype=torch.long)

        # Global continuous features (shape: (F_gcon,)); normalised if available.
        if self._global_continuous:
            X_g_con = torch.from_numpy(
                s2u(g_record[self._global_continuous], dtype=None)
            ).float()
            if self.global_object in self._norm:
                norm = self._norm[self.global_object]
                X_g_con = (X_g_con - norm['mean']) / norm['std']
        else:
            X_g_con = torch.zeros((0,), dtype=torch.float32)

        # Global categorical features (shape: (F_gcat,)); empty for legacy configs.
        if self._global_categorical:
            X_g_cat = torch.from_numpy(
                s2u(g_record[self._global_categorical], dtype=None)
            ).long()
        else:
            X_g_cat = torch.zeros((0,), dtype=torch.long)

        # Constituent features, one dict entry per object
        constituents = {}
        for obj_name in self.constituent_objects:
            c_record = self.c_dsets[obj_name][idx]
            X_cat = torch.from_numpy(
                s2u(c_record[self.variables[obj_name].inputs.categorical], dtype=None)
            )
            X_con = torch.from_numpy(
                s2u(c_record[self.variables[obj_name].inputs.continuous], dtype=None)
            ).float()

            if obj_name in self._norm:
                norm = self._norm[obj_name]
                X_con = (X_con - norm['mean']) / norm['std']

            valid = torch.from_numpy(c_record['valid'])

            constituents[obj_name] = {
                'categorical': X_cat,  # (N, F_cat)
                'continuous': X_con,  # (N, F_con)
                'valid': valid,  # (N,)
            }

        return {
            'label': label,
            'global': {
                'categorical': X_g_cat,  # (F_gcat,)
                'continuous': X_g_con,  # (F_gcon,)
            },
            'constituents': constituents,
        }
