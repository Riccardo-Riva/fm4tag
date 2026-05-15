from numpy.lib.recfunctions import structured_to_unstructured as s2u

import h5py
import os
import torch
from torch.utils.data import Dataset


def cat_con_collate_fn(batch: list[dict]) -> dict:
    """Collate a list of samples from DatasetCatCon into a batched dict.

    The default PyTorch collate_fn handles nested dicts but does not enforce
    dtypes. This function stacks tensors and casts each field to its expected
    dtype so model code never needs to cast internally.

    Expected input shape per sample (from DatasetCatCon.__getitem__):
        label       : ()
        global      : (F_g,)
        constituents: { obj_name: { categorical: (N, F_cat),
                                    continuous:  (N, F_con),
                                    valid:       (N,) } }

    Output shapes (batch size B):
        label       : (B,)          long
        global      : (B, F_g)      float32
        constituents: { obj_name: { categorical: (B, N, F_cat)  long
                                    continuous:  (B, N, F_con)  float32
                                    valid:       (B, N)         bool } }
    """
    labels = torch.stack([s['label'] for s in batch])  # (B,)
    globals = torch.stack([s['global'] for s in batch]).float()  # (B, F_g)

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

        # Global object: inputs is a flat list (all continuous, no categorical).
        # Constituent objects: inputs.continuous is the continuous feature list.
        all_objects = [self.global_object] + list(self.constituent_objects)
        for obj_name in all_objects:
            if obj_name not in norm_dict:
                continue
            obj_norm = norm_dict[obj_name]
            features = (
                self.variables[obj_name].inputs
                if obj_name == self.global_object
                else self.variables[obj_name].inputs.continuous
            )
            self._norm[obj_name] = {
                'mean': torch.tensor(
                    [obj_norm[f]['mean'] for f in features], dtype=torch.float32
                ),
                'std': torch.tensor(
                    [obj_norm[f]['std'] for f in features], dtype=torch.float32
                ),
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

        # Label
        label = torch.tensor(self.g_dset[idx][self.label_name], dtype=torch.long)

        # Global features (all numerical, shape: (F_g,))
        X_g = torch.from_numpy(
            s2u(
                self.g_dset[idx][self.variables[self.global_object].inputs],
                dtype=None,
            )
        ).float()

        if self.global_object in self._norm:
            mean = self._norm[self.global_object]['mean']
            std = self._norm[self.global_object]['std']
            X_g = (X_g - mean) / std.clamp(min=1e-8)

        # Constituent features, one dict entry per object
        constituents = {}
        for obj_name in self.constituent_objects:
            X_cat = torch.from_numpy(
                s2u(
                    self.c_dsets[obj_name][idx][
                        self.variables[obj_name].inputs.categorical
                    ],
                    dtype=None,
                )
            )

            X_con = torch.from_numpy(
                s2u(
                    self.c_dsets[obj_name][idx][
                        self.variables[obj_name].inputs.continuous
                    ],
                    dtype=None,
                )
            ).float()

            # Normalise continuous features with pre-computed tensors
            if obj_name in self._norm:
                mean = self._norm[obj_name]['mean']
                std = self._norm[obj_name]['std']
                X_con = (X_con - mean) / std.clamp(
                    min=1e-8
                )  # avoid div by zero (limits values in a tensor)

            valid = torch.from_numpy(self.c_dsets[obj_name][idx]['valid'])

            constituents[obj_name] = {
                'categorical': X_cat,  # (N, F_cat)
                'continuous': X_con,  # (N, F_con)
                'valid': valid,  # (N,)
            }

        return {
            'label': label,
            'global': X_g,
            'constituents': constituents,
        }
