from numpy.lib.recfunctions import structured_to_unstructured as s2u

import h5py
import numpy as np
import os
import torch
from h5py import h5s, h5t
from torch.utils.data import Dataset


class _SpanReader:
    """Read row spans of one HDF5 dataset into a single preallocated buffer.

    Exists to avoid a per-read leak in ``h5py.Dataset.__getitem__``: for a
    compound (structured) dtype the fast path is skipped and every call runs
    ``mtype = h5t.py_create(new_dtype)``, which retains ~21 Python objects
    (~0.9 kB) that ``gc`` never reclaims — measured with h5py 3.15.1 / HDF5
    1.14.6.  ``read_direct`` goes through the same ``py_create`` and leaks
    identically; creating the memory type ONCE and driving ``DatasetID.read``
    with it retains nothing (measured: 0 blocks/read).

    At 16 reads per batch (2 objects x ``reads_per_batch``) and ~1.3 k batches
    per worker per epoch, the leak was ~180 MB per worker per epoch — with 20
    persistent workers per job that is ~8 GB over 26 epochs, which is what
    pushed the 32 GB cgroup over the limit roughly a day into every training.

    Reading straight into one buffer also drops the per-batch
    ``np.concatenate`` that used to stitch the spans together.
    """

    def __init__(self, dset: h5py.Dataset) -> None:
        self._dsid = dset.id
        self._dtype = dset.dtype
        # Per-row shape: () for `jets`, (C,) for the constituent objects.
        self._tail = tuple(dset.shape[1:])
        self._zeros = (0,) * len(self._tail)
        # The two objects that must NOT be rebuilt per read.
        self._mtype = h5t.py_create(self._dtype)
        self._fspace = self._dsid.get_space()
        # Memory dataspaces are cheap but not free; a loader only ever asks for
        # one or two distinct row counts (full batch, and a short final batch).
        self._mspaces: dict[int, h5s.SpaceID] = {}

    def _mspace(self, total: int) -> h5s.SpaceID:
        mspace = self._mspaces.get(total)
        if mspace is None:
            mspace = h5s.create_simple((total,) + self._tail)
            self._mspaces[total] = mspace
        return mspace

    def read(self, spans: list[tuple[int, int]]) -> np.ndarray:
        """Read and concatenate contiguous row ranges, one HDF5 read each."""
        total = sum(stop - start for start, stop in spans)
        out = np.empty((total,) + self._tail, dtype=self._dtype)
        mspace = self._mspace(total)

        offset = 0
        for start, stop in spans:
            count = (stop - start,) + self._tail
            self._fspace.select_hyperslab((start,) + self._zeros, count)
            mspace.select_hyperslab((offset,) + self._zeros, count)
            self._dsid.read(mspace, self._fspace, out, self._mtype)
            offset += stop - start
        return out


class DatasetCatCon(Dataset):
    """Batch-at-a-time HDF5 dataset.

    ``__getitem__`` takes a list of contiguous ``(start, stop)`` spans (produced
    by :class:`~fm4tag.datasets.samplers.BatchSliceSampler`) and returns the
    fully collated batch — the DataLoader runs with ``batch_size=None`` and no
    ``collate_fn``, so there is no per-sample path at all.

    This is deliberate, not an optimisation detail.  The file is chunked and lzf
    compressed (``jets`` 3174 rows/chunk, ``tracks`` 100 rows/chunk), so h5py
    cannot read a single row: it decompresses the whole enclosing chunk.  The
    previous per-row ``__getitem__`` therefore decompressed ~485 MB to build a
    2.45 MB batch (13.4 s/batch measured) where a contiguous read costs 26.6 ms.
    Reading spans also lets normalisation and dtype casting run once per batch as
    vectorised numpy, instead of a few thousand tiny per-sample tensor ops.

    Output shapes (batch size B, constituent slots C):
        label       : (B,)          long
        global      : (B, F_g)      float32
        constituents: { obj_name: { categorical: (B, C, F_cat)  long
                                    continuous:  (B, C, F_con)  float32
                                    valid:       (B, C)         bool } }
    """

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

        # Field name lists, materialised once — s2u is called with these per
        # batch and omegaconf ListConfig would be re-resolved on every call.
        self._g_fields = list(self.variables[self.global_object]['inputs'])
        self._c_fields = {
            name: {
                'categorical': list(self.variables[name]['inputs']['categorical']),
                'continuous': list(self.variables[name]['inputs']['continuous']),
            }
            for name in self.constituent_objects
        }

        # Pre-build per-object mean/std arrays so the batch normalisation is a
        # single broadcast subtract/divide.
        self._build_norm_arrays(norm_dict)

        with h5py.File(self.file_path, 'r') as file:
            print(f'\nDatasetCatCon: {self.file_path}')
            self.len = file[self.global_object].shape[0]
            print(f'  samples : {self.len}')

        self.file = None  # lazy file opening: one handle opened per worker

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _build_norm_arrays(self, norm_dict: dict | None) -> None:
        """Convert norm_dict to float32 arrays broadcastable over a batch.

        norm_dict expected format::

            { "tracks": { "d0": {"mean": 0.0, "std": 1.0}, ... }, ... }

        Stored as ``self._norm[obj_name]["mean" | "std"]`` of shape ``(F_con,)``,
        which broadcasts against both ``(B, F_g)`` and ``(B, C, F_con)``.
        """
        self._norm: dict[str, dict[str, np.ndarray]] = {}
        if norm_dict is None:
            return

        all_objects = [self.global_object] + list(self.constituent_objects)
        for obj_name in all_objects:
            if obj_name not in norm_dict:
                continue
            obj_norm = norm_dict[obj_name]
            features = (
                self._g_fields
                if obj_name == self.global_object
                else self._c_fields[obj_name]['continuous']
            )
            self._norm[obj_name] = {
                'mean': np.array(
                    [obj_norm[f]['mean'] for f in features], dtype=np.float32
                ),
                # Pre-clamped so the divide needs no per-batch guard.
                'std': np.maximum(
                    np.array([obj_norm[f]['std'] for f in features], dtype=np.float32),
                    1e-8,
                ),
            }

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.len

    def _open_file(self) -> None:
        """Open the HDF5 file once per worker process and cache dataset handles.

        The readers built here own the cached memory datatype that keeps the
        per-read path allocation-free — see :class:`_SpanReader`.
        """
        self.file = h5py.File(self.file_path, 'r', swmr=True, libver='latest')
        self.g_dset = _SpanReader(self.file[self.global_object])
        self.c_dsets = {
            name: _SpanReader(self.file[name]) for name in self.constituent_objects
        }

    def __getitem__(self, spans) -> dict:
        """Return one collated batch.

        Args:
            spans: List of ``(start, stop)`` row ranges, or a single such tuple.
        """
        if self.file is None:
            self._open_file()

        if isinstance(spans, tuple) and len(spans) == 2 and np.isscalar(spans[0]):
            spans = [spans]

        # ── Global object ────────────────────────────────────────────────
        g_records = self.g_dset.read(spans)  # (B,) structured

        labels = torch.from_numpy(
            np.ascontiguousarray(g_records[self.label_name], dtype=np.int64)
        )

        X_g = s2u(g_records[self._g_fields], dtype=np.float32)  # (B, F_g)
        if self.global_object in self._norm:
            norm = self._norm[self.global_object]
            X_g = (X_g - norm['mean']) / norm['std']
        globals_ = torch.from_numpy(np.ascontiguousarray(X_g))

        # ── Constituent objects ──────────────────────────────────────────
        constituents = {}
        for obj_name in self.constituent_objects:
            c_records = self.c_dsets[obj_name].read(spans)  # (B, C)
            fields = self._c_fields[obj_name]

            X_cat = s2u(c_records[fields['categorical']], dtype=np.int64)
            X_con = s2u(c_records[fields['continuous']], dtype=np.float32)

            if obj_name in self._norm:
                norm = self._norm[obj_name]
                X_con = (X_con - norm['mean']) / norm['std']

            constituents[obj_name] = {
                'categorical': torch.from_numpy(np.ascontiguousarray(X_cat)),
                'continuous': torch.from_numpy(np.ascontiguousarray(X_con)),
                'valid': torch.from_numpy(
                    np.ascontiguousarray(c_records['valid'], dtype=np.bool_)
                ),
            }

        return {
            'label': labels,
            'global': globals_,
            'constituents': constituents,
        }
