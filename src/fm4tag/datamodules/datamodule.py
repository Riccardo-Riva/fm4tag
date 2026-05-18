import lightning as L
from torch.utils.data import DataLoader

import yaml

from ..datasets.datasets import DatasetCatCon, cat_con_collate_fn


class CatConDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset_path: str,
        val_dataset_path: str,
        test_dataset_path: str,
        variables: dict,
        global_object: str,
        constituent_objects: list[str],
        norm_dict_path: str | None = None,
        class_dict_path: str | None = None,
        # ── DataLoader shared args ───────────────────────────────────────
        batch_size: int = 1024,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # dataset options
        self._train_dataset_path = train_dataset_path
        self._val_dataset_path = val_dataset_path
        self._test_dataset_path = test_dataset_path
        self._variables = variables
        self._global_object = global_object
        self._constituent_objects = constituent_objects

        # dataloader options
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory

        # Loaded once in the main process; shared across both phases.
        self._norm_dict = self._load_yaml(norm_dict_path)
        self._class_dict = self._load_yaml(class_dict_path)

        self._train_dataset: DatasetCatCon | None = None
        self._val_dataset: DatasetCatCon | None = None
        self._test_dataset: DatasetCatCon | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: str | None) -> dict | None:
        """Load a YAML file and return a plain Python dict."""
        if path is None:
            return None
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _make_dataloader(self, dataset: DatasetCatCon, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            # Drop last incomplete batch during training for stable batch stats.
            drop_last=shuffle,
            num_workers=self._num_workers,
            collate_fn=cat_con_collate_fn,
            prefetch_factor=self._prefetch_factor if self._num_workers > 0 else None,
            # Keep workers alive between epochs so HDF5 handles are not reopened.
            persistent_workers=False, #self._num_workers > 0,
            pin_memory=self._pin_memory,
        )

    # ------------------------------------------------------------------
    # LightningDataModule protocol
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        """Instantiate datasets for the requested stage.

        Lightning calls this once per stage before the corresponding
        dataloader is requested::

            "fit"     -> train + val datasets  (phase-aware)
            "test"    -> test dataset
            "predict" -> test dataset (for inference without labels)
        """
        dataset_kwargs = dict(
            variables=self._variables,
            global_object=self._global_object,
            constituent_objects=self._constituent_objects,
            norm_dict=self._norm_dict,
            class_dict=self._class_dict,
        )

        if stage in ('fit', 'all'):
            if self._train_dataset_path is None:
                raise ValueError(
                    "train_dataset_path must be provided for the 'fit' stage"
                )
            self._train_dataset = DatasetCatCon(
                file_path=self._train_dataset_path, **dataset_kwargs
            )
            if self._val_dataset_path is None:
                raise ValueError(
                    "val_dataset_path must be provided for the 'fit' stage"
                )
            self._val_dataset = DatasetCatCon(file_path=self._val_dataset_path, **dataset_kwargs)

        if stage in ('test', 'predict', 'all'):
            if self._test_dataset_path is None:
                raise ValueError(
                    "test_dataset_path must be provided for the 'test' or 'predict' stage"
                )
            self._test_dataset = DatasetCatCon(file_path=self._test_dataset_path, **dataset_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self._test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._make_dataloader(self._test_dataset, shuffle=False)
