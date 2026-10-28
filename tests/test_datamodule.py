"""``CatConDataModule(train_max_rows=N)``: training reads only the first N rows of
its file (a low-label regime); validation and test stay whole."""

from __future__ import annotations

import numpy as np

from fm4tag.datamodules import CatConDataModule


class _Rows:
    """Stands in for a DatasetCatCon: the loaders only need its length."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


def _datamodule(**kwargs) -> CatConDataModule:
    dm = CatConDataModule(
        train_dataset_path='unused',
        val_dataset_path='unused',
        test_dataset_path='unused',
        variables={},
        global_object='jets',
        constituent_objects=[],
        batch_size=10,
        num_workers=0,
        reads_per_batch=2,
        **kwargs,
    )
    dm._train_dataset, dm._val_dataset = _Rows(1000), _Rows(1000)
    return dm


def _rows(loader) -> np.ndarray:
    return np.concatenate([np.arange(s, e) for b in loader.sampler for (s, e) in b])


def test_train_max_rows_limits_only_the_training_loader():
    dm = _datamodule(train_max_rows=100)
    train = dm.train_dataloader()
    assert len(train.sampler) == 10
    assert _rows(train).max() < 100
    assert np.array_equal(_rows(dm.val_dataloader()), np.arange(1000))


def test_train_max_rows_defaults_to_the_whole_file():
    train = _datamodule().train_dataloader()
    assert len(train.sampler) == 100
    assert _rows(train).max() >= 100
