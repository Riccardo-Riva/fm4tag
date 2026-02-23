import lightning as L
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .datasets import DatasetCatCon, cat_con_collate_fn


class PT_FT_DataModule(L.LightningDataModule):
    """LightningDataModule for jet flavor tagging HDF5 datasets.

    Supports a two-phase transfer learning workflow controlled by ``phase``:

    * ``phase="pretrain"`` — ``train_dataloader()`` serves ``pretrain_file``.
      An optional ``pretrain_val_file`` enables validation during pretraining.
    * ``phase="finetune"`` (default) — ``train_dataloader()`` serves
      ``train_file``; ``val_dataloader()`` serves ``val_file``.

    Variables, norm_dict, and class_dict are shared across both phases.
    norm_dict and class_dict are loaded once in the main process and passed to
    each DatasetCatCon, so worker processes never touch YAML files.

    Expected cfg structure::

        global_object: jets
        constituent_objects: [tracks]

        variables:
            jets:
                inputs: [pt_btagJes, eta_btagJes]
                label: flavour_label
            tracks:
                inputs:
                    continuous: [d0, z0SinTheta, ...]
                    categorical: [numberOfPixelHits, ...]

        norm_dict:  /path/to/norm_dict.yaml
        class_dict: /path/to/class_dict.yaml

        # ── Pretrain phase ──────────────────────────────────────────────────
        pretrain_file:     /path/to/pretrain.h5
        pretrain_val_file: null          # uncomment to enable val during pretrain
        pretrain_dataloader:             # optional: overrides dataloader for pretrain
            batch_size: 2048

        # ── Finetune / supervised phase ─────────────────────────────────────
        train_file: /path/to/train.h5
        val_file:   /path/to/val.h5
        test_file:  /path/to/test.h5
        dataloader:
            batch_size:      1024
            num_workers:     4
            prefetch_factor: 2
            pin_memory:      true
    """

    def __init__(self, cfg: DictConfig, phase: str = 'finetune') -> None:
        super().__init__()
        if phase not in ('pretrain', 'finetune'):
            raise ValueError(f"phase must be 'pretrain' or 'finetune', got {phase!r}")
        self.cfg = cfg
        self.phase = phase
        self.save_hyperparameters()

        # Loaded once in the main process; shared across both phases.
        self._norm_dict = self._load_yaml(cfg.get('norm_dict'))
        self._class_dict = self._load_yaml(cfg.get('class_dict'))

        self._pretrain_dataset: DatasetCatCon | None = None
        self._pretrain_val_dataset: DatasetCatCon | None = None
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
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    def _dataset_kwargs(self) -> dict:
        """Shared dataset kwargs used by both pretrain and finetune phases."""
        return dict(
            variables=self.cfg.variables,
            global_object=self.cfg.global_object,
            constituent_objects=list(self.cfg.constituent_objects),
            norm_dict=self._norm_dict,
            class_dict=self._class_dict,
        )

    def _make_dataloader(
        self, dataset: DatasetCatCon, *, shuffle: bool, phase: str = 'finetune'
    ) -> DataLoader:
        # Use a phase-specific dataloader block if provided, else fall back to
        # the shared one (pretrain often benefits from a larger batch size).
        dl_key = 'pretrain_dataloader' if phase == 'pretrain' else 'dataloader'
        dl = self.cfg.get(dl_key) or self.cfg.get('dataloader') or {}
        num_workers = dl.get('num_workers', 4)
        return DataLoader(
            dataset,
            batch_size=dl.get('batch_size', 1024),
            shuffle=shuffle,
            # Drop last incomplete batch during training for stable batch stats.
            drop_last=shuffle,
            num_workers=num_workers,
            collate_fn=cat_con_collate_fn,
            prefetch_factor=dl.get('prefetch_factor', 2) if num_workers > 0 else None,
            persistent_workers=False,
            pin_memory=dl.get('pin_memory', True),
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

        Phase behaviour for ``stage="fit"``:

        * ``phase="pretrain"`` — creates the pretrain dataset from
          ``pretrain_file``, and optionally a pretrain val dataset from
          ``pretrain_val_file`` if that key is set in cfg.
        * ``phase="finetune"`` — creates train/val datasets from
          ``train_file`` / ``val_file``.

        ``stage="test"`` / ``"predict"`` always uses ``test_file`` regardless
        of phase, since evaluation against labelled data is a supervised task.
        """
        kwargs = self._dataset_kwargs()

        if stage in ('fit', 'all'):
            if self.phase == 'pretrain':
                pretrain_file = self.cfg.get('pretrain_file')
                if pretrain_file is None:
                    raise ValueError(
                        "phase='pretrain' requires cfg.pretrain_file to be set."
                    )
                self._pretrain_dataset = DatasetCatCon(
                    file_path=pretrain_file, **kwargs
                )
                pretrain_val_file = self.cfg.get('pretrain_val_file')
                if pretrain_val_file is not None:
                    self._pretrain_val_dataset = DatasetCatCon(
                        file_path=pretrain_val_file, **kwargs
                    )
            else:  # "finetune"
                self._train_dataset = DatasetCatCon(
                    file_path=self.cfg.train_file, **kwargs
                )
                self._val_dataset = DatasetCatCon(file_path=self.cfg.val_file, **kwargs)

        if stage in ('test', 'predict', 'all'):
            self._test_dataset = DatasetCatCon(file_path=self.cfg.test_file, **kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.phase == 'pretrain':
            return self._make_dataloader(
                self._pretrain_dataset, shuffle=True, phase='pretrain'
            )
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader | list:
        """Return the validation dataloader.

        During pretraining, returns an empty list (Lightning skips validation)
        when no ``pretrain_val_file`` was provided in cfg.
        """
        if self.phase == 'pretrain':
            if self._pretrain_val_dataset is None:
                return []  # Lightning treats [] as "no validation this phase"
            return self._make_dataloader(
                self._pretrain_val_dataset, shuffle=False, phase='pretrain'
            )
        return self._make_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self._test_dataset, shuffle=False)
