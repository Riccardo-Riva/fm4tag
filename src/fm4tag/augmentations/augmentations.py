"""Data augmentations and embedding utilities for fm4tag.

Two-stage augmentation pipeline for contrastive pretraining:

* **Raw stage** — applied to ``(x_categ, x_cont)`` *before* embedding.
  Operates on discrete indices and normalised floats directly.
  Example: :class:`CutMix`, :class:`ContinuousDilation`.

* **Latent stage** — applied to ``(x_cat_enc, x_con_enc, ...)`` *after*
  embedding but before the transformer.  Operates in continuous embedding
  space.  Example: :class:`Mixup`.

:class:`AugmentationPipeline` wires the two stages together and represents
one augmented *view* of the data. :class:`MultiViewAugmentation` holds a
list of pipelines and returns one augmented pair per pipeline, enabling
multi-view contrastive training.

Object-aware augmentations (:class:`ContinuousFeatureDilation`,
:class:`CategoricalShift`) store per-object precomputed state populated by
:meth:`AugmentationPipeline.setup_for_object`. Pass ``obj_name`` through
:meth:`AugmentationPipeline.apply_raw` so they can look up the right state.

Backward-compatible function aliases are kept at module bottom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from ..datasets import DatasetCatCon


# ---------------------------------------------------------------------------
# Embedding utility (not an augmentation)
# ---------------------------------------------------------------------------


def embed_data(
    x_categ: torch.Tensor,
    x_cont: torch.Tensor,
    encoder: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Embed raw categorical indices and continuous values using the encoder's
    embedding layers.

    This step happens *before* the encoder transformer forward pass:
    categorical integer indices are looked up in the embedding table, and
    continuous features are passed through per-feature MLPs (grouped Conv1d).

    Args:
        x_categ: ``(N, F_cat)`` long tensor of raw categorical feature indices.
        x_cont:  ``(N, F_con)`` float tensor of continuous feature values
                 (already normalised by the DataModule).
        encoder: :class:`~fm4tag.models.components.encoder.Encoder` instance
                 whose embedding tables are used.

    Returns:
        ``x_categ_enc`` – ``(N, F_cat, dim)`` float
        ``x_cont_enc``  – ``(N, F_con, dim)`` float, or ``None`` when there
                          are no continuous features.
    """
    x_categ = x_categ + encoder.categories_offset.type_as(x_categ)
    x_categ_enc = encoder.embeds(x_categ)  # (N, F_cat, dim)

    x_cont_enc: torch.Tensor | None = None
    if encoder.num_continuous > 0:
        if encoder.cont_embeddings == 'MLP':
            x = x_cont.unsqueeze(-1)  # (N, F_con, 1)
            h = F.relu(encoder.cont_fc1(x))  # (N, F_con*H, 1)
            out = encoder.cont_fc2(h)  # (N, F_con*dim, 1)
            x_cont_enc = out.view(
                x_cont.size(0), encoder.num_continuous, encoder.dim
            )  # (N, F_con, dim)

        elif encoder.cont_embeddings == 'pos_singleMLP':
            x_cont_enc = encoder.cont_MLP[0](x_cont.unsqueeze(-1))  # (N, F_con, dim)

    return x_categ_enc, x_cont_enc


# ---------------------------------------------------------------------------
# Raw-space augmentations (nn.Module)
#
# Interface: forward(x_categ, x_cont, obj_name='') -> (x_categ, x_cont)
#   x_categ : (B, [N,] F_cat) long  |  None for global object
#   x_cont  : (B, [N,] F_con) float
#   obj_name: forwarded by AugmentationPipeline.apply_raw; object-aware
#             augmentations use it to look up precomputed per-object state;
#             object-agnostic ones accept **kwargs and ignore it.
# ---------------------------------------------------------------------------


class CutMix(nn.Module):
    """CutMix-style corruption on raw (pre-embedding) features.

    Each element is independently kept with probability ``1 - lam`` or
    replaced by the corresponding element of a randomly permuted sample.
    Works for both constituent objects (categorical + continuous) and the
    global object (continuous-only, pass ``x_categ=None``).

    Args:
        lam: Corruption fraction in ``[0, 1]``.
    """

    def __init__(self, lam: float = 0.1) -> None:
        super().__init__()
        self.lam = lam

    def forward(
        self,
        x_categ: torch.Tensor | None,
        x_cont: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        N = x_cont.size(0)
        index = torch.randperm(N, device=x_cont.device)

        x_categ_corr: torch.Tensor | None = None
        if x_categ is not None:
            cat_keep = torch.bernoulli(
                (1.0 - self.lam)
                * torch.ones(x_categ.shape, dtype=torch.float, device=x_categ.device)
            ).bool()
            x_categ_corr = torch.where(cat_keep, x_categ, x_categ[index])

        con_keep = torch.bernoulli((1.0 - self.lam) * torch.ones_like(x_cont)).bool()
        x_cont_corr = torch.where(con_keep, x_cont, x_cont[index])

        return x_categ_corr, x_cont_corr


class ContinuousDilation(nn.Module):
    """Scale all continuous features by a constant factor α.

    A simple multiplicative augmentation that stretches or shrinks the entire
    continuous feature vector uniformly.  Categorical features are unchanged.

    Args:
        alpha: Scale factor applied to every continuous feature value.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        x_categ: torch.Tensor | None,
        x_cont: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        return x_categ, x_cont * self.alpha


class ContinuousFeatureDilation(nn.Module):
    """Scale a named subset of continuous features by a constant factor α.

    Feature names are specified at construction time as strings.  Before the
    first forward pass, call :meth:`AugmentationPipeline.setup_for_object`
    (or :meth:`setup` directly) for every object the pipeline will process.
    The mapping from feature name to column index is stored per ``obj_name``
    so the same augmentation instance handles multiple objects correctly.

    Features absent from a given object's continuous-feature list are silently
    skipped for that object (the augmentation becomes a partial no-op).

    Args:
        features: Names of continuous features to dilate.
        alpha:    Scale factor applied to the selected features.
    """

    def __init__(self, features: list[str], alpha: float) -> None:
        super().__init__()
        self.feature_names = list(features)
        self.alpha = alpha
        # keyed by obj_name -> list of column indices into x_cont
        self._indices_by_obj: dict[str, list[int]] = {}

    def setup(self, obj_name: str, continuous_features: list[str], **kwargs) -> None:
        """Resolve feature names to column indices for *obj_name*."""
        self._indices_by_obj[obj_name] = [
            continuous_features.index(f)
            for f in self.feature_names
            if f in continuous_features
        ]

    def forward(
        self,
        x_categ: torch.Tensor | None,
        x_cont: torch.Tensor,
        obj_name: str = '',
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        indices = self._indices_by_obj.get(obj_name, [])
        if not indices:
            return x_categ, x_cont
        idx = torch.tensor(indices, device=x_cont.device)
        x_cont = x_cont.clone()
        x_cont[..., idx] = x_cont[..., idx] * self.alpha
        return x_categ, x_cont


class CategoricalShift(nn.Module):
    """Randomly shift categorical feature values by ±1.

    Each categorical feature element is independently shifted up or down by
    one class index with probability ``p``, then clamped to the valid range
    ``[0, n_classes - 1]``.  Requires :meth:`setup` (called automatically by
    :meth:`AugmentationPipeline.setup_for_object`) to provide per-object class
    counts so that the clamp bounds are correct.

    If ``x_categ`` is ``None`` (global object) or the object has not been set
    up yet, the augmentation is a no-op.

    Args:
        p: Per-element shift probability in ``[0, 1]``.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        # keyed by obj_name -> (F_cat,) long tensor of max valid class indices
        self._max_vals_by_obj: dict[str, torch.Tensor] = {}

    def setup(self, obj_name: str, n_classes: list[int], **kwargs) -> None:
        """Store max valid class index (n_classes - 1) per feature for *obj_name*."""
        if n_classes:
            self._max_vals_by_obj[obj_name] = torch.tensor(
                [n - 1 for n in n_classes], dtype=torch.long
            )

    def forward(
        self,
        x_categ: torch.Tensor | None,
        x_cont: torch.Tensor,
        obj_name: str = '',
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if x_categ is None:
            return x_categ, x_cont
        max_vals = self._max_vals_by_obj.get(obj_name)
        if max_vals is None:
            return x_categ, x_cont
        max_vals = max_vals.to(x_categ.device)
        delta = torch.randint(-1, 2, x_categ.shape, device=x_categ.device)
        mask = torch.bernoulli(
            self.p * torch.ones_like(x_categ, dtype=torch.float)
        ).bool()
        shifted = x_categ + delta * mask
        min_vals = torch.zeros_like(max_vals)
        return shifted.clamp(min=min_vals, max=max_vals), x_cont


# ---------------------------------------------------------------------------
# Latent-space augmentations (nn.Module)
# ---------------------------------------------------------------------------


class Mixup(nn.Module):
    """Mixup augmentation in embedding space.

    All input tensors are mixed with a random permutation of themselves using
    the *same* permutation index.  Accepts any number of tensors so it works
    for both the constituent case (cat + con embeddings) and the global case
    (single full embedding).

    Args:
        lam: Interpolation coefficient.  The mixed output is
             ``lam * x + (1 - lam) * x[shuffle]``.
    """

    def __init__(self, lam: float = 0.1) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        N = tensors[0].size(0)
        index = torch.randperm(N, device=tensors[0].device)
        return tuple(self.lam * t + (1.0 - self.lam) * t[index] for t in tensors)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AugmentationPipeline(nn.Module):
    """Two-stage augmentation pipeline representing a single augmented view.

    Chains raw-space augmentations (applied before embedding) and latent-space
    augmentations (applied after embedding) in order.

    Object-aware augmentations need to know which continuous feature sits at
    which column index in the batch tensor.  Call :meth:`setup_for_object`
    once per object at model initialisation time, then pass ``obj_name`` to
    :meth:`apply_raw` during training.

    Build from config using ``hydra.utils.instantiate``::

        raw_augs    = [instantiate(a) for a in view_cfg.get('raw', [])]
        latent_augs = [instantiate(a) for a in view_cfg.get('latent', [])]
        pipeline    = AugmentationPipeline(raw_augs, latent_augs)

    Args:
        raw:    List of raw-stage ``nn.Module`` augmentations.
                Each must implement
                ``forward(x_categ, x_cont, obj_name='', **kwargs)``.
        latent: List of latent-stage ``nn.Module`` augmentations.
                Each must implement ``forward(*tensors)``.
    """

    def __init__(
        self,
        raw: list[nn.Module] | None = None,
        latent: list[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.raw = nn.ModuleList(raw or [])
        self.latent = nn.ModuleList(latent or [])

    def setup_for_object(
        self,
        obj_name: str,
        dataset: 'DatasetCatCon',
    ) -> None:
        """Prepare object-aware augmentations for a specific object.

        Extracts the ordered continuous feature name list and per-categorical-
        feature class counts from the dataset variables, then calls
        ``aug.setup(obj_name, ...)`` on every raw augmentation that exposes
        that method.

        Args:
            obj_name: Dataset object name, e.g. ``"jets"`` or ``"tracks"``.
            dataset:  Instantiated :class:`~fm4tag.datasets.DatasetCatCon`.
        """
        variables = dataset.variables
        if obj_name == dataset.global_object:
            continuous_features = list(variables[obj_name].inputs)
            n_classes: list[int] = []
        else:
            continuous_features = list(variables[obj_name].inputs.continuous)
            n_classes = [
                len(v) for v in variables[obj_name].inputs.cat_classes.values()
            ]

        for aug in self.raw:
            if hasattr(aug, 'setup'):
                aug.setup(
                    obj_name=obj_name,
                    continuous_features=continuous_features,
                    n_classes=n_classes,
                )

    def apply_raw(
        self,
        x_categ: torch.Tensor | None,
        x_cont: torch.Tensor,
        obj_name: str = '',
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply all raw-stage augmentations in order.

        Args:
            x_categ:  ``(B, [N,] F_cat)`` long or ``None`` (global object).
            x_cont:   ``(B, [N,] F_con)`` float.
            obj_name: Forwarded to object-aware augmentations so they can
                      resolve per-object precomputed state.
        """
        for aug in self.raw:
            x_categ, x_cont = aug(x_categ, x_cont, obj_name=obj_name)
        return x_categ, x_cont

    def apply_latent(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply all latent-stage augmentations in order."""
        result: tuple[torch.Tensor, ...] = tensors
        for aug in self.latent:
            result = aug(*result)
        return result


# ---------------------------------------------------------------------------
# Multi-view augmentation
# ---------------------------------------------------------------------------


class MultiViewAugmentation(nn.Module):
    """Produces N augmented views of a raw-feature batch.

    Holds one :class:`AugmentationPipeline` per view.  A forward call returns
    a list of ``(x_categ_aug, x_cont_aug)`` tuples — one per pipeline — that
    form the positive pairs for contrastive loss computation.  The clean
    (anchor) view is not included; the training step handles it separately.

    Typical setup workflow::

        # Build once at the start of training
        aug = build_aug_module(cfg)
        aug.setup_for_dataset(dataset)   # resolves feature names

        # In the training loop, per object:
        views = aug(x_categ, x_cont, obj_name=obj_name)
        # views[i] is (x_categ_aug_i, x_cont_aug_i)

    Args:
        pipelines: List of :class:`AugmentationPipeline` instances.
    """

    def __init__(self, pipelines: list[AugmentationPipeline]) -> None:
        super().__init__()
        self.pipelines = nn.ModuleList(pipelines)

    @property
    def n_views(self) -> int:
        """Number of augmented views produced per forward call."""
        return len(self.pipelines)

    def setup_for_dataset(self, dataset: 'DatasetCatCon') -> None:
        """Call :meth:`~AugmentationPipeline.setup_for_object` for every object.

        Must be called once after the dataset is instantiated and before
        training, so that object-aware augmentations have their per-object
        state populated.

        Args:
            dataset: Instantiated :class:`~fm4tag.datasets.DatasetCatCon`.
        """
        all_objects = [dataset.global_object] + list(dataset.constituent_objects)
        for pipeline in self.pipelines:
            for obj_name in all_objects:
                pipeline.setup_for_object(obj_name, dataset)

    def forward(
        self,
        x_categ: torch.Tensor | None,
        x_cont: torch.Tensor,
        obj_name: str = '',
    ) -> list[tuple[torch.Tensor | None, torch.Tensor]]:
        """Return one raw-augmented pair per pipeline.

        Args:
            x_categ:  ``(B, [N,] F_cat)`` long or ``None``.
            x_cont:   ``(B, [N,] F_con)`` float.
            obj_name: Forwarded to each pipeline's :meth:`~AugmentationPipeline.apply_raw`.

        Returns:
            List of ``(x_categ_aug, x_cont_aug)`` tuples, length == :attr:`n_views`.
        """
        return [
            pipeline.apply_raw(x_categ, x_cont, obj_name=obj_name)
            for pipeline in self.pipelines
        ]


# ---------------------------------------------------------------------------
# Backward-compatible function aliases
# ---------------------------------------------------------------------------


def add_noise(
    x_categ: torch.Tensor | None,
    x_cont: torch.Tensor,
    lam: float = 0.1,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Alias for :class:`CutMix` — kept for backward compatibility."""
    return CutMix(lam)(x_categ, x_cont)


def mixup_data(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lam: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias for :class:`Mixup` on two tensors — kept for backward compatibility."""
    return Mixup(lam)(x1, x2)  # type: ignore[return-value]
