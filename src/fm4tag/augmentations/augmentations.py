"""Data augmentations and embedding utilities for fm4tag.

Two-stage augmentation pipeline for contrastive pretraining:

* **Raw stage** — applied to ``(x_categ, x_cont)`` *before* embedding.
  Operates on discrete indices and normalised floats directly.
  Example: :class:`CutMix`.

* **Latent stage** — applied to ``(x_cat_enc, x_con_enc, ...)`` *after*
  embedding but before the transformer.  Operates in continuous embedding
  space.  Example: :class:`Mixup`.

:class:`AugmentationPipeline` wires the two stages together and is the
object passed to :class:`~fm4tag.models.PretrainModule`.

Backward-compatible function aliases are kept at the bottom of this module
so existing notebooks and scripts continue to work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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
# Raw-space augmentations
# ---------------------------------------------------------------------------


class CutMix:
    """CutMix-style corruption on raw (pre-embedding) features.

    Each element is independently kept with probability ``1 - lam`` or
    replaced by the corresponding element of a randomly permuted sample.
    Works for both constituent objects (categorical + continuous) and the
    global object (continuous-only, pass ``x_categ=None``).

    Args:
        lam: Corruption fraction in ``[0, 1]``.
    """

    def __init__(self, lam: float = 0.1) -> None:
        self.lam = lam

    def __call__(
        self,
        x_categ: torch.Tensor | None,
        x_cont: torch.Tensor,
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


# ---------------------------------------------------------------------------
# Latent-space augmentations
# ---------------------------------------------------------------------------


class Mixup:
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
        self.lam = lam

    def __call__(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        N = tensors[0].size(0)
        index = torch.randperm(N, device=tensors[0].device)
        return tuple(self.lam * t + (1.0 - self.lam) * t[index] for t in tensors)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AugmentationPipeline:
    """Two-stage augmentation pipeline for contrastive pretraining.

    Chains raw-space augmentations (applied before embedding) and latent-space
    augmentations (applied after embedding) in order.  The pipeline is called
    once per view to produce the corrupted version; the clean view is always
    the original batch.

    Build from config using ``hydra.utils.instantiate``::

        raw_augs   = [instantiate(a) for a in cfg.augmentation.raw]
        latent_augs = [instantiate(a) for a in cfg.augmentation.latent]
        pipeline = AugmentationPipeline(raw_augs, latent_augs)

    Args:
        raw:    List of callables ``(x_categ, x_cont) → (x_categ, x_cont)``.
                ``x_categ`` may be ``None`` for the global object.
        latent: List of callables ``(*tensors) → tuple[Tensor, ...]``.
                Each callable receives all embedding tensors at once and must
                return a tuple of the same length.
    """

    def __init__(
        self,
        raw: list | None = None,
        latent: list | None = None,
    ) -> None:
        self.raw: list = raw or []
        self.latent: list = latent or []

    def apply_raw(
        self,
        x_categ: torch.Tensor | None,
        x_cont: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply all raw-stage augmentations in order."""
        for aug in self.raw:
            x_categ, x_cont = aug(x_categ, x_cont)
        return x_categ, x_cont

    def apply_latent(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply all latent-stage augmentations in order."""
        result: tuple[torch.Tensor, ...] = tensors
        for aug in self.latent:
            result = aug(*result)
        return result


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
