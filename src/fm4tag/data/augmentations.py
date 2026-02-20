import torch
import torch.nn.functional as F
from torch import nn


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
        encoder: :class:`saint_encoder` instance whose embedding tables are used.

    Returns:
        ``x_categ_enc`` – ``(N, F_cat, dim)`` float
        ``x_cont_enc``  – ``(N, F_con, dim)`` float, or ``None`` when there are
                          no continuous features.
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
            # Single shared MLP; apply to each feature independently.
            x_cont_enc = encoder.cont_MLP[0](x_cont.unsqueeze(-1))  # (N, F_con, dim)

        else:
            # No continuous embedding; caller receives None and should handle it.
            pass

    return x_categ_enc, x_cont_enc


def add_noise(
    x_categ: torch.Tensor,
    x_cont: torch.Tensor,
    lam: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix-style corruption on raw (pre-embedding) features.

    Each element is independently kept with probability ``1 - lam`` or
    replaced by the corresponding element of a randomly permuted sample.

    Args:
        x_categ: ``(N, F_cat)`` long tensor.
        x_cont:  ``(N, F_con)`` float tensor.
        lam:     Corruption fraction in ``[0, 1]``. ``0`` means no corruption,
                 ``1`` means fully random.

    Returns:
        Corrupted ``(x_categ_corr, x_cont_corr)`` with the same shapes and
        dtypes as the inputs.
    """
    N = x_categ.size(0)
    index = torch.randperm(N, device=x_categ.device)

    # Bernoulli mask: True = keep original, False = replace with shuffled sample.
    cat_keep = torch.bernoulli(
        (1.0 - lam)
        * torch.ones(x_categ.shape, dtype=torch.float, device=x_categ.device)
    ).bool()
    con_keep = torch.bernoulli((1.0 - lam) * torch.ones_like(x_cont)).bool()

    x_categ_corr = torch.where(cat_keep, x_categ, x_categ[index])
    x_cont_corr = torch.where(con_keep, x_cont, x_cont[index])

    return x_categ_corr, x_cont_corr


def mixup_data(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lam: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup augmentation in embedding space.

    Both ``x1`` and ``x2`` are mixed with a random permutation of themselves.
    Intended to be applied to *already-embedded* tensors (after
    :func:`embed_data`).

    Args:
        x1:  ``(N, F, dim)`` first embedded view.
        x2:  ``(N, F, dim)`` second embedded view.
        lam: Interpolation coefficient. The mixed output is
             ``lam * x + (1 - lam) * x[shuffle]``.

    Returns:
        ``(x1_mixed, x2_mixed)`` with the same shapes as inputs.
    """
    N = x1.size(0)
    index = torch.randperm(N, device=x1.device)

    x1_mixed = lam * x1 + (1.0 - lam) * x1[index]
    x2_mixed = lam * x2 + (1.0 - lam) * x2[index]

    return x1_mixed, x2_mixed
