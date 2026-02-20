import torch
from torch import nn

from .model import MLP_dropout, Classifier_Transformer


class ClassifierHead(nn.Module):
    """Attention-based classification head for jet flavour tagging.

    Operates on the per-constituent encoder outputs produced by
    :class:`saint_encoder`.  It extracts a per-constituent summary vector
    (the first feature token, index 0, which plays the role of a CLS token),
    runs cross-constituent attention, mean-pools over valid constituents, and
    produces final class logits.

    This module is intentionally encoder-agnostic: it only requires that the
    encoder output has shape ``(B, C, F, dim)`` and that the first token along
    the ``F`` dimension is used as the per-constituent summary.

    Args:
        dim:          Embedding dimension (must match the encoder's ``dim``).
        y_dim:        Number of output classes.
        mlp_dropout:  Dropout rate applied inside the MLP layers.
        ff_dropout:   Dropout rate inside feed-forward sub-layers of the
                      cross-constituent transformer.
        attn_dropout: Dropout rate inside the attention sub-layers.
        ff_mult:      Feed-forward expansion factor.
        heads:        Number of attention heads in the cross-constituent
                      transformer.
        dim_head:     Dimension per head.
        depth:        Number of transformer layers.
    """

    def __init__(
        self,
        *,
        dim: int,
        y_dim: int = 2,
        mlp_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
        heads: int = 8,
        dim_head: int = 16,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.mlpphi = MLP_dropout([dim, 512, dim], dropout=mlp_dropout)
        self.transformer = Classifier_Transformer(
            dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
        )
        self.mlpfory = MLP_dropout([dim, 512, 256, y_dim], dropout=mlp_dropout)

    def forward(self, x: torch.Tensor, valids: torch.Tensor) -> torch.Tensor:
        """Classify a batch of jets.

        Args:
            x:      ``(B, C, F, dim)`` encoder outputs for all jets and their
                    constituents. The token at ``F``-index 0 is used as the
                    per-constituent summary vector.
            valids: ``(B, C)`` bool mask — ``True`` for valid constituents,
                    ``False`` for padding.

        Returns:
            ``(B, y_dim)`` class logits (unnormalised).
        """
        x = x[:, :, 0, :]                              # (B, C, dim)
        x = self.mlpphi(x)                             # (B, C, dim)

        # Zero out padding positions before attention.
        x = torch.where(valids.unsqueeze(-1), x, 0.0)  # (B, C, dim)
        x = self.transformer(x, valids)                 # (B, C, dim)

        # Masked mean pooling over valid constituents.
        n_valids = valids.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # (B, 1)
        x = x.sum(dim=1) / n_valids                    # (B, dim)

        return self.mlpfory(x)                          # (B, y_dim)
