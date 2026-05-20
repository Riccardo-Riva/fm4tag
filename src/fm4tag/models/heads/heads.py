import torch
from torch import nn

from ..mlp import MLP_dropout
from .transformers import Classifier_Transformer


class MultiStreamClassifierHead(nn.Module):
    """Multi-stream classification head for jet flavour tagging.

    Receives **pre-projected** embeddings from the caller (already passed
    through ``encoder.pt_mlp1``) and produces class logits:

    1. **Global stream**: ``(B, g_dim)`` — passed straight through.

    2. **Constituent stream** (per type):

       a. ``(B, C, c_dim_i)`` — pre-projected constituent embeddings
       b. Cross-constituent transformer → ``(B, C, c_dim_i)``
       c. Masked mean pool over valid constituents → ``(B, c_dim_i)``

    3. **Concatenate** all streams → ``(B, g_dim + sum(c_dim_i))``
    4. **Classification MLP** (2-layer FF) → ``(B, y_dim)``

    The expected calling convention (in :class:`~fm4tag.models.FinetuneModule`)
    is::

        X        = encoder(x)                   # (B, F, dim)
        z        = encoder.pt_mlp1(X.flatten(1))# (B, proj_out)  ← projection here
        logits   = head(z_global, [z_const], valid_masks)

    Args:
        global_proj_out:  Output dimension of ``global_encoder.pt_mlp1``.
        const_proj_outs:  Output dimension of each constituent encoder's
                          ``pt_mlp1``, one per constituent type.
        y_dim:            Number of output classes.
        mlp_dropout:      Dropout inside the final classification MLP.
        ff_dropout:       Dropout inside transformer feed-forward sub-layers.
        attn_dropout:     Dropout inside transformer attention.
        ff_mult:          Feed-forward expansion factor in the transformer.
        heads:            Number of attention heads per constituent transformer.
        dim_head:         Dimension per head.
        depth:            Transformer depth per constituent stream.
    """

    def __init__(
        self,
        *,
        global_proj_out: int,
        const_proj_outs: list[int],
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

        # Cross-constituent transformer per stream, sized to that stream's dim.
        self.const_transformer = nn.ModuleList(
            [
                Classifier_Transformer(
                    out_dim,
                    depth=depth,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                )
                for out_dim in const_proj_outs
            ]
        )

        # Final classification: concat of all streams → 2-layer FF → y_dim.
        in_dim = global_proj_out + sum(const_proj_outs)
        _hidden = max(global_proj_out, max(const_proj_outs, default=global_proj_out))
        self.cls_mlp = MLP_dropout(
            [in_dim, _hidden, y_dim], act=nn.ReLU(), dropout=mlp_dropout
        )

    def forward(
        self,
        global_z: torch.Tensor,
        constituent_zs: list[torch.Tensor],
        constituent_valids: list[torch.Tensor],
    ) -> torch.Tensor:
        """Classify a batch of jets using pre-projected stream embeddings.

        Args:
            global_z:           ``(B, g_dim)`` — output of
                                ``global_encoder.pt_mlp1``.
            constituent_zs:     List of ``(B, C, c_dim_i)`` tensors — output of
                                ``encoder.pt_mlp1`` scattered back over the
                                padded constituent grid, zeros at invalid slots.
            constituent_valids: List of ``(B, C)`` bool masks.

        Returns:
            ``(B, y_dim)`` class logits (unnormalised).
        """
        reprs = [global_z]

        for i, (z, valids) in enumerate(zip(constituent_zs, constituent_valids)):
            # Zero out padding slots before attention.
            z = torch.where(valids.unsqueeze(-1), z, 0.0)

            # Cross-constituent transformer.
            z = self.const_transformer[i](z, valids)

            # Masked mean pool over valid constituents → (B, c_dim_i)
            n_valids = valids.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            z = z.sum(dim=1) / n_valids

            reprs.append(z)

        return self.cls_mlp(torch.cat(reprs, dim=-1))
