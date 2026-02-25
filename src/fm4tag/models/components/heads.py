import torch
from torch import nn

from .mlp import MLP_dropout, simple_MLP
from .transformers import Classifier_Transformer


class MultiStreamClassifierHead(nn.Module):
    """Multi-stream classification head for jet flavour tagging.

    Accepts encoder outputs for every object (global + all constituent types)
    and produces class logits via the following pipeline:

    1. **Global stream**: mean-pool the F_g feature tokens → 2-layer FF → ``(B, dim)``
    2. **Constituent stream** (per type):

       a. Mean-pool the F feature tokens per constituent → ``(B, C, dim)``
       b. DeepSet phi (per-constituent MLP) → ``(B, C, dim)``
       c. Cross-constituent transformer → ``(B, C, dim)``
       d. Masked mean pool over valid constituents → ``(B, dim)``

    3. **Concatenate** all streams → ``(B, (1 + n_constituent_types) × dim)``
    4. **Classification MLP** (2-layer FF) → ``(B, y_dim)``

    Args:
        dim:                 Embedding dimension (must match all encoders).
        n_constituent_types: Number of constituent object types.
        y_dim:               Number of output classes.
        mlp_dropout:         Dropout applied inside MLP layers.
        ff_dropout:          Dropout inside transformer feed-forward sub-layers.
        attn_dropout:        Dropout inside transformer attention.
        ff_mult:             Feed-forward expansion factor in the transformer.
        heads:               Number of attention heads per constituent transformer.
        dim_head:            Dimension per head.
        depth:               Transformer depth per constituent stream.
    """

    def __init__(
        self,
        *,
        dim: int,
        n_constituent_types: int,
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

        # Global stream: mean over F_g tokens → 2-layer FF → (B, dim)
        self.global_agg = simple_MLP([dim, 2 * dim, dim])

        # Per constituent type: phi + cross-constituent transformer
        self.const_phi = nn.ModuleList(
            [
                MLP_dropout([dim, 2 * dim, dim], act=nn.ReLU(), dropout=mlp_dropout)
                for _ in range(n_constituent_types)
            ]
        )
        self.const_transformer = nn.ModuleList(
            [
                Classifier_Transformer(
                    dim,
                    depth=depth,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(n_constituent_types)
            ]
        )

        # Final classification: concat of all streams → 2-layer FF → y_dim
        in_dim = (1 + n_constituent_types) * dim
        self.cls_mlp = MLP_dropout(
            [in_dim, dim, y_dim], act=nn.ReLU(), dropout=mlp_dropout
        )

    def forward(
        self,
        global_enc: torch.Tensor,
        constituent_encs: list[torch.Tensor],
        constituent_valids: list[torch.Tensor],
    ) -> torch.Tensor:
        """Classify a batch of jets using all object streams.

        Args:
            global_enc:          ``(B, F_g, dim)`` output of :class:`GlobalEncoder`.
            constituent_encs:    List of ``(B, C, F, dim)`` tensors — one per
                                 constituent type, zeros at padding positions.
            constituent_valids:  List of ``(B, C)`` bool masks — ``True`` for
                                 valid constituents, one per constituent type.

        Returns:
            ``(B, y_dim)`` class logits (unnormalised).
        """
        # ── Global stream ────────────────────────────────────────────────────
        # Mean over F_g feature tokens → (B, dim) → 2-layer FF
        g = global_enc.mean(dim=1)  # (B, dim)
        g = self.global_agg(g)  # (B, dim)

        reprs = [g]

        # ── Constituent streams ───────────────────────────────────────────────
        for i, (enc, valids) in enumerate(zip(constituent_encs, constituent_valids)):
            # Mean-pool F feature tokens per constituent: (B, C, F, dim) → (B, C, dim)
            x = enc.mean(dim=2)  # (B, C, dim)

            # DeepSet phi: per-constituent MLP
            x = self.const_phi[i](x)  # (B, C, dim)

            # Zero out padding constituents before cross-constituent attention.
            x = torch.where(valids.unsqueeze(-1), x, 0.0)  # (B, C, dim)

            # Cross-constituent transformer.
            x = self.const_transformer[i](x, valids)  # (B, C, dim)

            # Masked mean pool over valid constituents.
            n_valids = valids.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # (B, 1)
            x = x.sum(dim=1) / n_valids  # (B, dim)

            reprs.append(x)

        # ── Concatenate streams and classify ─────────────────────────────────
        x = torch.cat(reprs, dim=-1)  # (B, (1 + n_const) * dim)
        return self.cls_mlp(x)  # (B, y_dim)
