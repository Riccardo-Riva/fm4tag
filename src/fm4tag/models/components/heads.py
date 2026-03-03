import torch
from torch import nn

from .mlp import MLP_dropout, simple_MLP
from .transformers import Classifier_Transformer


class MultiStreamClassifierHead(nn.Module):
    """Multi-stream classification head for jet flavour tagging.

    Accepts encoder outputs for every object (global + all constituent types)
    and produces class logits via the following pipeline:

    1. **Global stream**:
       ``(B, F_g, dim)`` → flatten → ``(B, F_g·dim)`` → 2-layer FF → ``(B, cls_dim)``

    2. **Constituent stream** (per type):

       a. ``(B, C, F, dim)`` → flatten tokens → ``(B, C, F·dim)``
       b. 2-layer FF (per-constituent) → ``(B, C, cls_dim)``
       c. Cross-constituent transformer → ``(B, C, cls_dim)``
       d. Masked mean pool over valid constituents → ``(B, cls_dim)``

    3. **Concatenate** all streams → ``(B, (1 + n_constituent_types) × cls_dim)``
    4. **Classification MLP** (2-layer FF) → ``(B, y_dim)``

    Args:
        dim:                   Embedding dimension from the encoders.
        n_global_features:     Number of global feature tokens ``F_g``
                               (i.e. ``global_enc.shape[1]``).
        n_constituent_features: List with the number of feature tokens ``F``
                               for each constituent type
                               (i.e. ``[F_cat + F_con, ...]``).
        y_dim:                 Number of output classes.
        cls_dim:               Dimension of the projected representation fed into
                               the cross-constituent transformer and the final MLP.
                               ``None`` → equals ``dim``.
        mlp_dropout:           Dropout applied inside MLP layers.
        ff_dropout:            Dropout inside transformer feed-forward sub-layers.
        attn_dropout:          Dropout inside transformer attention.
        ff_mult:               Feed-forward expansion factor in the transformer.
        heads:                 Number of attention heads per constituent transformer.
        dim_head:              Dimension per head.
        depth:                 Transformer depth per constituent stream.
    """

    def __init__(
        self,
        *,
        dim: int,
        n_global_features: int,
        n_constituent_features: list[int],
        y_dim: int = 2,
        cls_dim: int | None = None,
        mlp_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
        heads: int = 8,
        dim_head: int = 16,
        depth: int = 3,
    ) -> None:
        super().__init__()

        _cls_dim = cls_dim if cls_dim is not None else dim

        # Global stream: flatten F_g*dim → 2-layer FF → (B, cls_dim)
        global_flat = n_global_features * dim
        self.global_proj = simple_MLP([global_flat, global_flat // 2, _cls_dim])

        # Per constituent type: flatten F*dim → 2-layer FF → cls_dim, then transformer
        self.const_proj = nn.ModuleList(
            [
                MLP_dropout(
                    [
                        n_feat * dim,
                        n_feat * dim // 2,
                        _cls_dim,
                    ],  # TODO: tune hidden dim?
                    act=nn.ReLU(),
                    dropout=mlp_dropout,
                )
                for n_feat in n_constituent_features
            ]
        )
        self.const_transformer = nn.ModuleList(
            [
                Classifier_Transformer(
                    _cls_dim,
                    depth=depth,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in n_constituent_features
            ]
        )

        # Final classification: concat of all streams → 2-layer FF → y_dim
        in_dim = (1 + len(n_constituent_features)) * _cls_dim
        self.cls_mlp = MLP_dropout(
            [in_dim, _cls_dim, y_dim], act=nn.ReLU(), dropout=mlp_dropout
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
        # Flatten F_g feature tokens: (B, F_g, dim) → (B, F_g*dim) → (B, cls_dim)
        g = self.global_proj(global_enc.flatten(1))  # (B, cls_dim)

        reprs = [g]

        # ── Constituent streams ───────────────────────────────────────────────
        for i, (enc, valids) in enumerate(zip(constituent_encs, constituent_valids)):
            # Flatten F feature tokens per constituent: (B, C, F, dim) → (B, C, F*dim)
            x = enc.flatten(2)  # (B, C, F*dim)

            # Per-constituent FF projection → (B, C, cls_dim)
            x = self.const_proj[i](x)

            # Zero out padding constituents before cross-constituent attention.
            x = torch.where(valids.unsqueeze(-1), x, 0.0)  # (B, C, cls_dim)

            # Cross-constituent transformer.
            x = self.const_transformer[i](x, valids)  # (B, C, cls_dim)

            # Masked mean pool over valid constituents.
            n_valids = valids.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # (B, 1)
            x = x.sum(dim=1) / n_valids  # (B, cls_dim)

            reprs.append(x)

        # ── Concatenate streams and classify ─────────────────────────────────
        x = torch.cat(reprs, dim=-1)  # (B, (1 + n_const) * cls_dim)
        return self.cls_mlp(x)  # (B, y_dim)
