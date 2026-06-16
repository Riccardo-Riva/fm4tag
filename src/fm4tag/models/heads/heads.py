import torch
from torch import nn

from ..mlp import MLP_dropout


class MultiStreamClassifierHead(nn.Module):
    """Classification head for jet flavour tagging.

    Receives the **already-aggregated** jet embedding ``z_jet`` (POINT B,
    produced by :class:`~fm4tag.models.aggregator.JetAggregator`) and produces
    class logits via a 2-layer feed-forward MLP:

        z_jet  (B, jet_dim) → MLP → logits (B, y_dim)

    Cross-constituent attention and masked pooling now live in the shared
    :class:`~fm4tag.models.aggregator.JetAggregator`; this head only owns the
    final classification MLP.

    The expected calling convention (in
    :class:`~fm4tag.modules.finetune_module.FinetuneModule`) is::

        z_global, z_consts, valids = self._encode_all(batch)  # projector output
        z_jet  = self.aggregator(z_global, z_consts, valids)  # POINT B
        logits = self.head(z_jet)                              # POINT C

    Args:
        jet_dim:     Dimension of the incoming jet embedding
                     (``aggregator.out_dim``).
        y_dim:       Number of output classes.
        mlp_hidden:  Hidden width of the classification MLP.  ``None`` → auto
                     (``max(jet_dim // 2, y_dim)``).
        mlp_dropout: Dropout inside the final classification MLP.
    """

    def __init__(
        self,
        *,
        jet_dim: int,
        y_dim: int = 2,
        mlp_hidden: int | None = None,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        _hidden = mlp_hidden if mlp_hidden is not None else max(jet_dim // 2, y_dim)
        self.cls_mlp = MLP_dropout(
            [jet_dim, _hidden, y_dim], act=nn.ReLU(), dropout=mlp_dropout
        )

    def forward(self, z_jet: torch.Tensor) -> torch.Tensor:
        """Classify a batch of jets from their aggregated embeddings.

        Args:
            z_jet: ``(B, jet_dim)`` — aggregated jet embedding from
                   :class:`~fm4tag.models.aggregator.JetAggregator`.

        Returns:
            ``(B, y_dim)`` class logits (unnormalised).
        """
        return self.cls_mlp(z_jet)
