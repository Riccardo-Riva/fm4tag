"""Legacy composable loss machinery — used only by the legacy
``ContrastiveDenoisingModule``; goes away together with it."""

from .pretrain_loss import (
    ContrastiveTermAdapter,
    DenoisingTermAdapter,
    JetContrastiveTermAdapter,
    PretrainLoss,
    loss_wants,
)

__all__ = [
    'PretrainLoss',
    'ContrastiveTermAdapter',
    'JetContrastiveTermAdapter',
    'DenoisingTermAdapter',
    'loss_wants',
]
