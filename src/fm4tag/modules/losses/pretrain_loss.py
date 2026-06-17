"""Composable pretraining loss built from a weighted list of loss terms.

``PretrainLoss`` sums ``weight_i * term_i(...)`` over a list of term modules.
Each term is an :class:`~torch.nn.Module` whose ``forward`` returns
``(scalar, log_dict)``.  Terms are dispatched via :func:`_call_term`, which
inspects each term's ``forward`` signature and passes only the keyword
arguments it declares (terms that accept ``**kwargs`` receive everything).

Dispatch is **scope-aware**: a term only fires when every *required* parameter
of its ``forward`` is present in the call.  This lets the pretraining module
hand a single bag of intermediate tensors to one ``PretrainLoss`` at two
different scopes without double-counting:

* a **per-object** call passes ``z_list`` / denoising tensors → only the
  per-object terms (contrastive, denoising) fire, logged as ``<obj>/<key>``;
* a single **jet-level** call passes ``z_jet_list`` → only the jet-embedding
  term fires, logged top-level (e.g. ``jet_embedding/loss_contrastive``).
"""

from __future__ import annotations

import inspect

import torch
from torch import nn

from fm4tag.losses import DenoisingLoss, MultiViewSupConLoss


def _ref_tensor(kwargs: dict) -> torch.Tensor | None:
    """Return any tensor found in ``kwargs`` (scanning into list/tuple values).

    Used only as a device/dtype reference for a zero fallback when no term
    matched a call.
    """
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, torch.Tensor):
                    return item
    return None


def loss_wants(loss: 'PretrainLoss', param_name: str) -> bool:
    """Whether any term in ``loss`` consumes ``param_name``.

    Detection is signature-based (never by class name): a term that accepts
    ``**kwargs`` (``VAR_KEYWORD``) is treated as wanting everything; otherwise
    the term wants ``param_name`` iff it is a named parameter of its ``forward``.
    Used by the modules to skip computing inputs no term needs (denoising
    reconstructions, jet aggregation).
    """
    for term in loss.terms:
        params = inspect.signature(term.forward).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return True
        if param_name in params:
            return True
    return False


def _call_term(
    term: nn.Module, kwargs: dict
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
    """Call ``term`` with only the kwargs its ``forward`` declares — if it applies.

    Terms whose ``forward`` accepts ``**kwargs`` (``VAR_KEYWORD``) receive the
    full ``kwargs`` dict.  Otherwise the term receives only the subset of keys
    matching its named parameters, but **only if every required parameter** (one
    with no default) is present in ``kwargs``.  When a required parameter is
    missing the term does not apply at this scope and ``None`` is returned.
    """
    sig = inspect.signature(term.forward)
    params = sig.parameters

    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return term(**kwargs)

    required = [
        name
        for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    if any(name not in kwargs for name in required):
        return None

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return term(**filtered)


class PretrainLoss(nn.Module):
    """Composable pretraining loss: ``total = Σ weight_i * term_i(...)``.

    Args:
        terms:   List of loss-term modules, each returning ``(scalar, log_dict)``.
        weights: List of float weights, one per term (same length as ``terms``).

    Raises:
        ValueError: If ``terms`` is empty or ``terms`` and ``weights`` differ
            in length.
    """

    def __init__(self, terms: list[nn.Module], weights: list[float]) -> None:
        super().__init__()
        if len(terms) == 0:
            raise ValueError(f'{type(self).__name__} requires at least one term.')
        if len(terms) != len(weights):
            raise ValueError(
                f'terms and weights must have the same length, got '
                f'{len(terms)} terms and {len(weights)} weights.'
            )
        self.terms = nn.ModuleList(terms)
        self.weights = list(weights)

    def forward(self, **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Dispatch to every term via fan-out and combine the results.

        Only terms whose required inputs are present in ``kwargs`` contribute
        (see :func:`_call_term`).  If no term applies, the total is a zero
        scalar on the device of any tensor found in ``kwargs``.

        Returns:
            ``(total_loss, log_dict)``.  ``log_dict`` merges every applied
            term's log dict and always includes a top-level ``'loss'`` key with
            the total.
        """
        total: torch.Tensor | None = None
        log_dict: dict[str, torch.Tensor] = {}

        for term, weight in zip(self.terms, self.weights):
            result = _call_term(term, kwargs)
            if result is None:
                continue
            term_loss, term_log = result
            contrib = weight * term_loss
            total = contrib if total is None else total + contrib
            log_dict.update(term_log)

        if total is None:
            ref = _ref_tensor(kwargs)
            total = ref.new_zeros(()) if ref is not None else torch.zeros(())

        log_dict['loss'] = total
        return total, log_dict


# ---------------------------------------------------------------------------
# Loss-term adapters
# ---------------------------------------------------------------------------


class ContrastiveTermAdapter(nn.Module):
    """Multi-view contrastive term on per-object (POINT A) projections.

    Wraps :class:`~fm4tag.losses.MultiViewSupConLoss`.

    Args:
        temperature:          Contrastive temperature.
        loss_type:            ``'out'`` (L_out) or ``'in'`` (L_in).
        include_pos_in_denom: SupCon denominator includes positives if ``True``.
    """

    def __init__(
        self,
        temperature: float,
        loss_type: str = 'out',
        include_pos_in_denom: bool = True,
    ) -> None:
        super().__init__()
        self.loss = MultiViewSupConLoss(
            temperature=temperature,
            loss_type=loss_type,
            include_pos_in_denom=include_pos_in_denom,
        )

    def forward(
        self, z_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Args:
        z_list: One ``(N, D)`` projection tensor per view.
        """
        loss_val = self.loss(z_list)
        return loss_val, {'loss_contrastive': loss_val}


class JetContrastiveTermAdapter(nn.Module):
    """Multi-view contrastive term on jet-level (POINT B) embeddings.

    Wraps :class:`~fm4tag.losses.MultiViewSupConLoss`; identical to
    :class:`ContrastiveTermAdapter` but consumes the per-view list of ``z_jet``
    tensors instead of per-object projections.

    Args:
        temperature:          Contrastive temperature.
        loss_type:            ``'out'`` (L_out) or ``'in'`` (L_in).
        include_pos_in_denom: SupCon denominator includes positives if ``True``.
    """

    def __init__(
        self,
        temperature: float,
        loss_type: str = 'out',
        include_pos_in_denom: bool = True,
    ) -> None:
        super().__init__()
        self.loss = MultiViewSupConLoss(
            temperature=temperature,
            loss_type=loss_type,
            include_pos_in_denom=include_pos_in_denom,
        )

    def forward(
        self, z_jet_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Args:
            z_jet_list: One ``(B, jet_dim)`` jet embedding per view.

        The log key is namespaced (``jet_embedding/``) so it never collides
        with the per-object ``<obj>/loss_contrastive`` keys (e.g. the global
        object's ``jets/loss_contrastive``).
        """
        loss_val = self.loss(z_jet_list)
        return loss_val, {'jet_embedding/loss_contrastive': loss_val}


class DenoisingTermAdapter(nn.Module):
    """Denoising reconstruction term.

    Wraps :class:`~fm4tag.losses.DenoisingLoss` and returns the weighted sum
    ``weight_cat * loss_cat + weight_con * loss_con``.  The categorical loss is
    only logged when categorical features are present (``cat_outs`` non-empty),
    matching the global object (continuous-only) vs constituent (both) split.

    Args:
        weight_cat: Weight on the categorical reconstruction loss.
        weight_con: Weight on the continuous reconstruction loss.
    """

    def __init__(self, weight_cat: float = 1.0, weight_con: float = 1.0) -> None:
        super().__init__()
        self.weight_cat = weight_cat
        self.weight_con = weight_con
        self.loss = DenoisingLoss()

    def forward(
        self,
        cat_outs: list[torch.Tensor],
        x_categ: torch.Tensor,
        con_outs: list[torch.Tensor],
        x_cont: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Args:
        cat_outs: Per-feature categorical reconstruction logits.
        x_categ:  Original categorical indices.
        con_outs: Per-feature continuous reconstruction predictions.
        x_cont:   Original continuous values.
        """
        l_cat, l_con = self.loss(cat_outs, x_categ, con_outs, x_cont)
        total = self.weight_cat * l_cat + self.weight_con * l_con

        log: dict[str, torch.Tensor] = {}
        if len(cat_outs) > 0:
            log['loss_denoising_cat'] = l_cat
        log['loss_denoising_con'] = l_con
        return total, log
