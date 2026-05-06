"""Encoder representation-quality evaluation.

Loads a pretrained encoder from a checkpoint and evaluates the quality of its
representations on a dataset (typically ``test_file`` or ``pretrain_val_file``),
without any fine-tuning.

Metrics computed
----------------
Per object (global + each constituent type):

* **Uniformity** — how uniformly projected embeddings are spread on the unit
  hypersphere (Wang & Isola, NeurIPS 2020).  More negative = better.
  Computed on the *projection-head* output (``pt_mlp1``) when available,
  otherwise on the flattened encoder output.
* **Effective rank** — exp(H) where H is the entropy of the normalised singular
  value spectrum of the embedding matrix.  Higher = less collapsed.

Jet-level (constituent objects only):

* **Jet-level uniformity** — uniformity computed after masked mean-pooling all
  valid track embeddings per jet.

Usage::

    # Via installed entry-point / Hydra:
    fm4tag-eval --config-name=my_config ckpt_path=outputs/exp/version_0/checkpoints/best.ckpt

    # Via Python module:
    python -m fm4tag.eval_encoder --config-name=my_config \\
        ckpt_path=outputs/exp/version_0/checkpoints/best.ckpt

    # From a notebook (no Hydra):
    from omegaconf import OmegaConf
    from fm4tag.eval_encoder import evaluate
    cfg = OmegaConf.load('src/fm4tag/configs/my_config.yaml')
    results = evaluate(cfg, ckpt_path='outputs/.../best.ckpt')

Results are printed as a table and saved as ``encoder_eval.json`` in the same
directory as the checkpoint.
"""

from __future__ import annotations

import os
from collections import defaultdict

import hydra
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from fm4tag.augmentations import embed_data
from fm4tag.datasets import DatasetCatCon, cat_con_collate_fn
from fm4tag.utils.builders import build_encoders as _build_encoders, load_pretrained_encoders as _load_pretrained_encoders
from fm4tag.metrics import effective_rank, uniformity


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _collect_embeddings(
    z: torch.Tensor,
    store: list[torch.Tensor],
    max_total: int,
) -> None:
    """Append ``z`` to ``store`` until ``max_total`` rows are collected."""
    already = sum(t.size(0) for t in store)
    remaining = max_total - already
    if remaining <= 0:
        return
    store.append(z[:remaining].detach().cpu())


# ---------------------------------------------------------------------------
# Per-object encoding helpers
# ---------------------------------------------------------------------------


def _encode_global(batch: dict, encoder, device: torch.device) -> torch.Tensor:
    """Encode the global object; returns ``(B, proj_dim)`` projected embeddings."""
    x = batch['global'].to(device)
    with torch.no_grad():
        X = encoder(x)  # (B, F_g, dim)
        z = encoder.pt_mlp1(X.flatten(1))  # (B, proj_dim)
    return z


def _encode_constituent(
    batch: dict,
    obj_name: str,
    encoder,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one constituent type.

    Returns:
        z_track: ``(N_valid, proj_dim)`` projected track embeddings.
        z_jet:   ``(B, proj_dim)`` jet-level embeddings (masked mean pool over
                 valid tracks, projected through ``pt_mlp1``).
    """
    const = batch['constituents'][obj_name]
    x_categ = const['categorical'].to(device)   # (B, C, F_cat)
    x_cont = const['continuous'].to(device)     # (B, C, F_con)
    valid = const['valid'].to(device)           # (B, C)

    B, C = valid.shape
    valids_flat = rearrange(valid, 'b c -> (b c)')

    x_categ_flat = rearrange(x_categ, 'b c f -> (b c) f')[valids_flat]
    x_cont_flat = rearrange(x_cont, 'b c f -> (b c) f')[valids_flat]

    with torch.no_grad():
        x_cat_enc, x_con_enc = embed_data(x_categ_flat, x_cont_flat, encoder)
        X = encoder(x_cat_enc, x_con_enc)  # (N_valid, F, dim)
        F_feat, dim = X.shape[1], X.shape[2]

        # Track-level projected embeddings.
        z_track = encoder.pt_mlp1(X.flatten(1))  # (N_valid, proj_dim)

        # Scatter back to (B, C, F, dim) for jet-level pooling.
        buf = torch.zeros(B * C, F_feat, dim, device=device)
        buf[valids_flat] = X
        buf = buf.view(B, C, F_feat, dim)

        # Masked mean pool → (B, F, dim) → project.
        n_valid = valid.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # (B, 1)
        jet_emb = (buf * valid.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (B, F, dim)
        jet_emb = jet_emb / n_valid.unsqueeze(-1)
        z_jet = encoder.pt_mlp1(jet_emb.flatten(1))  # (B, proj_dim)

    return z_track, z_jet


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    cfg: DictConfig,
    *,
    ckpt_path: str | None = None,
    eval_file: str | None = None,
    max_track_samples: int = 50_000,
    max_jet_samples: int = 10_000,
    device: str | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate encoder representation quality.

    Args:
        cfg:              Fully resolved OmegaConf config.
        ckpt_path:        Path to a :class:`PretrainModule` checkpoint.
                          Overrides ``cfg.ckpt_path``.
        eval_file:        HDF5 file to evaluate on.  Defaults to
                          ``cfg.pretrain_val_file`` → ``cfg.test_file``.
        max_track_samples: Maximum number of track embeddings to accumulate
                          for uniformity / effective-rank computation.
        max_jet_samples:  Maximum number of jet embeddings to accumulate.
        device:           ``'cuda'``, ``'cpu'``, etc.  Auto-detected if ``None``.

    Returns:
        Nested dict ``{obj_name: {metric_name: value}}``.
        Constituent objects also have a ``jet_uniformity`` entry.
    """
    _ckpt = ckpt_path or cfg.get('ckpt_path')
    if _ckpt is None:
        raise ValueError('ckpt_path must be provided either as argument or in cfg.')

    _eval_file = (
        eval_file
        or cfg.get('test_file')
        or cfg.get('pretrain_val_file')
    )
    if _eval_file is None:
        raise ValueError(
            'No eval file found. Set pretrain_val_file or test_file in cfg, '
            'or pass eval_file explicitly.'
        )

    _device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ── Build and load encoders ───────────────────────────────────────────────
    encoders = _build_encoders(cfg)
    _load_pretrained_encoders(encoders, _ckpt)
    encoders = encoders.to(_device).eval()

    # ── Dataset / dataloader ─────────────────────────────────────────────────
    dl_cfg = cfg.get('dataloader') or {}
    dataset = DatasetCatCon(
        file_path=_eval_file,
        variables=cfg.variables,
        global_object=cfg.global_object,
        constituent_objects=list(cfg.constituent_objects),
        norm_dict=OmegaConf.to_container(OmegaConf.load(cfg.norm_dict), resolve=True)
        if cfg.get('norm_dict')
        else None,
        class_dict=OmegaConf.to_container(OmegaConf.load(cfg.class_dict), resolve=True)
        if cfg.get('class_dict')
        else None,
    )
    num_workers = dl_cfg.get('num_workers', 4)
    loader = DataLoader(
        dataset,
        batch_size=dl_cfg.get('batch_size', 1024),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cat_con_collate_fn,
        prefetch_factor=dl_cfg.get('prefetch_factor', 2) if num_workers > 0 else None,
        pin_memory=dl_cfg.get('pin_memory', True),
    )

    # ── Accumulate embeddings ─────────────────────────────────────────────────
    # Keys: obj_name → list of (N, D) tensors
    track_store: dict[str, list[torch.Tensor]] = defaultdict(list)
    jet_store: dict[str, list[torch.Tensor]] = defaultdict(list)

    for batch in loader:
        # Global.
        global_name = cfg.global_object
        enc_global = encoders[global_name]
        if hasattr(enc_global, 'pt_mlp1'):
            z_global = _encode_global(batch, enc_global, _device)
            _collect_embeddings(z_global, track_store[global_name], max_jet_samples)

        # Constituents.
        for obj_name in cfg.constituent_objects:
            enc = encoders[obj_name]
            if not hasattr(enc, 'pt_mlp1'):
                continue
            z_track, z_jet = _encode_constituent(batch, obj_name, enc, _device)
            _collect_embeddings(z_track, track_store[obj_name], max_track_samples)
            _collect_embeddings(z_jet, jet_store[obj_name], max_jet_samples)

        # Stop early once we have enough samples everywhere.
        enough_tracks = all(
            sum(t.size(0) for t in v) >= max_track_samples
            for v in track_store.values()
        )
        enough_jets = all(
            sum(t.size(0) for t in v) >= max_jet_samples
            for v in jet_store.values()
        )
        if enough_tracks and enough_jets:
            break

    # ── Compute metrics ───────────────────────────────────────────────────────
    results: dict[str, dict[str, float]] = {}

    # Global object.
    global_name = cfg.global_object
    if track_store[global_name]:
        z = torch.cat(track_store[global_name], dim=0)
        results[global_name] = {
            'uniformity': float(uniformity(z).item()),
            'effective_rank': effective_rank(z),
            'n_samples': z.size(0),
        }

    # Constituent objects.
    for obj_name in cfg.constituent_objects:
        entry: dict[str, float] = {}

        if track_store[obj_name]:
            z_t = torch.cat(track_store[obj_name], dim=0)
            entry['track_uniformity'] = float(uniformity(z_t).item())
            entry['track_effective_rank'] = effective_rank(z_t)
            entry['n_tracks'] = z_t.size(0)

        if jet_store[obj_name]:
            z_j = torch.cat(jet_store[obj_name], dim=0)
            entry['jet_uniformity'] = float(uniformity(z_j).item())
            entry['jet_effective_rank'] = effective_rank(z_j)
            entry['n_jets'] = z_j.size(0)

        if entry:
            results[obj_name] = entry

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_results(results: dict[str, dict[str, float]]) -> None:
    col_w = 26
    val_w = 12

    sep = '─' * (col_w + val_w * 3)
    print(f'\n{sep}')
    print('Encoder evaluation results')
    print(sep)

    for obj_name, metrics in results.items():
        print(f'\n  {obj_name}')
        for k, v in metrics.items():
            if k.startswith('n_'):
                print(f'    {k:<{col_w - 4}}{int(v):>{val_w}d}')
            else:
                print(f'    {k:<{col_w - 4}}{v:>{val_w}.4f}')

    print(f'\n{sep}\n')


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='../../configs', config_name='default')
def main(cfg: DictConfig) -> None:
    _ckpt = cfg.get('ckpt_path')
    results = evaluate(cfg, ckpt_path=_ckpt)
    _print_results(results)

    # Save YAML alongside the checkpoint.
    if _ckpt:
        out_path = os.path.join(os.path.dirname(_ckpt), 'encoder_eval.yaml')
        OmegaConf.save(OmegaConf.create(results), out_path)
        print(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
