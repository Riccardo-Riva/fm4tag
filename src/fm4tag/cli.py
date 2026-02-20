"""Typer-based CLI for fm4tag.

Entry point: ``fm4tag`` (defined in ``pyproject.toml``).

Commands
--------
fm4tag pretrain-encoder  -c <config>
fm4tag train-classifier  -c <config> [--encoder-ckpt <path>] [--ckpt-path <path>]
fm4tag test-classifier   -c <config> [--ckpt-path <path>]
fm4tag predict-classifier -c <config> [--ckpt-path <path>]

Run ``fm4tag --help`` or ``fm4tag <command> --help`` for details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from omegaconf import OmegaConf

app = typer.Typer(
    name='fm4tag',
    help='Foundation Model for Jet Flavour Tagging — training CLI.',
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Shared option definitions (reused across commands for consistency)
# ---------------------------------------------------------------------------

_CONFIG_OPTION = typer.Option(
    ...,
    '-c',
    '--config',
    help='Path to the YAML configuration file.',
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
)

_CKPT_OPTION = typer.Option(
    None,
    '--ckpt-path',
    help='Lightning checkpoint — resume training (fit) or evaluate (test/predict).',
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def pretrain_encoder(
    config: Path = _CONFIG_OPTION,
    ckpt_path: Optional[Path] = _CKPT_OPTION,
) -> None:
    """Self-supervised pretraining of the SAINT encoder.

    Runs contrastive + denoising pretraining on the constituent-level data
    (tracks) and saves the encoder weights as a Lightning checkpoint.
    """
    from fm4tag.train import run

    cfg = OmegaConf.load(config)
    OmegaConf.resolve(cfg)

    run(
        cfg,
        phase='pretrain',
        action='fit',
        ckpt_path=str(ckpt_path) if ckpt_path else None,
    )


@app.command()
def train_classifier(
    config: Path = _CONFIG_OPTION,
    encoder_ckpt: Optional[Path] = typer.Option(
        None,
        '--encoder-ckpt',
        help='PretrainModule checkpoint to initialise the encoder backbone.',
    ),
    ckpt_path: Optional[Path] = _CKPT_OPTION,
) -> None:
    """Supervised fine-tuning of the classifier.

    Loads a pretrained encoder (optional) and trains the full
    encoder + ClassifierHead model on labelled jet data.

    With ``freeze_encoder: true`` in the config the encoder starts frozen;
    the BackboneFinetuning callback progressively unfreezes it.
    """
    from fm4tag.train import run

    cfg = OmegaConf.load(config)
    OmegaConf.resolve(cfg)

    run(
        cfg,
        phase='finetune',
        action='fit',
        encoder_ckpt=str(encoder_ckpt) if encoder_ckpt else None,
        ckpt_path=str(ckpt_path) if ckpt_path else None,
    )


@app.command()
def test_classifier(
    config: Path = _CONFIG_OPTION,
    ckpt_path: Optional[Path] = _CKPT_OPTION,
) -> None:
    """Evaluate a trained classifier on the test set.

    Logs test loss and accuracy to the CSV logger.
    Defaults to the best checkpoint saved by ModelCheckpoint when
    ``--ckpt-path`` is not given.
    """
    from fm4tag.train import run

    cfg = OmegaConf.load(config)
    OmegaConf.resolve(cfg)

    run(
        cfg,
        phase='finetune',
        action='test',
        ckpt_path=str(ckpt_path) if ckpt_path else None,
    )


@app.command()
def predict_classifier(
    config: Path = _CONFIG_OPTION,
    ckpt_path: Optional[Path] = _CKPT_OPTION,
) -> None:
    """Run inference and save class probabilities.

    Outputs a ``predictions.pt`` file (list of softmax tensors, one per batch)
    in the logger's output directory.
    Defaults to the best checkpoint when ``--ckpt-path`` is not given.
    """
    from fm4tag.train import run

    cfg = OmegaConf.load(config)
    OmegaConf.resolve(cfg)

    run(
        cfg,
        phase='finetune',
        action='predict',
        ckpt_path=str(ckpt_path) if ckpt_path else None,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == '__main__':
    main()
