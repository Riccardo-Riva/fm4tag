"""The config layout rules (src/fm4tag/configs/).

* One base config per model (the top-level *.yaml); a base never includes
  another base: its defaults list only selects group options (and _self_).
* A group option never includes anything (no defaults list of its own).
* Every key is written in exactly one file: no option of any group writes a
  key that the base, or an option of another group, also writes — so nothing
  is overwritten except on the command line / in a cells file.
* Every base composes with every arm, callbacks and logger option, and builds.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

from fm4tag.configs import __path__ as cfg_path

CONFIG_DIR = Path(cfg_path[0])
BASES = sorted(p.stem for p in CONFIG_DIR.glob('*.yaml'))
GROUPS = sorted(
    p.name for p in CONFIG_DIR.iterdir() if p.is_dir() and p.name != '__pycache__'
)


def _options(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / group).glob('*.yaml'))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _package(group: str, option: str) -> str:
    """Where an option's keys land: its `# @package` header, else the group."""
    head = (CONFIG_DIR / group / f'{option}.yaml').read_text().splitlines()[0]
    m = re.match(r'#\s*@package\s+(\S+)', head)
    pkg = m.group(1) if m else group
    return '' if pkg == '_global_' else pkg


def _leaves(node, prefix: str = '') -> set[str]:
    """Dotted paths of the values a file writes (lists and scalars are leaves)."""
    if isinstance(node, dict) and node:
        out: set[str] = set()
        for k, v in node.items():
            if k == 'defaults':
                continue
            out |= _leaves(v, f'{prefix}.{k}' if prefix else str(k))
        return out
    return {prefix}


def _option_leaves(group: str, option: str) -> set[str]:
    return _leaves(
        _load(CONFIG_DIR / group / f'{option}.yaml'), _package(group, option)
    )


def _clash(a: set[str], b: set[str]) -> list[tuple[str, str]]:
    """Pairs where one file writes a key the other writes, or a parent of it."""
    return [
        (x, y)
        for x in a
        for y in b
        if x == y or x.startswith(f'{y}.') or y.startswith(f'{x}.')
    ]


def _base_groups(base: str) -> dict[str, str]:
    entries = _load(CONFIG_DIR / f'{base}.yaml')['defaults']
    return {k: v for e in entries if isinstance(e, dict) for k, v in e.items()}


@pytest.mark.parametrize('base', BASES)
def test_a_base_only_selects_group_options(base):
    entries = _load(CONFIG_DIR / f'{base}.yaml')['defaults']
    for e in entries:
        assert e == '_self_' or (
            isinstance(e, dict) and len(e) == 1 and next(iter(e)) in GROUPS
        ), f'{base}.yaml: defaults entry {e!r} is not a group option'
    assert entries[-1] == '_self_'


@pytest.mark.parametrize('group', GROUPS)
def test_a_group_option_includes_nothing(group):
    for option in _options(group):
        assert 'defaults' not in _load(CONFIG_DIR / group / f'{option}.yaml'), (
            f'{group}/{option}.yaml has a defaults list'
        )


@pytest.mark.parametrize('base', BASES)
def test_every_key_is_written_in_one_file(base):
    groups = _base_groups(base)
    base_leaves = _leaves(_load(CONFIG_DIR / f'{base}.yaml'))
    for group in groups:
        for option in _options(group):
            clash = _clash(base_leaves, _option_leaves(group, option))
            assert not clash, (
                f'{base}.yaml and {group}/{option}.yaml both write {clash}'
            )
    for g1, g2 in itertools.combinations(groups, 2):
        for o1, o2 in itertools.product(_options(g1), _options(g2)):
            clash = _clash(_option_leaves(g1, o1), _option_leaves(g2, o2))
            assert not clash, f'{g1}/{o1}.yaml and {g2}/{o2}.yaml both write {clash}'


def test_every_arm_writes_the_same_keys():
    keys = {arm: _option_leaves('arm', arm) for arm in _options('arm')}
    common = set.intersection(*keys.values())
    extra = {arm: k - common for arm, k in keys.items() if k - common}
    # pretraining alone clips its gradients
    assert extra == {'pretrain': {'trainer.gradient_clip_val'}}


def _compose(base: str, *overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=base, overrides=list(overrides))


@pytest.mark.parametrize('base', BASES)
@pytest.mark.parametrize('arm', _options('arm'))
def test_every_base_composes_with_every_arm(base, arm):
    from omegaconf import OmegaConf

    cfg = _compose(base, f'arm={arm}', 'encoder_ckpt=/unused.ckpt')
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert cfg.phase == ('pretrain' if arm == 'pretrain' else 'finetune')


@pytest.mark.parametrize('base', BASES)
@pytest.mark.parametrize('callbacks', _options('callbacks'))
@pytest.mark.parametrize('arm', ['pretrain', 'scratch'])
def test_every_callbacks_option_instantiates(base, callbacks, arm):
    from omegaconf import OmegaConf

    from fm4tag.utils import instantiate_callbacks

    cfg = _compose(base, f'arm={arm}', f'callbacks={callbacks}')
    node = cfg.callbacks
    if cfg.phase == 'finetune' and cfg.get('callbacks_finetune'):
        node = OmegaConf.merge(node, cfg.callbacks_finetune)  # what run.py does
    assert instantiate_callbacks(node)


@pytest.mark.parametrize('base', BASES)
def test_the_lr_schedule_is_written_only_in_optimizer(base):
    import inspect

    from fm4tag.modules import FinetuneModule, PretrainModule

    cfg = _compose(base)
    keys = set(cfg.optimizer)
    for block, module in (('pretrain', PretrainModule), ('finetune', FinetuneModule)):
        both = keys & set(cfg[block])
        assert not both, f'{base}.yaml: {block} also writes {both}'
        # the runner passes every optimizer key to the module
        missing = keys - set(inspect.signature(module).parameters)
        assert not missing, f'{module.__name__} does not take {missing}'


@pytest.mark.parametrize('logger', _options('logger'))
def test_every_logger_option_composes(logger):
    cfg = _compose('default', f'logger={logger}')
    assert 'csv' in cfg.logger


@pytest.mark.parametrize('base', BASES)
def test_every_base_builds_its_model(base):
    from hydra.utils import instantiate as hydra_instantiate

    from fm4tag.utils.model_builders import build_aggregator, build_encoders, build_head

    cfg = _compose(base, 'arm=scratch')
    encoders = build_encoders(cfg)
    aggregator = build_aggregator(cfg, encoders)
    module = hydra_instantiate(
        cfg.finetune,
        encoders=encoders,
        aggregator=aggregator,
        head=build_head(cfg, aggregator),
        views=[hydra_instantiate(v) for v in cfg.views],
        n_classes=len(cfg.variables[cfg.global_object].unique_labels),
        _convert_='all',
    )
    assert sum(p.numel() for p in module.parameters()) > 0
