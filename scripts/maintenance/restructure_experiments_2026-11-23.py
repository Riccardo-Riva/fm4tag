"""One-off restructuring of 2026-11-23: every experiment gets its own folder experiments/<campaign>/.

    experiments/<campaign>/{README.md, slurm/, configs/, outputs/, plots/, notes/, diagnostics/, slides/}

Tracked files (cells, plot configs, campaign launchers, notes) are moved with ``git mv``; git-ignored results (run dirs,
plots, logs) with a plain rename (same filesystem).  Then every path that points at a moved file or folder is rewritten
in the tracked text files (absolute /storage3/... paths, ${REPO}/... paths and repo-relative ones).  Old run records
(.hydra/config.yaml, caches) are not edited: slurm/ssl_campaign/<stage> becomes a symlink to the stage's new folder,
so the encoder_ckpt paths recorded in old runs keep resolving.

    python scripts/maintenance/restructure_experiments_2026-11-23.py            # dry run: print the plan
    python scripts/maintenance/restructure_experiments_2026-11-23.py --apply
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

REPO = Path('/storage3/DSIP/rriva/research/fm4tag')
ABS = '/storage3/DSIP/rriva/research/fm4tag'

EARLY = '2026-03_early-runs'
GRIDS = '2026-07_ce-jc-grids'
PROF = '2026-07_kernel-profiling'
LAMBDA = '2026-08_lambda-scan'
SSL = '2026-10_ssl-campaign'
DROP = '2026-11_dropout'
WSD = '2026-11_wsd-lr-search'
FOUR = '2026-11_four-way-jc-dropout'
CAMPAIGNS = [EARLY, GRIDS, PROF, LAMBDA, SSL, DROP, WSD, FOUR]


def stage_campaign(name: str) -> str | None:
    """Campaign of a stage dir / cells file / log in slurm/ssl_campaign, by its name prefix."""
    if re.match(
        r'(smoke_|S1_|S2_|S3_|P_full|F_final|C_gap|F_scjc|F_p1_nojc|L_lr|L_reg|L_scan)',
        name,
    ):
        return SSL
    if re.match(r'(D_drop|P_drop)', name):
        return DROP
    if re.match(r'W_', name):
        return WSD
    if re.match(r'T4way', name):
        return FOUR
    return None


def yaml_campaign(name: str) -> str:
    stem = name.removesuffix('.yaml')
    if stem.startswith('dropout_'):
        return DROP
    if stem == 'wsd_lr_search':
        return WSD
    if stem in {
        'arm_epochs',
        'arm_epochs_p1',
        'arm_epochs_scjc',
        'arm_summary',
        'arm_test_metrics',
        'fraction_constrained_arms_final',
        'fraction_constrained_arms_final_p1',
        'method_scan_labels',
        'method_scan_lr',
        'method_scan_reg',
        'p0_vs_p1_rejections',
        'p0_vs_p1_summary',
        'pretrain_curves',
        'pretrain_curves_final',
        'roc_comparison_arms_final',
        'roc_comparison_arms_final_p1',
        'test_rejections',
    }:
        return SSL
    return LAMBDA  # every other plot / ROC / fraction config belongs to the lambda-scan ("beat GN2") analysis


PLOT_DIRS = {
    'arm_comparison': SSL,
    'arm_comparison_p1': SSL,
    'p0_vs_p1': SSL,
    'label_scan': SSL,
    'pretrain_curves': SSL,
    'dropout': DROP,
    'wsd_lr_search': WSD,
    'val_loss_ce': LAMBDA,
    'losses': LAMBDA,
    'uniformity': LAMBDA,
    'ce_vs_embedding': LAMBDA,
    'ce_vs_embedding_for_scp': LAMBDA,
}
SMOKE_TOOLS = {'cpu_pytest.sh', 'gpu_pytest.sh', 'gpu_health.sh'}  # generic -> slurm/
LAUNCHERS = {  # tracked campaign-specific launchers: old path -> new path
    'slurm/pretrain.sh': f'experiments/{EARLY}/slurm/pretrain.sh',
    'slurm/finetune.sh': f'experiments/{EARLY}/slurm/finetune.sh',
    'slurm/jet_only/classify_from_scratch.sh': f'experiments/{EARLY}/slurm/jet_only/classify_from_scratch.sh',
    'slurm/jet_only/finetune.sh': f'experiments/{EARLY}/slurm/jet_only/finetune.sh',
    'slurm/jet_only/pretrain.sh': f'experiments/{EARLY}/slurm/jet_only/pretrain.sh',
    'slurm/classify_from_scratch.sh': f'experiments/{GRIDS}/slurm/classify_from_scratch.sh',
    'slurm/classify_param_scan.sh': f'experiments/{LAMBDA}/slurm/classify_param_scan.sh',
    'slurm/ssl_campaign.sh': f'experiments/{SSL}/slurm/ssl_campaign.sh',
    'slurm/launch_drop_stage.sh': f'experiments/{DROP}/slurm/launch_drop_stage.sh',
    'slurm/launch_pretrain_drop_stage.sh': f'experiments/{DROP}/slurm/launch_pretrain_drop_stage.sh',
    'slurm/share_measure.sh': f'experiments/{WSD}/slurm/share_measure.sh',
}
DOCS = {
    'docs/ssl_campaign.md': f'experiments/{SSL}/notes/ssl_campaign.md',
    'docs/ssl_campaign_report.md': f'experiments/{SSL}/notes/ssl_campaign_report.md',
    'docs/ssl_tests_plan.pptx': f'experiments/{SSL}/notes/ssl_tests_plan.pptx',
    'docs/ssl_tests_plan_preview.jpg': f'experiments/{SSL}/notes/ssl_tests_plan_preview.jpg',
    'docs/experiments/README.md': 'experiments/README.md',
    'plot_paths.txt': f'experiments/{LAMBDA}/plot_paths.txt',
}


def plan() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(git moves of tracked files, plain moves of untracked/ignored content), as repo-relative paths."""
    git_mv, mv = [], []
    tracked = set(
        subprocess.run(
            ['git', 'ls-files'], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.split()
    )
    for cells in sorted((REPO / 'scripts/configs/campaign').glob('*.cells')):
        git_mv.append(
            (
                str(cells.relative_to(REPO)),
                f'experiments/{stage_campaign(cells.stem)}/configs/{cells.name}',
            )
        )
    for y in sorted(
        p
        for p in (REPO / 'scripts/configs').glob('*.yaml')
        if str(p.relative_to(REPO)) in tracked
    ):
        git_mv.append(
            (
                str(y.relative_to(REPO)),
                f'experiments/{yaml_campaign(y.name)}/configs/{y.name}',
            )
        )
    for y in sorted((REPO / 'scripts/configs/slides').glob('*.yaml')):
        git_mv.append(
            (str(y.relative_to(REPO)), f'experiments/{SSL}/configs/slides/{y.name}')
        )
    git_mv += list(LAUNCHERS.items()) + list(DOCS.items())

    for st in sorted((REPO / 'slurm/ssl_campaign').iterdir()):
        camp = stage_campaign(st.name)
        if camp is None:
            raise SystemExit(f'no campaign for {st}')
        mv.append(
            (f'slurm/ssl_campaign/{st.name}', f'experiments/{camp}/outputs/{st.name}')
        )
    for d, camp in PLOT_DIRS.items():
        mv.append((f'plots/{d}', f'experiments/{camp}/plots/{d}'))
    for d in sorted((REPO / 'scripts/plots').iterdir()):
        mv.append((f'scripts/plots/{d.name}', f'experiments/{LAMBDA}/plots/{d.name}'))
    for t in sorted((REPO / 'scripts/configs').glob('_tmp_*')):
        mv.append(
            (f'scripts/configs/{t.name}', f'experiments/{LAMBDA}/configs/{t.name}')
        )
    mv += [
        ('scripts/logs', f'experiments/{LAMBDA}/outputs/script_logs'),
        (
            'slurm/classify_param_scan',
            f'experiments/{LAMBDA}/outputs/classify_param_scan',
        ),
        (
            'slurm/classify_from_scratch',
            f'experiments/{GRIDS}/outputs/classify_from_scratch',
        ),
        ('slurm/old_tests', f'experiments/{EARLY}/outputs/old_tests'),
        ('outputs', f'experiments/{EARLY}/outputs/hydra_outputs'),
        ('slurm/plot_logs', f'experiments/{SSL}/outputs/plot_logs'),
        ('docs/slides', f'experiments/{SSL}/slides'),
        ('slurm/finetuning', f'experiments/{EARLY}/outputs/finetuning'),
    ]
    for p in sorted((REPO / 'slurm/profiling').iterdir()):
        mv.append((f'slurm/profiling/{p.name}', f'experiments/{PROF}/outputs/{p.name}'))
    for f in sorted((REPO / 'slurm/arm_roc').iterdir()):
        target = (
            f'slurm/{f.name}'
            if f.suffix == '.sh'
            else f'experiments/{SSL}/outputs/arm_roc_logs/{f.name}'
        )
        mv.append((f'slurm/arm_roc/{f.name}', target))
    for f in sorted((REPO / 'slurm/smoke').iterdir()):
        if f.name in SMOKE_TOOLS:
            mv.append((f'slurm/smoke/{f.name}', f'slurm/{f.name}'))
        elif f.suffix in {'.py', '.sh'}:
            mv.append(
                (f'slurm/smoke/{f.name}', f'experiments/{SSL}/diagnostics/{f.name}')
            )
        else:  # outputs and logs of the diagnostics
            mv.append(
                (
                    f'slurm/smoke/{f.name}',
                    f'experiments/{SSL}/diagnostics/outputs/{f.name}',
                )
            )
    return git_mv, mv


PREFIX = (
    r'(?P<pre>'
    + re.escape(ABS)
    + r'(?:-chunk(?:-s0)?)?/|\$\{REPO\}/|\$REPO/|(?<![\w/.$\-]))'
)


def rewrite(text: str, git_mv, mv) -> str:
    def stage(m):
        camp = stage_campaign(m['name'])
        return f'{m["pre"]}experiments/{camp}/outputs/{m["name"]}' if camp else m[0]

    text = re.sub(
        PREFIX + r'slurm/ssl_campaign/(?P<name>[A-Za-z][A-Za-z0-9_.%]*)', stage, text
    )
    # moved files and folders, longest source first so a folder never shadows a file inside it
    pairs = sorted(
        [
            (s, d)
            for s, d in git_mv + mv
            if not s.startswith('slurm/ssl_campaign/') and s != 'outputs'
        ],
        key=lambda p: -len(p[0]),
    )
    pairs += [
        ('slurm/pretraining', f'experiments/{EARLY}/outputs/pretraining'),
        ('slurm/finetuning', f'experiments/{EARLY}/outputs/finetuning'),
        (
            'slurm/jet_only/classify_from_scratch',
            f'experiments/{EARLY}/outputs/jet_only/classify_from_scratch',
        ),
        (
            'slurm/jet_only/finetuning',
            f'experiments/{EARLY}/outputs/jet_only/finetuning',
        ),
        (
            'slurm/jet_only/pretraining',
            f'experiments/{EARLY}/outputs/jet_only/pretraining',
        ),
        ('slurm/arm_roc/', f'experiments/{SSL}/outputs/arm_roc_logs/'),
        ('slurm/smoke/', f'experiments/{SSL}/diagnostics/'),
        ('scripts/plots/', f'experiments/{LAMBDA}/plots/'),
    ]
    for src, dst in pairs:
        text = re.sub(
            PREFIX + re.escape(src) + r'(?=$|[/\s"\'`,:;)\]}|])',
            lambda m, d=dst: m['pre'] + d,
            text,
            flags=re.M,
        )
    # the old root outputs/ (hydra outputs of the early runs): absolute paths only, 'outputs/' is a common word
    text = text.replace(
        f'{ABS}/outputs/', f'{ABS}/experiments/{EARLY}/outputs/hydra_outputs/'
    )
    return text.replace(f'{ABS}-chunk/', f'{ABS}/').replace(
        f'{ABS}-chunk-s0/', f'{ABS}/'
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    os.chdir(REPO)
    git_mv, mv = plan()
    for src, dst in git_mv + mv:
        if not (REPO / src).exists() and not (REPO / src).is_symlink():
            raise SystemExit(f'missing source {src}')
        if (REPO / dst).exists():
            raise SystemExit(f'target exists {dst}')
    print(f'{len(git_mv)} tracked moves, {len(mv)} untracked/ignored moves')
    for src, dst in git_mv + mv:
        print(f'  {src}  ->  {dst}')
    if not args.apply:
        return
    for src, dst in git_mv:
        (REPO / dst).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'mv', src, dst], check=True)
    for src, dst in mv:
        (REPO / dst).parent.mkdir(parents=True, exist_ok=True)
        os.rename(REPO / src, REPO / dst)
    # newly tracked: the generic job scripts and the campaign diagnostics' code
    new = [d for s, d in mv if d.endswith(('.sh', '.py')) and '/outputs/' not in d]
    subprocess.run(['git', 'add', '-f', *new], check=True)
    # old run records point at slurm/ssl_campaign/<stage>: keep those paths alive as symlinks
    legacy = REPO / 'slurm/ssl_campaign'
    legacy.mkdir(exist_ok=True)
    for src, dst in mv:
        if src.startswith('slurm/ssl_campaign/'):
            os.symlink(os.path.relpath(REPO / dst, legacy), REPO / src)
    # absolute symlinks inside moved trees (plot caches)
    for _, dst in mv:
        for root, dirs, files in os.walk(REPO / dst):
            for n in dirs + files:
                p = Path(root) / n
                if p.is_symlink() and os.readlink(p).startswith(ABS + '/'):
                    rel = os.readlink(p)[len(ABS) + 1 :]
                    new_target = rewrite(rel, git_mv, mv)
                    if new_target != rel:
                        p.unlink()
                        p.symlink_to(f'{ABS}/{new_target}')
            if dst.endswith('outputs') or '/outputs/' in dst:
                dirs[:] = []  # do not descend into run folders (101 GB)
    # rewrite paths in every tracked text file except the historical inspection report and this script
    files = subprocess.run(
        ['git', 'ls-files'], check=True, capture_output=True, text=True
    ).stdout.split()
    skip = {
        'docs/repo_inspection_2026-11-23.md',
        'scripts/maintenance/restructure_experiments_2026-11-23.py',
    }
    changed = 0
    for f in files:
        p = REPO / f
        if (
            f in skip
            or p.is_symlink()
            or not p.is_file()
            or p.suffix in {'.pptx', '.jpg', '.png', '.pdf'}
        ):
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        new_text = rewrite(text, git_mv, mv)
        if new_text != text:
            p.write_text(new_text)
            changed += 1
    print(f'rewrote paths in {changed} tracked files')


if __name__ == '__main__':
    main()
