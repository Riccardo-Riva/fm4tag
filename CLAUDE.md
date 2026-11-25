# fm4tag — working rules

Framework to test Contrastive Loss and Transformer-based encoders, together with transfer learning and multi-task training, in Jet Flavour Tagging.
Built step by step together with the user, who wants to understand and review every piece.
`DIARY.md` holds the plan, the decision log and the progress; read it at the start of every session.

## Workflow
- **Never commit, push or create branches unless the user explicitly asks.** The user commits.
- One small step at a time (diff under ~300 lines), with tests, then stop for the user's review.
- Explore, then plan, then code: read the relevant code first, propose the approach, and wait for the go-on before editing anything non-trivial.
- Ask instead of guessing when a requirement is ambiguous; state any assumption explicitly.
- Before implementing a component, give a short side-by-side (reference repos / current best
  practice) and a proposal; the user decides. Record the choice in the decision log.
- Verify with evidence: run the tests or commands and report what was run and the actual output. Never say something works without having run it.
- At each stop, summarise: what changed and why, where to look first, assumptions made, and what is not tested.
- At the end of every session, add a dated entry to `DIARY.md`: Done / Decisions / Numbers /
  Open questions / Next.
- Reference repos are read-only: `<path/to/reference-repo>`.

## Ask first
- Adding or upgrading a dependency.
- Anything long, expensive or irreversible: full training runs, large downloads, deleting data, checkpoints or run outputs.
- Changing a metric, data split, normalisation or label definition: these silently change every result.
- Never modify raw data and never write outside the repo.

## Code conventions
- Google-style docstrings (`Args`, `Returns`, `Raises`) and type hints on public functions.
- Comments are short and explain *why*. Do not reference the reference repos, and do not mention the specifics of the hardware we develop on, anywhere in the codebase.
- Do not restate in this file what the linter or formatter already enforces.

## Simplicity rules
- The config system's `_target_` (or equivalent) is the only factory: no registries, no type-string switches.
- No base class until two implementations need it.
- Dimensions (input sizes, number of classes) are derived from the config in one place; no hidden config injection.
- Delete dead code instead of keeping "superseded" modules.
- Benchmark numbers go in the diary, not in code or YAML.
- Split a file when it grows past ~300 lines.
- Speed decisions are taken on measured numbers (samples/s, memory), recorded in the diary.

## Testing
- Every feature ships with a test on few samples (if data is already provided) that runs on CPU.
- For models and training code, cover at least: output shapes, determinism under a fixed seed, that a tiny batch can be overfitted, no NaN/inf in loss or gradients, and a checkpoint
save/load round trip.

## ML correctness
- Seed everything in one place (all RNGs and dataloader workers). Every run logs its resolved
  config, seed and git commit.
- Comparisons that drive a decision use several seeds (at least 3, ideally 5) and report mean ± std, not the best run.
- No leakage: fit normalisation statistics on train only; split by the unit that must not be shared (group, subject, event), not by sample; never tune on the test set.
- Sanity checks before scaling up: assert shapes, dtypes and devices at module boundaries; the initial loss matches the expected value (e.g. -log(1/C) for C classes); handle NaN/inf explicitly, never silently.
- A number in the diary names the run (config + commit) that produced it.

## Commands
```
uv sync                                          # environment
uv run pytest                                    # tests (CPU, toy data)
uv run ruff check . && uv run ruff format .      # lint + format
<command to generate toy data>
<command to launch a training run>
```

## Environment
- Dev machine: GPU with available gpu queues.
- Training runs on cluster; code reaches it through git.
  such as compression or non-finite values>.
- Machine-specific details (paths, partitions, credentials) go in `CLAUDE.local.md`, not here.
