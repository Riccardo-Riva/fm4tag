# working rules

## Rules
- **Never commit, push unless the user explicitly asks.** The user commits. You can eventually suggest to create branches or worktrees.
- One small step at a time (diff under ~300 lines), with tests, then stop for the user's review.
- **Everything we do is recorded in `docs/DIARY.md`.** One dated entry per step, appended at the
  end (newest last): what changed, the decisions taken and why, the numbers measured, and what is
  left open. Update it in the same step as the code, and say in the report that it was updated.
- Use Google format for docstrings:
    """
    This is an example of Google style.

    Args:
        param1: This is the first param.
        param2: This is a second param.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.
    """ 
- Do not leave in the comments of the codebase, specific references to the repository we are
  using as references. Neither to the actual specifics of the hardware on which we are
  developing the repo. 


## Simplicity rules
- No base class until two implementations need it.
- Delete dead code instead of keeping "superseded" modules.
- Comments are short and explain *why*; benchmark numbers go in the diary, not in code or YAML.
- Split a file when it grows past ~300 lines.
- Every feature ships with a test on toy data.
- Speed decisions are taken on measured numbers (seconds, memory), recorded in the diary.

## Commands
```
uv sync                                          # environment
uv run pytest                                    # tests (CPU, plus GPU if available)
uv run ruff check . && uv run ruff format .      # lint + format (notebooks included)
uv run jupyter lab                               # notebooks, kernel = project .venv
```