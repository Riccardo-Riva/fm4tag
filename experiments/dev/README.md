# Development tests

Short CPU or GPU tests that inform a decision (memory, speed, a sanity check) but are not experiments. Each test has a
folder `<yyyy-mm-dd>_<name>/` with its script, its slurm job and a README (question, how, result). The jobs write their
logs and results to the folder's `outputs/` (not in git). Results that drive a decision also go to `DIARY.md`.

| Test | Question | Status |
|---|---|---|
| `2026-11-26_cpu-mem-estimate` | GPU memory of the small_RC / small_C / normal_C pretrainings, before choosing how many share a card | done (job 7477983); **estimate 1.5–1.9× too low on GPU, not used** (README) |
