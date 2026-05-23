# Braindecode roadmap

This document tracks medium-term engineering work that is too big for a
single PR but is too concrete to live as a vague "ideas" issue. Each
item links back to the discussion / audit that motivated it where
applicable.

The roadmap is not a release plan — releases are driven by `whats_new`
entries — and items can be picked up by anyone. Maintainers will revise
the priorities once per release cycle.

## Status legend

* :white_check_mark: shipped (linked PR / commit)
* :hammer: in progress (open PR or active branch)
* :calendar: queued (no work started yet)

---

## P0 — Project hygiene (largely shipped)

These were the small, high-leverage cleanups identified in the May 2026
quality pass. Most are merged or have a PR open.

| Item | Status | Notes |
|------|--------|-------|
| Fix Codecov upload guard (it was filtering on a removed Python version) | :white_check_mark: | `[CI] Fix Codecov upload guard pinned to a removed Python version` |
| PyPI classifier `Alpha` → `Production/Stable` + license/OS/topic classifiers | :white_check_mark: | `[MNT] PyPI metadata: stable status, ...` |
| Ship a PEP 561 `py.typed` marker so downstream type checkers see our annotations | :white_check_mark: | same commit as classifier bump |
| Add an advisory `mypy` CI job + baseline `[tool.mypy]` config | :white_check_mark: | `.github/workflows/type-check.yml` |
| Extend ruff selection to `E,W,F` for `braindecode/` (was: only `NPY201` + unused imports) | :white_check_mark: | `[CI] Extend ruff to lint braindecode/...` |
| Bump CircleCI image off EOL Python 3.9 | :white_check_mark: | `cimg/python:3.11` |
| Document checkpoint-loading safely (`weights_only=True`) | :white_check_mark: | `meta_neuromotor` example |
| `SECURITY.md` with reporting policy and known considerations | :white_check_mark: | top-level `SECURITY.md` |
| Promote input-validation `assert`s to `TypeError`/`ValueError` in `windowers.py` | :white_check_mark: | survives `python -O` |
| Dependabot config for GitHub Actions + pinned dev tooling | :white_check_mark: | `.github/dependabot.yml` |
| Bump `actions/cache@v3` → `@v4`, `setup-python@v4` → `@v5` | :white_check_mark: | `.github/workflows/tests.yml` |

---

## P1 — Reliability, performance and safety

| Item | Status | Notes |
|------|--------|-------|
| Switch to `joblib.dump` / `joblib.load` for the dataset cache, gated behind an opt-in flag with a `FutureWarning` on the legacy path | :hammer: | see `SECURITY.md` §1 |
| Run pytest in parallel under `pytest-xdist -n auto` in CI | :calendar: | wallclock dominates CI cost |
| Move `linear_attention_transformer` from core to an optional extra and lazy-import inside the few models that need it | :calendar: | shrinks default install footprint |
| Lazy import the model zoo (`braindecode.models` currently eagerly imports 56 files at first `import braindecode`) | :calendar: | use module-level `__getattr__` per PEP 562 |
| Smoke-test one end-to-end example (e.g. `plot_basic_training_epochs.py`) on synthetic data in CI | :calendar: | catches example-only regressions |
| Convert the rest of the high-traffic `assert`s in `augmentation/transforms.py`, `models/signal_jepa.py`, `preprocessing/windowers.py`, `datasets/bbci.py` to typed exceptions | :hammer: | windowers done; ~100 left |
| ONNX / TorchScript export round-trip test for the most-used models (`EEGNet`, `ShallowFBCSP`, `Deep4`, `EEGConformer`, `Labram`) | :calendar: | enables production deployment |

---

## P2 — Type coverage and API documentation

| Item | Status | Notes |
|------|--------|-------|
| Lift `mypy` from advisory to required on a curated allow-list of typed modules, ratcheting that list over time | :calendar: | start with `version`, `util`, `models.base`, `eegneuralnet`, `classifier`, `regressor` |
| Reach 80 % type-annotated functions (currently ~33 %) | :calendar: | tracked by the helper script in `scripts/` |
| Auto-generated model overview table in the docs (parameters, FLOPs, recommended input shape, paper reference) | :calendar: | reuse `model_card_template` TODO in `models/base.py:243` |
| Document the `ChannelTypes` story — which models support ECoG / MEG and which are EEG-only | :calendar: | content for `docs/models/` |
| Numpydoc validation as a required pre-commit hook (currently behind `stages: [manual]`) | :calendar: | requires a docstring cleanup pass first |

---

## P3 — Architecture and research

| Item | Status | Notes |
|------|--------|-------|
| Refactor: extract common attention blocks (currently duplicated across `EEGConformer`, `Labram`, `BIOT`, `EEG-PT`) into `modules/attention.py` | :calendar: | reduces parameter drift between architectures |
| Unify the `ChannelTypes` handling so every model declares which physiological modalities it supports | :calendar: | follow-up to the docs task above |
| Benchmark suite: per-model accuracy + throughput on BCI IV 2a / SleepEDF / TUH, published as a table in `docs/` and updated on each release | :calendar: | currently scattered across individual papers |
| Model card dataclass + autogeneration into the rendered docs | :calendar: | unblocks the `model_card_template` TODO |

---

## Triage notes

* "Status" reflects the state in the public `braindecode/braindecode`
  repository, not the state of any private fork or experimental branch.
* If you start work on a `:calendar:` item, open a draft PR and flip the
  status to `:hammer:` in the same PR so others can see the lane is
  taken.
* When an item ships, replace the status with `:white_check_mark:` and
  link the merge commit / PR in the right-hand column.
