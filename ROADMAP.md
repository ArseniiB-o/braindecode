# Braindecode Roadmap

This document tracks the prioritized engineering and product plan for the
next development cycles of Braindecode. It is a living document — entries
are added, removed, or re-prioritized as the project evolves. Items are
grouped by priority, not by release.

The roadmap complements `docs/whats_new.rst` (a per-release changelog) and
the GitHub Issues / Discussions backlog (where each item is tracked in
detail). When picking up a task, please open or claim the corresponding
issue first.

## Guiding principles

Braindecode's north-star use cases:

1. A neuroscientist who wants to apply deep learning to their EEG / ECoG /
   MEG / EMG / iEEG recordings should be able to go from raw data to a
   trained model in a single notebook, using a published architecture.
2. A deep-learning researcher who wants to work with neurophysiological
   data should find the same standard model zoo, datasets, augmentations,
   and pretraining recipes they expect in any modern PyTorch ecosystem.
3. Trained models should be shareable and reusable through the Hugging
   Face Hub with no boilerplate.

Every roadmap item below should make at least one of those journeys
shorter, more reproducible, or more correct.

## P0 — Quick wins (landed or in progress)

These are low-risk, high-impact fixes that reduce friction immediately.
Most of them ship in the 1.6.0 development line.

- **PyPI metadata correctness.** Promote `Development Status` from
  `3 - Alpha` to `5 - Production/Stable`, drop the unrelated
  `Topic :: Software Development :: Build Tools` classifier, and add
  bio-informatics / medical / healthcare classifiers. *(done — see
  `whats_new.rst`)*
- **`braindecode[hug]` → `[hub]` typos.** Eight places across docs and
  model docstrings advertised a non-existent extra. *(done)*
- **CI matrix restores Python 3.11.** Required by `requires-python`, also
  re-enables the Codecov upload step. *(done)*
- **`CONTRIBUTING.md` opens with a real "Questions and Support" section.**
  Replaces the `Todo: In case of questions, discord?` placeholder.
  *(done)*
- **This roadmap document.** *(done)*

Still outstanding:

- Switch hard runtime dependencies on `linear_attention_transformer` and
  `rotary_embedding_torch` to soft imports under a new
  `braindecode[foundation]` extra. These packages are only needed for a
  handful of foundation models and currently bloat every install.
- Add a `Dockerfile` and `.devcontainer/devcontainer.json` for a
  reproducible CUDA + MNE + Braindecode environment.
- Adopt `towncrier` for `whats_new.rst` so contributors drop a
  `changelog.d/<pr>.<type>` fragment instead of editing the global file
  (which is a frequent source of merge conflicts).

## P1 — Targeted for 1.6.x / 1.7.0

### Continuous integration & quality

- **GPU runner.** Self-hosted or third-party (e.g. cirun.io, depot.dev).
  Required to actually exercise `F.scaled_dot_product_attention`,
  flash-attention paths, and the larger foundation models
  (`EEGPT`, `LaBraM`, `BENDR`, `SignalJEPA`) under realistic conditions.
- **Tighten Codecov floor to 80 %.** Currently `codecov.yml` advertises a
  `70…100` range. With 874 tests we should be comfortably above 80 %.
- **Snapshot tests for every model in the zoo.** Seed-pinned forward pass
  → assert on output shape and on a hash of a small fixed input. Catches
  silent numerical regressions across PyTorch upgrades.
- **Parametric tests on the `n_chans` / `n_times` / `sfreq` /
  `input_window_seconds` / `chs_info` auto-inference matrix in
  `EEGModuleMixin`.** This is where most user-facing API bugs originate.
- **Reduce the `# type: ignore` count from 84 to under 20** by tightening
  signatures in `braindecode/models/base.py`, `eegneuralnet.py`,
  `classifier.py`, and `regressor.py`.

### Training & callbacks

- `braindecode.training.callbacks` should ship first-class wrappers for:
  - `EarlyStopping` with best-checkpoint restore (skorch's default
    requires manual wiring).
  - Stochastic Weight Averaging (SWA).
  - Exponential Moving Average of model weights (EMA) — useful for
    foundation-model finetuning.
  - `WandbCallback`, `MLflowCallback`, `TensorBoardCallback` for
    experiment tracking, plus a documented example.
- `braindecode.training.losses` should grow:
  - `InfoNCE` / `NTXentLoss` for contrastive pretraining.
  - `CTCLoss` (or a thin EMG-friendly wrapper) for `EMG2QwertyNet`.
  - `FocalLoss` for imbalanced sleep-staging / seizure-detection tasks.

### Pretraining and foundation models

- **Unified self-supervised pretraining loop.** Today, every foundation
  model (`SignalJEPA`, `BENDR`, `EEGPT`, `BIOT`, `CBraMod`, `LaBraM`,
  `CodeBrain`) re-implements its own SSL stack inside a single 1 000–
  1 800-line file. Goal: `braindecode.training.pretraining.SSLTrainer`
  that accepts a backbone, a tokenizer / masker, and a head, and lets
  contributors describe a new recipe declaratively. The current god-file
  models become *configurations* of this trainer rather than parallel
  re-implementations.
- **Refactor the four largest model files** into subpackages
  (`braindecode/models/<name>/{model,blocks,pretrained,tokenizer}.py`):
  - `meta_neuromotor.py` (1 774 LOC)
  - `signal_jepa.py` (1 597 LOC)
  - `labram.py` (1 560 LOC)
  - `eegpt.py` (1 437 LOC)
- **Auto-generated Hugging Face Hub model cards.**
  `model.push_to_hub(..., generate_card=True)` should assemble a README
  containing architecture, parameter count, training sfreq, dataset
  fingerprint, headline metrics, and the BibTeX entry for the source
  paper.
- **Lazy imports.** `braindecode/models/__init__.py` currently imports
  every model (and therefore `transformers`, `linear_attention_transformer`,
  `rotary_embedding_torch`, etc.) at import time, which makes cold-start
  painful. Migrate to PEP 562 module-level `__getattr__`.

### Benchmarking

- A `benchmarks/` package that runs every model from `summary.csv` on a
  canonical task suite (BCI-IV-2a, Sleep-EDF, TUAB, SEED-IV) on a nightly
  GitHub Actions schedule and publishes results to
  `docs/benchmarks.rst`. Treat it as a public leaderboard — that is the
  single largest competitive advantage Braindecode could ship.
- Auto-update the `#Parameters` and `get_#Parameters` columns of
  `summary.csv` from the actual model constructors (today they are
  hand-maintained and drift).

### Documentation

- A `docs/migration/` directory with explicit upgrade notes between
  every minor release (1.4 → 1.5, 1.5 → 1.6). Auto-extract candidate
  entries from `@deprecated` decorators with a Sphinx extension.
- One-click Google Colab badges on every tutorial in
  `docs/auto_examples/`, with pre-downloaded checkpoints from the
  `braindecode/` org on the Hub.
- A dedicated `docs/deployment.rst` covering ONNX export, TorchScript,
  INT8 quantization, and the streaming inference engine (see P2).

## P2 — Strategic, 1.7+ (6–12 months)

### Deployment and edge

- **ONNX / TorchScript export examples.** At least `EEGNet`,
  `ShallowFBCSPNet`, `EEGConformer`, `USleep`, and one foundation model.
  Add a `braindecode.export` module so users can call
  `braindecode.export.to_onnx(model)` and get a validated artifact.
- **Quantization.** `torch.ao.quantization` PTQ and QAT recipes for the
  small convolutional models that realistically run on embedded EEG
  devices (EEGNet, ShallowFBCSPNet, EEGITNet, SCCNet).
- **Streaming / online inference.** A `braindecode.streaming` module
  built around a ring-buffer over MNE's `Stream` API. Provides
  sliding-window inference with configurable overlap-add for any
  Braindecode model — unlocks the realtime BCI use case that the
  library does not address today.

### Distribution and scale

- **Multi-GPU / FSDP / DeepSpeed examples** for foundation-model
  pretraining, alongside `examples/advanced_training/`.
- **PyTorch Lightning as a first-class API**, parallel to
  `EEGClassifier` / `EEGRegressor`. Provide a `LightningEEGModule` mixin
  and migrate one tutorial per major workflow (trialwise, cropped, SSL,
  sleep staging) to Lightning. Lightning dominates academic ML; we lose
  a generation of users if Braindecode stays skorch-only.

### Command-line and configuration

- A real CLI, e.g. `braindecode train --model EEGNet --dataset
  bnci2014_001 --config foo.yaml`. Implemented with `typer` and
  `pydantic` (already a runtime dependency). Plays well with Hydra /
  Optuna sweeps.
- A `braindecode.hub.search(task=..., n_chans=...)` registry on the
  library side that resolves to checkpoints on
  `huggingface.co/braindecode/`, so users do not need to manually map
  classes to repo IDs.

### New scientific capabilities

- **Federated learning hooks** (Flower or PySyft integration). EEG data
  is privacy-sensitive and almost always sits behind hospital firewalls;
  this is a natural USP that no competing library has shipped.
- **Interpretability suite.** GradCAM / Integrated Gradients / SmoothGrad
  / SHAP per model, verified with unit tests. Currently `captum` is an
  optional dependency and the wiring is per-tutorial rather than
  per-model.
- **Tokenizer / discrete-code module.** `CBraMod` and `CodeBrain`
  introduce VQ-VAE-style discrete tokens for EEG; unify these under a
  `braindecode.tokenizers` namespace so future SSL recipes can share
  them.

## P3 — Nice-to-have / experimental

- `braindecode-bench` as a separate companion package + CLI (think
  `mteb` for EEG embeddings).
- Video tutorials and an interactive model-zoo browser that goes beyond
  the static `summary.csv`.
- First-class support for additional modalities (fNIRS, pupillometry,
  EOG) that already partially work through the MNE pipeline.

## Out of scope (for now)

- Re-implementing dataset I/O that MNE-Python already covers well.
- Maintaining preprocessing primitives that duplicate `eegprep`.
- Owning a new public web service / leaderboard frontend. The
  benchmark suite from P1 should publish to GitHub Pages via the
  existing docs build.

## How to contribute to the roadmap

Open or comment on the corresponding GitHub issue. If an item does not
yet have one, please open it and link back to this document. Substantive
direction changes are discussed in GitHub Discussions before any
roadmap edit.
