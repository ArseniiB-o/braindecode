# Security Policy

## Supported Versions

Security fixes are only backported to the most recent minor release line.

| Version | Supported          |
| ------- | ------------------ |
| 1.5.x   | :white_check_mark: |
| 1.4.x   | :x: (archived)     |
| < 1.4   | :x:                |

## Reporting a Vulnerability

**Please do not open a public issue for security problems.**

Send a private report to the maintainers via the
[GitHub Security Advisory](https://github.com/braindecode/braindecode/security/advisories/new)
form, or, if that is not possible, email any maintainer listed in
`pyproject.toml` under `maintainers`.

Please include:

* A clear description of the vulnerability and the affected component.
* A minimal reproducer (a small script, a dataset name, a saved file you
  can share, etc.) so we can confirm the issue.
* The version of braindecode (`python -c "import braindecode; print(braindecode.__version__)"`)
  and Python you are using.

We aim to acknowledge new reports within **5 working days** and to
publish a fix or a documented workaround within **30 days** of confirming
a high-severity issue.

## Known security considerations

The following behaviours are **by design** but can be misused. They are
documented so that downstream users can make an informed decision.

### 1. Binary cache files (`.pkl`) on disk

`braindecode.datautil.serialization.load_concat_dataset` and related
helpers read and write a small binary cache next to each `.fif` file to
speed up subsequent reads. The cache is a Python binary serialization
artefact and is fundamentally **unsafe to deserialize from an untrusted
source** — a malicious cache file could execute arbitrary code on load.

**Recommendations:**

* Only load preprocessed datasets that **you produced yourself** or that
  come from a trusted, integrity-checked source.
* If you receive a dataset directory from a collaborator, delete any
  `*.pkl` files inside it before loading. braindecode will transparently
  regenerate them from the underlying `.fif` files.
* On shared filesystems, prefer read-only mounts for cached datasets.

A safer, opt-in serialization path based on `joblib` is on the roadmap
(`ROADMAP.md`, item B1).

### 2. Loading PyTorch checkpoints from untrusted sources

`torch.load(..., weights_only=False)` can execute arbitrary code embedded
in a checkpoint. Always pass `weights_only=True` (the recommended default
since PyTorch 2.4) when loading model weights from a source you do not
fully control. Examples in the documentation have been updated to follow
this pattern.

### 3. Network downloads

Several dataset loaders fetch raw EEG / MEG files over HTTP(S) on first
use (MOABB datasets, the TUH corpora, HuggingFace Hub checkpoints). The
underlying libraries (`mne`, `wfdb`, `huggingface_hub`) are responsible
for transport security. If you operate behind a corporate proxy, follow
the configuration guidance of each library.

## Dependency security

`braindecode` runs on top of a large scientific Python stack
(PyTorch, MNE, NumPy, SciPy, pandas, etc.). Critical security updates
in those projects are picked up automatically when you `pip install -U`,
provided your environment allows the upgrade.

We rely on
[pre-commit.ci](https://pre-commit.ci/) and GitHub's Dependabot to keep
our own pinned tools (linters, doc builders) up to date.
