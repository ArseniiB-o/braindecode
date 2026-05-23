# Authors: Arsenii Boichenko
#
# License: BSD-3

"""Tests for the BRAINDECODE_DISABLE_PICKLE_CACHE opt-out.

These tests exercise the small, pure-Python policy layer added to
``braindecode.datautil.serialization`` and intentionally do **not**
require MNE data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from braindecode.datautil import serialization as ser


@pytest.fixture(autouse=True)
def _reset_warned_paths():
    """Make sure each test starts with an empty warn-once set."""
    ser._WARNED_PICKLE_PATHS.clear()
    yield
    ser._WARNED_PICKLE_PATHS.clear()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("YES", True),
        ("on", True),
        ("  1  ", True),
        ("0", False),
        ("", False),
        ("no", False),
        ("anything-else", False),
    ],
)
def test_cache_disabled_env_parsing(monkeypatch, value, expected):
    """The env-var parser recognises a small set of truthy strings."""
    monkeypatch.setenv(ser._DISABLE_CACHE_ENV_VAR, value)
    assert ser._pickle_cache_disabled() is expected


def test_cache_disabled_default_false(monkeypatch):
    """When the env var is unset, the cache stays enabled."""
    monkeypatch.delenv(ser._DISABLE_CACHE_ENV_VAR, raising=False)
    assert ser._pickle_cache_disabled() is False


def test_warn_cache_load_fires_once(tmp_path):
    """Each cache path should emit exactly one UserWarning per process."""
    cache_file = tmp_path / "trial.pkl"
    cache_file.write_bytes(b"")

    with pytest.warns(UserWarning, match="cache"):
        ser._warn_pickle_cache_load(cache_file)

    # Second call for the same path is silent.
    with pytest.warns(None) as captured:
        ser._warn_pickle_cache_load(cache_file)
    assert all("cache" not in str(w.message) for w in captured)


def test_warn_cache_load_distinct_paths(tmp_path):
    """Two distinct cache paths each emit one warning."""
    a = tmp_path / "a.pkl"
    b = tmp_path / "b.pkl"
    a.write_bytes(b"")
    b.write_bytes(b"")

    with pytest.warns(UserWarning):
        ser._warn_pickle_cache_load(a)
    with pytest.warns(UserWarning):
        ser._warn_pickle_cache_load(b)


def test_warn_cache_load_uses_resolved_path(tmp_path):
    """Symlinked / relative paths pointing at the same file dedupe correctly."""
    real = tmp_path / "real.pkl"
    real.write_bytes(b"")

    with pytest.warns(UserWarning):
        ser._warn_pickle_cache_load(real)

    # A Path with ".." segments resolves to the same absolute path and
    # must not produce a second warning.
    indirect = Path(str(tmp_path / "subdir" / ".." / "real.pkl"))
    with pytest.warns(None) as captured:
        ser._warn_pickle_cache_load(indirect)
    assert all("cache" not in str(w.message) for w in captured)
