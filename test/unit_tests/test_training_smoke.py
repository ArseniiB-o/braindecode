# Authors: Arsenii Boichenko
#
# License: BSD-3

"""End-to-end smoke test for the basic training loop.

This test mirrors what the ``examples/model_building/plot_basic_training_epochs.py``
sphinx-gallery example does, but using purely synthetic data so it can
run on CI in well under a minute without any dataset download. It
guards against regressions in the glue layer between the skorch-based
``EEGClassifier`` and the model zoo — which historically broke silently
because the gallery examples are not executed during the test job.

We deliberately do **not** check accuracy here: a 1-epoch fit on white
noise has nothing to predict. The contract is purely "the public API
runs to completion without raising and produces output of the right
shape".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from braindecode import EEGClassifier
from braindecode.models import EEGNetv4


@pytest.fixture
def synthetic_dataset():
    """Two-class synthetic EEG: (batch, n_chans, n_times) with int labels."""
    rng = np.random.default_rng(0)
    n_samples = 16
    n_chans = 8
    n_times = 256
    X = rng.standard_normal(size=(n_samples, n_chans, n_times)).astype(np.float32)
    y = (rng.integers(0, 2, size=n_samples)).astype(np.int64)
    return X, y


def test_eegclassifier_fits_synthetic_one_epoch(synthetic_dataset):
    """EEGClassifier(EEGNetv4).fit(...) runs end-to-end on synthetic data."""
    X, y = synthetic_dataset
    n_chans = X.shape[1]
    n_times = X.shape[2]

    clf = EEGClassifier(
        module=EEGNetv4,
        module__n_chans=n_chans,
        module__n_outputs=2,
        module__n_times=n_times,
        module__sfreq=128.0,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-3,
        train_split=None,
        max_epochs=1,
        batch_size=8,
        verbose=0,
        device="cpu",
    )
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)
    assert preds.dtype.kind in {"i", "u"}
    assert set(np.unique(preds)).issubset({0, 1})


def test_eegclassifier_predict_proba_shape(synthetic_dataset):
    """``predict_proba`` returns one row per sample and one column per class."""
    X, y = synthetic_dataset
    clf = EEGClassifier(
        module=EEGNetv4,
        module__n_chans=X.shape[1],
        module__n_outputs=2,
        module__n_times=X.shape[2],
        module__sfreq=128.0,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        max_epochs=1,
        batch_size=8,
        verbose=0,
        device="cpu",
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
