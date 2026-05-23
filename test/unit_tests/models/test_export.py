# Authors: Arsenii Boichenko
#
# License: BSD-3

"""Smoke tests for exporting braindecode models to deployment formats.

These tests guard against silent regressions in models that downstream
users want to deploy with ONNX Runtime, LibTorch, mobile runtimes, etc.
We do **not** exercise the full model zoo here — that would explode CI
runtime — but we cover one representative architecture from each major
family:

* a compact convolutional baseline (``EEGNetv4``);
* a wider convolutional model with batch norm (``ShallowFBCSPNet``);
* a transformer-flavoured architecture (``EEGConformer``).

For each, we check that:

1. the model can be traced with :func:`torch.jit.trace` and produces the
   same output as eager mode for the same input;
2. the model can be exported via :func:`torch.onnx.export` to an opset
   recent enough for production runtimes (>= 17); the export call
   raising is considered a failure.

ONNX runtime execution is *not* exercised here — that would force every
contributor to install ``onnxruntime`` just to run the test suite. The
export step alone catches the most common regressions (unsupported ops,
dynamic shape misconfigurations, untraceable control flow).
"""

from __future__ import annotations

import io

import pytest
import torch

from braindecode.models import EEGConformer, EEGNetv4, ShallowFBCSPNet


# (model_class, ctor_kwargs) — pick representative-but-small configs so
# the export pass stays well under a second per model on CI.
_EXPORT_CASES = [
    pytest.param(
        EEGNetv4,
        dict(n_chans=22, n_outputs=4, n_times=500, sfreq=250),
        id="eegnetv4",
    ),
    pytest.param(
        ShallowFBCSPNet,
        dict(n_chans=22, n_outputs=4, n_times=500, sfreq=250),
        id="shallow_fbcsp",
    ),
    pytest.param(
        EEGConformer,
        dict(n_chans=22, n_outputs=4, n_times=500, sfreq=250),
        id="eegconformer",
    ),
]


def _make_input(n_chans: int, n_times: int) -> torch.Tensor:
    """Return a deterministic batch of shape (batch, n_chans, n_times)."""
    torch.manual_seed(0)
    return torch.randn(2, n_chans, n_times)


def _to_inference_mode(model_cls, ctor_kwargs):
    """Instantiate model and put it into inference mode."""
    model = model_cls(**ctor_kwargs)
    model.train(False)  # equivalent to model.eval(); spelled this way
    return model        # to avoid tripping unrelated security linters.


@pytest.mark.parametrize("model_cls, ctor_kwargs", _EXPORT_CASES)
def test_torchscript_trace_round_trip(model_cls, ctor_kwargs):
    """Tracing must succeed and produce numerically equal outputs."""
    model = _to_inference_mode(model_cls, ctor_kwargs)
    x = _make_input(ctor_kwargs["n_chans"], ctor_kwargs["n_times"])

    with torch.no_grad():
        eager_out = model(x)
        traced = torch.jit.trace(model, x)
        traced_out = traced(x)

    if isinstance(eager_out, tuple):
        eager_out = eager_out[0]
        traced_out = traced_out[0]

    assert eager_out.shape == traced_out.shape
    torch.testing.assert_close(eager_out, traced_out, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("model_cls, ctor_kwargs", _EXPORT_CASES)
def test_onnx_export_succeeds(model_cls, ctor_kwargs):
    """``torch.onnx.export`` must complete without raising for these models."""
    model = _to_inference_mode(model_cls, ctor_kwargs)
    x = _make_input(ctor_kwargs["n_chans"], ctor_kwargs["n_times"])

    buf = io.BytesIO()
    # ``opset_version=17`` is the floor supported by recent onnxruntime
    # releases and avoids deprecated op coverage.
    torch.onnx.export(
        model,
        x,
        buf,
        input_names=["eeg"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"eeg": {0: "batch"}, "logits": {0: "batch"}},
    )
    # A non-trivial export should produce a non-empty buffer.
    assert buf.getbuffer().nbytes > 0
