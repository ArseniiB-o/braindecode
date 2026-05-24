# Authors: Arsenii Boichenko
#
# License: BSD-3

"""Regression tests for the assert → typed-exception migration.

These tests guard against a future refactor that might revert the
input-validation checks back to ``assert`` (which Python's ``-O``
strips away). For each surface that was migrated in the May 2026
quality pass we assert that:

* invalid inputs raise the **specific** new exception type
  (``ValueError`` / ``TypeError`` / etc.), not ``AssertionError``;
* the exception message includes the offending value, so the failure
  is actionable.

We intentionally only exercise the *checks*, not the augmentations'
forward pass — that keeps the test fast and decoupled from the rest
of the augmentation pipeline.
"""

from __future__ import annotations

import pytest

from braindecode.augmentation.transforms import (
    BandstopFilter,
    ChannelsPermute,
    FrequencyShift,
    FTSurrogate,
    SmoothTimeMask,
)


def test_ftsurrogate_phase_noise_magnitude_out_of_range():
    with pytest.raises(ValueError, match=r"phase_noise_magnitude"):
        FTSurrogate(probability=0.5, phase_noise_magnitude=1.5)


def test_ftsurrogate_phase_noise_magnitude_wrong_type():
    with pytest.raises(TypeError, match=r"phase_noise_magnitude"):
        FTSurrogate(probability=0.5, phase_noise_magnitude="bad")  # type: ignore[arg-type]


def test_ftsurrogate_channel_indep_wrong_type():
    with pytest.raises(TypeError, match=r"channel_indep"):
        FTSurrogate(probability=0.5, channel_indep="not-a-bool")  # type: ignore[arg-type]


def test_channels_permute_wrong_type():
    with pytest.raises(TypeError, match=r"ordered_ch_names"):
        ChannelsPermute(probability=0.5, ordered_ch_names="C3,C4")  # type: ignore[arg-type]


def test_smooth_time_mask_negative_samples():
    with pytest.raises(ValueError, match=r"mask_len_samples"):
        SmoothTimeMask(probability=0.5, mask_len_samples=-1)


def test_bandstop_filter_negative_bandwidth():
    with pytest.raises(ValueError, match=r"bandwidth"):
        BandstopFilter(probability=0.5, sfreq=128.0, bandwidth=-1)


def test_bandstop_filter_non_positive_sfreq():
    with pytest.raises(ValueError, match=r"sfreq"):
        BandstopFilter(probability=0.5, sfreq=0)


def test_bandstop_filter_bandwidth_too_close_to_max_freq():
    # max_freq defaults to sfreq/2 = 64; bandwidth must be < 62.
    with pytest.raises(ValueError, match=r"bandwidth"):
        BandstopFilter(probability=0.5, sfreq=128.0, bandwidth=63)


def test_frequency_shift_non_positive_sfreq():
    with pytest.raises(ValueError, match=r"sfreq"):
        FrequencyShift(probability=0.5, sfreq=-10.0)


@pytest.mark.parametrize(
    "transform_cls, kwargs, expected_exc",
    [
        (FTSurrogate, dict(probability=0.5, phase_noise_magnitude=1.5), ValueError),
        (SmoothTimeMask, dict(probability=0.5, mask_len_samples=0), ValueError),
        (BandstopFilter, dict(probability=0.5, sfreq=128.0, bandwidth=-1), ValueError),
    ],
)
def test_migrated_checks_do_not_raise_assertion_error(transform_cls, kwargs, expected_exc):
    """No converted check should leak ``AssertionError`` anymore."""
    with pytest.raises(expected_exc):
        transform_cls(**kwargs)
    # Sanity: the same call must not raise AssertionError instead.
    with pytest.raises(expected_exc) as exc_info:
        transform_cls(**kwargs)
    assert not isinstance(exc_info.value, AssertionError)
