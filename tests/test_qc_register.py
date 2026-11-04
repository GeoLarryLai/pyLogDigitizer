# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Depth registration + value fit in qc_compare."""
import numpy as np
from pylogdigitizer import qc_compare as qc


def _log(n, seed):
    rng = np.random.default_rng(seed)
    x = np.convolve(rng.normal(0, 1, n), np.ones(5) / 5, "same")   # thin beds, ~2 ft
    return 60 + 10 * x / x.std()


def test_registration_recovers_slow_stretch():
    d = np.arange(1000, 3000, 0.5); ref = _log(len(d), 1)
    drift = 3.0 * np.sin((d - 1000) / 2000 * np.pi)              # up to 3 ft, smooth (stretch-type error)
    test_d = d + drift
    c = qc.compare(d, ref, test_d, ref * 1.0)
    assert c["registered"]["r"] > c["aligned"]["r"] + 0.05
    assert c["registered"]["r"] > 0.95
    assert abs(c["registered"]["gain"] - 1) < 0.05


def test_registration_does_not_manufacture_correlation():
    d = np.arange(1000, 3000, 0.5)
    c = qc.compare(d, _log(len(d), 1), d, _log(len(d), 2))     # unrelated curves
    assert abs(c["registered"]["r"]) < 0.3


def test_value_fit_flags_scale_error():
    d = np.arange(0, 500, 0.5); ref = _log(len(d), 3)
    c = qc.compare(d, ref, d, ref / 2 + 5)                       # digitized on a scale twice too wide
    assert abs(c["registered"]["gain"] - 2) < 0.05 and abs(c["registered"]["offset"] + 10) < 0.5
