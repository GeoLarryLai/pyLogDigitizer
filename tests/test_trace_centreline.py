# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Regression: a dashed curve is traced along its ink centreline, joined straight across dash gaps,
with no staircase (vertical holds through gaps followed by sideways jumps)."""
import numpy as np
import pytest

from pylogdigitizer import curve_tracing as ct


def _dashed_sine(h=1500, w=320, amp=110.0, period=360.0, on=18.0, off=11.0, half=3):
    """Bool panel with a dashed sinusoid (dashes measured along the curve) and its true x(row)."""
    rows = np.arange(h, dtype=float)
    xt = w / 2 + amp * np.sin(2 * np.pi * rows / period)
    fine = np.arange(0, h - 1, 0.05)
    xf = w / 2 + amp * np.sin(2 * np.pi * fine / period)
    s = np.r_[0, np.cumsum(np.hypot(np.diff(fine), np.diff(xf)))]
    keep = (s % (on + off)) < on
    panel = np.zeros((h, w), bool)
    for r, x in zip(fine[keep], xf[keep]):
        ri, xi = int(round(r)), int(round(x))
        panel[max(0, ri - half):ri + half + 1, max(0, xi - half):xi + half + 1] = True
    return panel, xt


def _plateaus(x, xt_rows, min_len=3, slope=1.5):
    """Number of runs of >= min_len DP rows where x does not move while the true curve moves
    more than ``slope`` px per DP row."""
    dx = np.abs(np.diff(x)) < 0.25
    dt = np.abs(np.diff(xt_rows)) > slope
    bad = dx & dt
    runs, n = 0, 0
    for b in bad:
        n = n + 1 if b else 0
        if n == min_len:
            runs += 1
    return runs


@pytest.mark.parametrize("use", ["python", None])
def test_dashed_sine_follows_ink_without_staircase(use):
    panel, xt = _dashed_sine()
    T = ct.trace_track(panel, 0, px_per_ft=10, min_width=3, min_area=0, straight_len=0, grid_frac=1.01,
                       use=use or None)
    rows = np.asarray(T["rows"], float)
    x = np.asarray(T["x"], float)
    m = ~(T["off_left"] | T["off_right"])
    m[:5] = m[-5:] = False
    truth = np.interp(rows, np.arange(len(xt)), xt)
    err = np.abs(x - truth)[m]
    assert np.median(err) < 2.0 and np.percentile(err, 95) < 5.0, (np.median(err), np.percentile(err, 95))
    assert _plateaus(x[m], truth[m]) == 0
    # the raw DP cells (no centreline step) do show the defect this guards against
    xr = np.asarray(T["path"], float) * T["fx"] + T["fx"] / 2
    assert _plateaus(xr[m], truth[m]) > 10


def test_gap_rows_are_straight_joins_not_holds():
    """Two ink pieces with an empty gap between: x across the gap is the straight line between their
    ends (connected points, no smoothing of the ink rows themselves)."""
    cost = np.ones((20, 40))
    cost[0:6, 5] = 0.0          # piece 1 ends at row 5, column 5
    cost[14:20, 25] = 0.0       # piece 2 starts at row 14, column 25
    path = np.r_[np.full(10, 5), np.full(10, 25)] + 1      # raw DP: hold, then jump (pad = 1)
    cost_p = np.pad(cost, ((0, 0), (1, 1)), constant_values=0.8)
    x, on = ct.ink_centreline(cost_p, path, pad=1, fx=2, fy=2)
    assert on[:6].all() and on[14:].all() and not on[6:14].any()
    assert np.allclose(x[:6], 5 * 2 + 1) and np.allclose(x[14:], 25 * 2 + 1)
    gap = x[5:15]
    assert np.all(np.diff(gap) > 0) and np.allclose(np.diff(gap), np.diff(gap)[0])


def test_qc_grid_keeps_null_stretches():
    """QC scores only depths with values: a null stretch is not filled by interpolating across it."""
    from pylogdigitizer.qc_compare import to_grid
    d = np.arange(10.0); v = d.copy(); v[4:6] = np.nan
    g = to_grid(d, v, np.arange(0, 9.5, 0.5))
    assert np.isnan(g[7:12]).all() and np.isfinite(np.r_[g[:7], g[12:]]).all()
    assert np.allclose(g[[0, 6, 12, 18]], [0, 3, 6, 9])
