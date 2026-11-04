# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Rule-based grid / curve / other separation on a synthetic track with text and a stamp."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from pylogdigitizer import grid_rectify as gr, separation as sp, curve_tracing as ct

H, W = 2400, 420


@pytest.fixture(scope="module")
def sheet():
    ink, _, _ = gr.synthetic_grid(H=H, W=W, nv=11, nh=25, margin=20, amp=0, shear=0, seed=4)
    v = np.arange(0, H, 0.25); uc = 20 + 380 * (0.5 + 0.3 * np.sin(v / 140))
    curve = np.zeros_like(ink); gr._stamp(curve, uc, v, 4)
    img = np.zeros((H, W), np.uint8)
    for k in range(0, H, 300):
        cv2.putText(img, "4350 FT", (40 + (k % 600) // 3, k + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 3)
    cv2.rectangle(img, (250, 900), (380, 1050), 255, -1)
    stamp = np.zeros_like(ink); stamp[895:1056, 245:386] = True
    return ink, curve, img > 0, stamp, v, uc


def test_labels(sheet):
    grid, curve, txt, stamp, _, _ = sheet
    full = grid | curve | txt
    lab = sp.classify(full); o = lab == sp.OTHER
    only_txt = txt & ~curve & ~grid
    assert (o & only_txt).sum() / only_txt.sum() > 0.9
    assert (o & curve & ~stamp).sum() == 0                     # no curve ink lost outside the stamp
    g = lab == sp.GRID
    assert (g & grid).sum() / grid.sum() > 0.85 and (g & curve).sum() / curve.sum() < 0.05


def test_masking_helps_tracing(sheet):
    grid, curve, txt, _, v, uc = sheet
    full = grid | curve | txt
    wrong = []
    for m in (False, True):
        T = ct.trace_track(full, 0, px_per_ft=10, mask_other=m)
        wrong.append(float((np.abs(T["x"] - np.interp(T["rows"], v, uc)) > 10).mean()))
    assert wrong[1] < wrong[0], wrong


def test_hairline_curve_falls_back_to_thin_mode():
    """A 1 px curve is erased by the default width-3 opening; trace_track must notice and recover it."""
    H, W = 1500, 300
    v = np.arange(0, H, 0.25); uc = 150 + 90 * np.sin(v / 90)
    pan = np.zeros((H, W), bool); gr._stamp(pan, uc, v, 1)
    T = ct.trace_track(pan, 0, px_per_ft=7.5)
    assert T["params"].get("thin_fallback")
    err = np.abs(T["x"] - np.interp(T["rows"], v, uc))
    assert np.median(err) < 3 and (err > 10).mean() < 0.05


def test_first_depth_compared_with_first_confirmed_label():
    from pylogdigitizer import depth_calibration as dc
    rows = [50 + 600 * k for k in range(8)]                 # lattice index 0 is header junk read as "900"
    vals = [900] + [1500 + 100 * (k - 1) for k in range(1, 8)]
    vals[0] = 7                                             # unreadable junk: does not match the lattice
    _, _, _, info = dc.calibrate_labels(rows, rows, values=vals, first_depth=1500)
    assert "ocr_disagrees_with_first_depth" not in info
    _, _, _, info = dc.calibrate_labels(rows, rows, values=vals, first_depth=1600)
    assert info["ocr_disagrees_with_first_depth"] == (1500.0, 1600)


def test_edge_jump_wraps():
    from pylogdigitizer import scale_wraparound as sw
    x = np.r_[np.linspace(50, 99, 20), np.linspace(1, 40, 30), np.linspace(40, 2, 10), np.linspace(98, 60, 20)]
    assert sw.edge_jump_wraps(x, 100) == [(20, 60)]
    assert sw.edge_jump_wraps(x[::-1] * 0 + 50, 100) == []
    # the left-wrapping (DT-style) mirror
    assert sw.edge_jump_wraps(100 - x, 100, wrap_side="left") == [(20, 60)]
    # faint exit: the path loses the curve at 0.7 W and lands on the backup trace at the left edge
    x2 = np.r_[np.linspace(50, 72, 20), np.linspace(3, 40, 30), np.linspace(40, 2, 10), np.linspace(75, 60, 20)]
    assert sw.edge_jump_wraps(x2, 100) == [(20, 60)]
    # an ordinary swing inside the track is not a wrap
    assert sw.edge_jump_wraps(np.r_[np.linspace(60, 30, 20), np.linspace(30, 8, 20)], 100) == []
