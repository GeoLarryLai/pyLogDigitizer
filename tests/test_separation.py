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
