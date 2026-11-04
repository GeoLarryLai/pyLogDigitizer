# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Synthetic renderer + bundled ONNX segmentation model (plain onnxruntime)."""
import numpy as np
import pytest

pytest.importorskip("cv2")
from pylogdigitizer import segmentation as sg


def test_renderer_deterministic_and_labelled():
    a, la = sg.render_synthetic(seed=7); b, lb = sg.render_synthetic(seed=7)
    assert (a == b).all() and (la == lb).all()
    assert ((la > 0) == a).all()                      # every ink pixel has a class, paper has none
    assert (la == sg.CURVE).sum() > 300 and (la == sg.GRID).sum() > 1000


def test_model_on_held_out_synthetic():
    pytest.importorskip("onnxruntime")
    assert sg.available(), "bundled model missing"
    tp = fn = fp = 0
    for s in range(900_000, 900_012):                 # seeds never used in training batches' validation set
        ink, lab = sg.render_synthetic(384, 256, seed=s)
        pr = sg.get_segmenter().predict(ink).argmax(0)
        c, t = (pr == sg.CURVE) & ink, lab == sg.CURVE
        tp += (c & t).sum(); fn += (~c & t).sum(); fp += (c & ~t).sum()
    assert tp / (tp + fn) > 0.9 and tp / (tp + fp) > 0.85, (tp / (tp + fn), tp / (tp + fp))


def test_trace_with_segmentation_cost():
    pytest.importorskip("onnxruntime")
    from pylogdigitizer import grid_rectify as gr, curve_tracing as ct
    import cv2
    H, W = 2000, 400
    grid, _, _ = gr.synthetic_grid(H=H, W=W, nv=11, nh=25, margin=20, amp=0, shear=0, seed=5)
    v = np.arange(0, H, 0.25); uc = 20 + 360 * (0.5 + 0.35 * np.sin(v / 120))
    cur = np.zeros_like(grid); gr._stamp(cur, uc, v, 3)
    txt = np.zeros((H, W), np.uint8)
    for k in range(0, H, 250):
        cv2.putText(txt, "4350 GAPI", (30, k + 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3)
    full = grid | cur | (txt > 0)
    err = {}
    for m in (False, True):
        T = ct.trace_track(full, 0, px_per_ft=10, seg=m)
        err[m] = float((np.abs(T["x"] - np.interp(T["rows"], v, uc)) > 10).mean())
    assert err[True] <= err[False] and err[True] < 0.05, err
