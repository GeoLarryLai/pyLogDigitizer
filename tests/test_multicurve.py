# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Stage H: several curves in one track (trace_multi, untangle, roughness)."""
import numpy as np
import pytest

pytest.importorskip("cv2")
from pylogdigitizer import grid_rectify as gr, curve_tracing as ct


def _two_curves(seed=0, H=2400, W=400):
    rng = np.random.default_rng(seed)
    v = np.arange(0, H, 0.25)
    wig = np.convolve(rng.normal(0, 1, len(v)), np.ones(160) / 160, "same")
    g = np.clip(200 + 150 * np.sin(v / 210) * np.cos(v / 150) + 12 * wig / wig.std(), 15, W - 15)
    cal = 130 + 60 * np.sin(v / 260)
    ink, _, _ = gr.synthetic_grid(H=H, W=W, nv=11, nh=25, margin=20, amp=0, shear=0, seed=seed)
    gr._stamp(ink, g, v, 3); gr._stamp(ink, cal, v, 3)
    return ink, v, g, cal


def test_second_pass_avoids_first_curve():
    ink, v, g, cal = _two_curves()
    A = ct.trace_track(ink, 0, px_per_ft=10)
    B = ct.trace_track(ink, 0, px_per_ft=10, avoid_paths=[A["path"]], avoid_cost=0.97)
    assert np.mean(np.abs(A["path"] - B["path"]) <= 3) < 0.05


def test_multi_returns_candidates_and_pick():
    ink, *_ = _two_curves()
    T = ct.trace_multi(ink, 0, n_curves=2, px_per_ft=10, min_on_ink=0)
    assert len(T["candidates"]) == 2 and T["picked"] in (0, 1)
    assert T is T["candidates"][T["picked"]] and all("roughness" in t for t in T["candidates"])
    assert ct.trace_multi(ink, 0, n_curves=2, pick=1, px_per_ft=10, min_on_ink=0)["picked"] == 1


def test_multi_on_single_curve_track_keeps_one():
    ink, _, _ = gr.synthetic_grid(H=1500, W=300, nv=9, nh=15, margin=20, amp=0, shear=0, seed=2)
    v = np.arange(0, 1500, 0.25); x = 150 + 100 * np.sin(v / 100); gr._stamp(ink, x, v, 3)
    T = ct.trace_multi(ink, 0, n_curves=2, px_per_ft=10)
    assert len(T["candidates"]) == 1 and np.median(np.abs(T["x"] - np.interp(T["rows"], v, x))) < 3


def test_untangle_restores_identity_at_crossings():
    t = np.arange(600.0)
    p, q = 100 + 0.5 * (t - 300), 100 - 0.4 * (t - 300)          # cross at t = 300
    a = np.where(t < 300, p, q); b = np.where(t < 300, q, p)      # a DP that hopped at the crossing
    sw = ct.untangle(a, b)
    a2 = np.where(sw, b, a)
    touch = np.abs(p - q) <= 4                                    # inside the contact the two are within 4 px
    assert np.max(np.abs(a2 - p)[~touch]) < 1e-9 and np.max(np.abs(a2 - p)) <= 4


def test_roughness_prefers_detail():
    rng = np.random.default_rng(0); t = np.arange(3000.0)
    smooth = 100 + 30 * np.sin(t / 400); rough = smooth + np.convolve(rng.normal(0, 8, 3000), np.ones(5) / 5, "same")
    assert ct.roughness(rough) > 3 * ct.roughness(smooth)
