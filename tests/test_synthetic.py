# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Unit tests on synthetic images (fast, no example data needed)."""
import numpy as np
import pytest

from pylogdigitizer import (scan_cleanup as scl, track_grid_detection as tgd, depth_calibration as dc,
                            scale_calibration as sc, curve_tracing as ct, qc_compare as qc,
                            scale_wraparound as sw)


def synth_log(h=1200, seed=0, thick_bounds=True):
    """Track1 (60-460, 10 divs) | depth col (460-560) | track2 (560-960); grid rows every 20 px,
    heavy every 100; thick sinusoid curve in track1."""
    ink = np.zeros((h, 1000), bool)
    for x in np.linspace(60, 460, 11).astype(int):
        ink[:, x:x + 2] = True
    for x in np.linspace(560, 960, 11).astype(int):
        ink[:, x:x + 2] = True
    if thick_bounds:
        for x in (60, 460, 560, 960):
            ink[:, x - 3:x + 4] = True
    for y in range(0, h, 20):
        ink[y, 60:460] = ink[y, 560:960] = True
    for y in range(0, h, 100):
        ink[y:y + 4, 60:460] = ink[y:y + 4, 560:960] = True
    xs = (260 + 100 * np.sin(np.arange(h) / 100)).astype(int)
    for d in range(-3, 4):
        ink[np.arange(h), xs + d] = True
    return ink, xs


# --- scan_cleanup -----------------------------------------------------------------------------
def test_binarize_otsu_and_bool():
    img = np.full((20, 20), 230, np.uint8); img[5:10, 5:10] = 20
    b = scl.binarize(img)
    assert b[7, 7] and not b[0, 0]
    assert np.array_equal(scl.binarize(img > 128), ~(img > 128))


def test_deskew_recovers_shear():
    ink = np.zeros((400, 600), bool); ink[100::50, 50:550] = True
    ang = 0.6
    sh, _ = scl.deskew_blocks(ink, 400)  # straight lines: ~0
    sheared = np.zeros_like(ink)
    for x in range(600):
        s = int(round(x * np.tan(np.radians(ang))))
        sheared[:, x] = np.roll(ink[:, x], s)
    est = scl.estimate_skew(sheared, max_angle=2)
    assert abs(abs(est) - ang) < 0.15
    fixed, _ = scl.deskew(sheared, est)
    prof = fixed[:, 100:500].sum(1)
    assert prof.max() >= 0.9 * 400


def test_tiles_cover():
    cov = np.zeros((500, 700), int)
    for ys, xs in scl.iter_tiles((500, 700), 256, 32):
        cov[ys, xs] += 1
    assert cov.min() >= 1


def test_line_masks_and_thick():
    ink, xs = synth_log(400)
    h, v = scl.line_masks(ink, 61)
    assert h[0, 100] and v[200, 61]
    th = scl.thick_strokes(ink, 3)
    assert th[np.arange(10, 390), xs[10:390]].mean() > 0.8
    assert th[50, 61] == th[50, 61]  # boolean


def test_split_sections():
    rows = np.r_[np.arange(0, 1000, 100), np.arange(5000, 5600, 100)]
    s = scl.split_sections(rows)
    assert (0, 10) in s and (10, 16) in s


# --- track_grid_detection -----------------------------------------------------------------------
def test_detect_tracks_thick_and_spacing():
    ink, _ = synth_log(800)
    tr = tgd.detect_tracks(ink)
    kinds = [t["kind"] for t in tr]
    assert "depth" in kinds and kinds.count("track") == 2
    thin, _ = synth_log(800, thick_bounds=False)
    tr2 = tgd.detect_tracks_by_spacing(thin)
    assert [t["kind"] for t in tr2] == ["track", "depth", "track"]
    assert abs(tr2[0]["left"] - 61) <= 3 and abs(tr2[1]["right"] - 561) <= 3
    assert abs(tr2[0]["spacing"] - 40) < 1


def test_classify_grid():
    assert tgd.classify_grid(list(np.linspace(0, 400, 11))) == "linear"
    dec = np.log10(np.arange(1, 11)) * 100
    logx = np.r_[dec, dec[1:] + 100]
    assert tgd.classify_grid(list(logx)) == "log"


def test_grid_rows_major():
    ink, _ = synth_log(600)
    c, major = tgd.grid_rows(ink[:, 60:460], 0.6)
    assert np.allclose(np.diff(c[major]), 100, atol=2)


def test_follow_and_rectify():
    ink = np.zeros((3000, 400), bool)
    for y in range(3000):            # drifting boundaries
        o = int(y * 0.01)
        ink[y, 50 + o:53 + o] = ink[y, 300 + o:303 + o] = True
        ink[y, 175 + o] = y % 7 == 0
    cen, xs = tgd.follow_lines(ink, [51, 301], block=500, rows=(0, 3000))
    assert abs(xs[-1][0] - (51 + cen[-1] * 0.01)) < 3
    pan = tgd.rectify(ink, (0, 3000), cen, xs[:, 0], xs[:, 1], 250)
    assert pan.shape == (3000, 250)


# --- depth_calibration --------------------------------------------------------------------------
def test_depth_map_and_residuals():
    anchors = [(100.0, 1000.0), (1100.0, 1100.0), (2100.0, 1200.0), (3100.0, 1300.0)]
    dm = dc.fit_depth_map(anchors)
    assert dm(1600) == pytest.approx(1150)
    assert dm.inverse(1250) == pytest.approx(2600)
    assert dm.px_per_ft == pytest.approx(10)
    assert np.nanmax(np.abs(dc.residuals(anchors))) < 1e-9
    bad = anchors[:2] + [(2150.0, 1200.0)] + anchors[3:]
    assert np.nanmax(np.abs(dc.residuals(bad))) > 4


def test_assign_snap_densify():
    pairs = dc.assign_depths([100, 1100, 2100], 1500, 100)
    assert pairs[2] == (2100, 1700)
    heavy = np.arange(98, 2200, 100.0)
    snapped = dc.snap_to_lines(pairs, heavy)
    assert snapped[0][0] == 98
    dense = dc.densify_with_grid(snapped, heavy, 10)
    assert len(dense) >= 20


def test_regular_rows_rejects_noise():
    rows = [500, 1000, 1010, 1505, 1730, 2000, 2510, 3500, 3640]
    idx = dc.regular_rows(rows, min_spacing=100)
    assert [rows[i] for i in idx] == [500, 1000, 1505, 2000, 2510, 3500]


def test_label_finders():
    import cv2
    col = np.zeros((1500, 120), np.uint8)
    for y in (300, 800, 1300):
        cv2.putText(col, "4500", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    labs = dc.find_label_rows(col > 0)
    assert len(labs) == 3
    rot = np.ascontiguousarray(np.rot90(np.zeros((120, 1500), np.uint8)))
    for y in (300, 800, 1300):
        t = np.zeros((40, 100), np.uint8); cv2.putText(t, "4500", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
        rot[y - 50:y + 50, 40:80] = np.rot90(t)
    blobs = dc.find_label_blobs(rot > 0)
    assert [round(b["row"] / 100) for b in blobs] == [3, 8, 13]


# --- scale_calibration -----------------------------------------------------------------------
def test_scales():
    s = sc.linear_scale(0, 750, 100, 40)
    assert s(0) == 100 and s(750) == pytest.approx(40)
    assert s.inverse(70) == pytest.approx(375)
    lg = sc.log_scale(0, 400, 0.2, 2000)
    assert lg(100) == pytest.approx(2.0)
    assert lg.inverse(20) == pytest.approx(200)


# --- curve_tracing ---------------------------------------------------------------------------
@pytest.mark.parametrize("use", ["python", "numba", "rust"])
def test_backends_agree(use):
    if use == "rust" and not ct.HAVE_RUST:
        pytest.skip("rust extension not built")
    if use == "numba" and not ct.HAVE_NUMBA:
        pytest.skip("numba missing")
    rng = np.random.default_rng(1); c = rng.random((120, 50))
    ref = ct.trace_dp_python(c, 6, 0.05, 2.0)
    assert np.array_equal(ct.trace(c, 6, 0.05, 2.0, use=use), ref)


def test_trace_ignores_grid_and_follows_curve():
    ink, xs = synth_log(1200)
    track = ink[:, 64:457]
    gcols = [(l["x0"] - 2, l["x1"] + 2) for l in tgd.vertical_lines(track, 0.5)]
    cost = ct.build_cost(track, gcols, fy=3, fx=2)
    p = ct.trace(cost, 40, 0.02, 3.0)
    x = p * 2 + 1; truth = xs[1::3][: len(x)] - 64
    assert np.median(np.abs(x - truth)) <= 3


def test_anchor_pulls_path():
    c = np.ones((50, 40)); c[:, 5] = 0; c[:, 30] = 0.05
    p = ct.trace(c, 3, 0.1, 0, anchors=[(25, 30)])
    assert p[25] == 30


def test_confidence_intervals():
    conf = np.r_[np.ones(20), np.zeros(10), np.ones(20)]
    assert ct.low_confidence_intervals(conf, 0.5, 5) == [(20, 30)]


# --- scale_wraparound ---------------------------------------------------------------------------
def test_wraps_and_edge_extent():
    off = np.r_[np.zeros(5, bool), np.ones(4, bool), np.zeros(3, bool)]
    assert sw.detect_wraps(off, 2) == [(5, 9)]
    rows = np.zeros((3, 200), bool); rows[:, 0:40] = True; rows[:, 150:155] = True
    assert np.all(sw.edge_extent(rows, True, 30, 0) == 39)
    v = sw.unwrap(np.arange(12.0), [(5, 9)], np.full(12, 99.0))
    assert v[5] == 99 and v[4] == 4


def test_hatch_mask_detects_diagonals():
    pan = np.zeros((200, 300), bool)
    for k in range(-200, 120, 10):
        for y in range(200):
            x = y + k
            if 0 <= x < 120:
                pan[y, x] = True
    pan[:, 250] = True  # vertical line is not hatch
    hm = sw.hatch_mask(pan)
    assert hm[:, :120].sum() > 0.5 * pan[:, :120].sum() and hm[:, 240:].sum() == 0


# --- qc_compare ---------------------------------------------------------------------------------
def test_qc_alignment_and_metrics(tmp_path):
    pytest.importorskip("matplotlib")
    d = np.arange(1000, 1500, 0.5); v = np.sin(d / 7) * 30 + 60
    c = qc.compare(d, v, d + 3.0, v, step=0.5, max_shift_ft=10)
    assert abs(abs(c["shift_ft"]) - 3.0) < 0.6 and c["aligned"]["r"] > 0.99
    m = qc.metrics(v, v + 2)
    assert m["bias"] == pytest.approx(2) and m["rmse"] == pytest.approx(2)
    qc.qc_plot(c["grid"], c["ref"], c["test_aligned"], str(tmp_path / "q.png"), "t")
    assert (tmp_path / "q.png").exists()
