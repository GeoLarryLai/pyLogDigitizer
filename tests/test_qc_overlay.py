# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Overlay QA on the original scan: exact strip->sheet inverse of the deskew, grid lines snapped to
the printed (bent) lines in every track, trace drawn at its printed ink, nulls skipped."""
import numpy as np
from types import SimpleNamespace
from PIL import Image
from pylogdigitizer import qc_overlay as qo

H, W = 1200, 420
bend = lambda r: 3.0 * np.sin(np.asarray(r) / 180.0)          # paper warp: grid lines wander +-3 px


def _sheet(tmp_path):
    g = np.full((H, W), 255, np.uint8); r = np.arange(H)
    for x in (20, 400):                                          # track edges (heavy, straight)
        g[:, x - 1:x + 2] = 0
    for k in range(1, 10):                                       # bent vertical grid in the track
        xc = 20 + 38 * k + bend(r)
        for y in r:
            g[y, int(round(xc[y])) - 1:int(round(xc[y])) + 1] = 0
    for y in range(40, H, 40):                                   # horizontal grid
        g[y:y + 2, 20:401] = 0
    xa = 120 + 60 * np.sin(r / 90.0)                             # the curve
    for y in r:
        g[y, int(xa[y]) - 2:int(xa[y]) + 2] = 0
    p = tmp_path / "sheet.png"; Image.fromarray(g).save(p)
    ink = g < 128
    L = SimpleNamespace(ink=ink, x_offset=0, y_offset=0, skew_deg=np.array([0.0]), centers=np.array([0.0, H]),
                        edges=np.array([[20.0, 400.0], [20.0, 400.0]]), edge_names=["track1.left", "track1.right"],
                        track_names=["track1"])
    L.width_of = lambda n: 380
    from pylogdigitizer import track_grid_detection as tgd
    L.panel = lambda n, rows, width=None: tgd.rectify(ink, rows, L.centers, L.edges[:, 0], L.edges[:, 1], width or 380)
    rows = np.arange(0, H, 2).astype(float)
    x_pan = (xa[rows.astype(int)] - 20) / 380 * 380 - 0.5
    T = dict(rows=rows, x=x_pan)
    R = dict(layout=L, trace=T, panel=np.zeros((H, 380), bool), panel_rows=(0, H), value=np.full(len(rows), 50.0))
    return p, R, xa


def test_strip_to_sheet_inverts_deskew():
    from pylogdigitizer import scan_cleanup as scl
    rng = np.random.default_rng(0)
    sheet = np.zeros((3000, 800), bool)
    for y in range(100, 3000, 60):                               # skewed lines (0.4 deg)
        xs = np.arange(800); ys = np.round(y + (xs - 400) * np.tan(np.deg2rad(0.4))).astype(int)
        ok = (ys >= 0) & (ys < 3000); sheet[ys[ok], xs[ok]] = True
    strip, ang = scl.deskew_blocks(sheet, 2000)
    L = SimpleNamespace(ink=strip, x_offset=0, y_offset=0, skew_deg=ang)
    ys, xs = np.nonzero(strip)
    X, Y = qo.strip_to_sheet(L, xs, ys)
    ok = (Y >= 0) & (Y < 3000)
    assert sheet[np.round(Y[ok]).astype(int), np.round(X[ok]).astype(int)].mean() > 0.99


def test_grid_snaps_to_bent_lines_and_trace_sits_on_ink(tmp_path):
    p, R, xa = _sheet(tmp_path)
    img, res = qo.overlay_scan(p, R, "track1", tmp_path / "o.png", side_panel=False, max_height=10 ** 6)
    assert (tmp_path / "o.png").exists()
    r = res["track1"]
    assert r["n_v"] >= 8 and r["n_h"] >= 25
    assert r["v_after"]["median"] <= 0.5 and r["v_after"]["p95"] <= 1.5          # nearly perfect
    assert r["v_before"]["p95"] > r["v_after"]["p95"]                             # snapping helped
    band = img.shape[0] - H                                                      # legend band on top
    y = 600; x = int(round(xa[y]))
    red = (np.abs(img[band + y, x - 4:x + 5].astype(int) - qo.CURVE_C[0]).sum(-1) < 120).any()
    assert red


def test_null_samples_not_drawn(tmp_path):
    p, R, xa = _sheet(tmp_path)
    R["value"][200:400] = np.nan                                                 # rows 400-800 null
    img, _ = qo.overlay_scan(p, R, "track1", side_panel=False, max_height=10 ** 6)
    band = img.shape[0] - H; y = 600; x = int(round(xa[y]))
    assert not (np.abs(img[band + y, x - 4:x + 5].astype(int) - qo.CURVE_C[0]).sum(-1) < 60).any()
