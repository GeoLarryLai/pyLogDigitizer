# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Regression tests for the 0.3.0 audit items H1-H9 (synthetic arrays, fast)."""
import numpy as np
import pytest

from pylogdigitizer import curve_tracing as ct


def _sine_panel(H=1500, W=300, width=5):
    import cv2
    img = np.zeros((H, W), np.uint8); r = np.arange(H)
    x = 150 + 80 * np.sin(r / 120.0)
    cv2.polylines(img, [np.c_[x, r].astype(np.int32).reshape(-1, 1, 2)], False, 1, width)
    return img.astype(bool), x


def test_h1_centreline_switch_stored_and_honoured():
    panel, _ = _sine_panel()
    T = ct.trace_track(panel, 0, px_per_ft=10, straight_len=0, min_width=3)
    assert T["params"]["centreline"] is True
    R = ct.trace_track(panel, 0, px_per_ft=10, straight_len=0, min_width=3, centreline=False)
    assert R["params"]["centreline"] is False
    assert np.allclose(R["x"], R["path"] * R["fx"] + R["fx"] / 2)       # the 0.2.x output


def test_h1_assisted_retrace_keeps_the_switch(tmp_path):
    from PIL import Image
    from pylogdigitizer.assisted_tracing import AssistedSession
    panel, x = _sine_panel(H=1200, W=320)
    Image.fromarray(np.where(panel, 0, 255).astype(np.uint8)).save(tmp_path / "s.tif")
    s = AssistedSession(tmp_path / "s.tif", curve="GR", well="S", auto_layout=False, scale=(0, 150),
                        trace_kw={"straight_len": 0, "centreline": False}).open()
    s.set_depth_anchors([(0, 1000), (1200, 1120)]); s.set_section(box=(0, 320), rows=(0, 1200)); s.auto_trace()
    s.pin(600, float(x[600]))
    T = s.T
    assert T["params"]["centreline"] is False
    k = np.arange(len(T["x"])) != s._dp(600)
    assert np.allclose(T["x"][k], (T["path"] * T["fx"] + T["fx"] / 2)[k])


def test_h2_flat_dash_ends_on_their_rows():
    """Sloped dashes (4 px per row): the joined centreline has no half-row offset."""
    h, w = 64, 400
    cost = np.ones((h, w), np.float32); P = np.zeros(h, int)
    for k in range(h):
        c = 4 * k + 20; P[k] = c + 1
        if k % 7 < 4:
            cost[k, c:c + 4] = 0.0
    x, _ = ct.ink_centreline(cost, P, 0, 1, 1)
    e = (x - (4 * np.arange(h) + 22.0))[3:-3]
    assert abs(e.mean()) < 0.2 and np.abs(e).max() < 0.6


def test_h3_pins_set_x_without_ink():
    h, w = 60, 100
    cost = np.ones((h, w), np.float32); cost[:20, 40:44] = 0; cost[40:, 40:44] = 0
    P = np.full(h, 41)
    x, on = ct.ink_centreline(cost, P, 0, 2, 1)
    assert abs(x[30] - 84.0) < 1e-6                         # a gap: joined straight
    x, on = ct.ink_centreline(cost, P, 0, 2, 1, keep={30: 120.0})
    assert x[30] == 120.0 and on[30]                         # the pin sets x, no ink needed
    assert 84.0 < x[25] < 120.0 and 84.0 < x[35] < 120.0     # neighbours join through it
    T = ct.finish_trace(np.pad(cost, ((0, 0), (1, 1)), constant_values=0.8), P + 1, 0, 1, 2, 1, keep={30: 120.0})
    assert T["x"][30] == 120.0


def test_h4_line_neutral_forwarded_and_center_weight_checked():
    h, w = 40, 60
    cost = np.ones((h, w), np.float32)
    cost[:10, 10] = 0; cost[10:30, 10] = 0.8; cost[30:, 12] = 0
    P = np.r_[np.full(30, 10), np.full(10, 12)]
    keep_dp, _ = ct.ink_centreline(cost, P, 0, 1, 1, line_neutral=0.8)
    joined, _ = ct.ink_centreline(cost, P, 0, 1, 1)              # 0.85: the neutral run is not recognised
    assert np.allclose(keep_dp[10:30], 10.5) and not np.allclose(joined[10:30], 10.5)
    panel, _ = _sine_panel(H=600)
    T = ct.trace_track(panel, 0, px_per_ft=10, straight_len=0, min_width=3, line_neutral=0.8)
    assert T["params"]["line_neutral"] == 0.8
    with pytest.raises(ValueError):
        ct.trace_track(panel, 0, px_per_ft=10, center_weight=0.5)


def test_h5_centre_is_geometric_unless_plain_centre_weight():
    h, w = 10, 40
    cost = np.ones((h, w), np.float32)
    cost[:, 10:20] = np.linspace(0.05, 0.45, 10)[::-1]        # cheapest cell at the right edge
    P = np.full(h, 15)
    xg, _ = ct.ink_centreline(cost, P, 0, 1, 1, min_cost_centre=False)
    xm, _ = ct.ink_centreline(cost, P, 0, 1, 1, min_cost_centre=True)
    assert np.allclose(xg, 15.0) and np.allclose(xm, 19.5)
    panel, _ = _sine_panel(H=600)
    assert ct.trace_track(panel, 0, px_per_ft=10, min_width=3, center_weight=0.3)["params"]["min_cost_centre"]
    assert not ct.trace_track(panel, 0, px_per_ft=10, min_width=3)["params"]["min_cost_centre"]


def test_h6_thin_layout_may_only_add_splits():
    from pylogdigitizer import track_grid_detection as tgd
    base = [dict(left=50, right=400, kind="track"), dict(left=400, right=500, kind="depth"),
            dict(left=500, right=850, kind="track")]
    split = base[:2] + [dict(left=500, right=670, kind="track"), dict(left=674, right=850, kind="track")]
    moved = [dict(left=50, right=380, kind="track"), dict(left=380, right=500, kind="depth")] + base[2:]
    assert tgd.only_adds_splits(base, split) and not tgd.only_adds_splits(base, moved)
    assert not tgd.only_adds_splits(base, [])                    # nothing found is not a refinement
    assert not tgd.only_adds_splits(base, base[:2])               # a dropped edge


def test_h7_left_right_candidates_never_cross():
    import cv2
    H, W = 2000, 400
    r = np.arange(H)
    img = np.zeros((H, W), np.uint8)
    for x in (170 + 60 * np.sin(r / 150.0), 215 + 60 * np.sin(r / 150.0 + 0.3)):
        for a in range(0, H, 24):                                 # dashes 16 rows long, gaps 8
            seg = np.c_[x[a:a + 16], r[a:a + 16]].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [seg], False, 1, 4)
    T = ct.trace_multi(img.astype(bool), 0, n_curves=2, pick="left", px_per_ft=10, straight_len=0,
                       min_width=3, min_area=0)
    A, B = T["candidates"]
    on = ~(A["off_left"] | A["off_right"] | B["off_left"] | B["off_right"])
    assert T["picked"] == 0 and on.mean() > 0.9 and (A["x"][on] <= B["x"][on]).all()


def test_h8_track_styles_names_are_unique():
    from pylogdigitizer import line_styles as ls
    from pylogdigitizer.ml_training_data.synth_scan import render_style_track
    H, W = 3000, 700
    r = np.arange(H)
    xs = [180 + 80 * np.sin(r / 170.0), 480 + 80 * np.sin(r / 230.0 + 1)]
    trk = render_style_track(xs, ["solid", "solid"], [4, 5], W)
    S = ls.suggest_line_styles(trk, {"track1": (0, W)}, None, margin=6)
    names = [s["name"] for s in S["styles"]["track1"]]
    assert len(names) == len(set(names)) and len(S["names"]) == len(set(S["names"]))


def test_h9_line_header_names_are_unique():
    from pylogdigitizer.header_ocr import propose_curves_from_lines
    w = lambda t, x0, y0: dict(text=t, x0=x0, y0=y0, x1=x0 + 50, y1=y0 + 14)
    tracks = {"track1": dict(x=(0, 500), words=[
        w("GR", 20, 20), w("0", 20, 42), w("GAPI", 210, 42), w("150", 360, 42),
        w("GR", 20, 70), w("0", 20, 92), w("GAPI", 210, 92), w("150", 360, 92),
        w("GR", 20, 120), w("0", 20, 142), w("GAPI", 210, 142), w("150", 360, 142)])}
    names = [r_["name"] for r_ in propose_curves_from_lines(tracks)]
    assert len(names) == 3 and len(set(names)) == 3
