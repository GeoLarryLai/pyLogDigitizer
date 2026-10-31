# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Guided fixing (pins as hard constraints, move/remove/undo, guide strokes, check spots, progress,
session + training data round trip, zero-pick equivalence). Synthetic two-curve scan, fast."""
import json
import numpy as np
import pytest

H, W = 3000, 320
xa = lambda r: 90 + 45 * np.sin(np.asarray(r) / 150)          # the wanted curve
xb = lambda r: 235 + 40 * np.cos(np.asarray(r) / 170)         # distractor


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    import cv2
    from PIL import Image
    from pylogdigitizer.las_export import write_las
    root = tmp_path_factory.mktemp("guided")
    d = root / "example_data" / "SYN_1"; d.mkdir(parents=True)
    img = np.full((H, W), 255, np.uint8); r = np.arange(H)
    cv2.polylines(img, [np.c_[xa(r), r].astype(np.int32).reshape(-1, 1, 2)], False, 0, 6)
    cv2.polylines(img, [np.c_[xb(r), r].astype(np.int32).reshape(-1, 1, 2)], False, 0, 4)
    Image.fromarray(img).save(d / "syn.tif")
    depth = 1000 + np.arange(0, 300, 0.5)
    write_las(d / "syn.las", depth, {"GR": xa((depth - 1000) * 10) / W * 150}, units={"GR": "API"})
    return root


def _session(project):
    from pylogdigitizer.assisted_tracing import AssistedSession
    s = AssistedSession(project / "example_data/SYN_1/syn.tif", curve="GR", well="SYN_1",
                        las_path=project / "example_data/SYN_1/syn.las", auto_layout=False, scale=(0, 150),
                        trace_kw={"straight_len": 0}).open()
    s.set_depth_anchors([(0, 1000), (H, 1300)]); s.set_section(box=(0, W), rows=(0, H)); s.auto_trace()
    return s


def _wrong_row(s):
    """A row where the automatic trace is on the distractor."""
    R = s.result(); bad = np.flatnonzero(np.abs(R["x_panel"] - xa(R["rows"])) > 40)
    assert len(bad), "synthetic trace should be partly wrong"
    return float(R["rows"][bad[len(bad) // 2]])


def test_zero_picks_equal_plain_trace(project):
    s0 = _session(project); s1 = _session(project)
    assert np.array_equal(s0.values, s1.values, equal_nan=True)
    v, x = s1.values.copy(), s1.T["x"].copy()
    s1.pin(_wrong_row(s1), float(xa(_wrong_row(s1))))
    s1.add_stroke([(500, float(xa(500))), (700, float(xa(700)))])
    assert not np.array_equal(s1.T["x"], x)
    assert s1.undo() == "stroke" and s1.undo() == "pin" and s1.undo() is None
    assert np.array_equal(s1.T["x"], x) and np.array_equal(s1.values, v, equal_nan=True)
    assert s1.anchors == [] and s1.strokes == [] and np.array_equal(s1.T["cost"], s1.T["cost0"])
    s1.auto_trace(anchors_panel=None, excluded=None, strokes=None)
    assert np.array_equal(s1.values, s0.values, equal_nan=True)


def test_pin_is_hard_constraint_move_remove(project):
    s = _session(project); r = _wrong_row(s)
    a = s.pin(r, float(xa(r)) + 3)                              # imprecise click, snapped to the ink
    k = int(np.argmin(np.abs(s.T["rows"] - a[0])))
    assert abs(s.T["x"][k] - a[1]) <= 1.5 * s.T["fx"]           # trace passes through the pin
    assert abs(a[1] - xa(r)) < 6
    e = s.progress()["last"]; assert e["action"] == "pin" and e["changed_ft"] > 0
    r2 = r + 150
    b = s.move_pin(0, r2, float(xa(r2)))
    assert len(s.anchors) == 1 and abs(s.anchors[0][0] - r2) < 1
    k = int(np.argmin(np.abs(s.T["rows"] - b[0]))); assert abs(s.T["x"][k] - b[1]) <= 1.5 * s.T["fx"]
    s.unpin(0); assert s.anchors == []
    assert s.undo() == "unpin" and len(s.anchors) == 1 and s.undo() == "move" and abs(s.anchors[0][0] - a[0]) < 1
    with pytest.raises(Exception):
        s.unpin(5)
    assert len(s.anchors) == 1                                  # a failed edit changes nothing


def test_guide_stroke_pulls_trace(project):
    s = _session(project); r = _wrong_row(s)
    pts = [(rr, float(xa(rr))) for rr in np.linspace(r - 60, r + 60, 7)]
    s.add_stroke(pts)
    R = s.result(); m = (R["rows"] >= r - 50) & (R["rows"] <= r + 50)
    assert np.abs(R["x_panel"][m] - xa(R["rows"][m])).max() < 10
    assert s.overlay()["strokes"] and len(s.overlay()["strokes"][0]) == 7
    s.remove_stroke(0); assert s.strokes == []


def test_check_spots_and_answers(project):
    s = _session(project); sp = s.check_spots()
    assert sp and all(c["x_panel"] is not None for p in sp for c in p["choices"])
    cross = [p for p in sp if p["kind"] == "crossover"]
    assert cross and cross[0]["question"] == "Does the curve go left or right here?"
    p = cross[0]; want = min(p["choices"], key=lambda c: abs(c["x_panel"] - xa(p["row"])))
    s.answer_spot(p["id"], want["id"])
    assert s.decisions[-1]["choice"] == want["id"] and len(s.anchors) == 1
    assert all(not (q["k0"] <= p["k"] <= q["k1"]) for q in s.check_spots())      # not asked again
    left = s.check_spots()
    if left:
        s.answer_spot(left[0]["id"], "skip"); assert s.decisions[-1]["choice"] == "skip"
    assert np.asarray(s.disagreement).shape == s.T["x"].shape


def test_progress_settles(project):
    s = _session(project); assert s.progress()["last"] is None
    r = _wrong_row(s); s.pin(r, float(xa(r)))
    k = int(np.argmin(np.abs(s.T["rows"] - (r + 30))))
    s.pin(float(s.T["rows"][k]), float(s.T["x"][k]), panel_x=True)          # clicks on the trace itself:
    k = int(np.argmin(np.abs(s.T["rows"] - (r - 30))))
    s.pin(float(s.T["rows"][k]), float(s.T["x"][k]), panel_x=True)          # nothing changes
    k = int(np.argmin(np.abs(s.T["rows"] - (r + 60))))
    s.pin(float(s.T["rows"][k]), float(s.T["x"][k]), panel_x=True)
    p = s.progress(); assert p["settled"] and p["last"]["changed_ft"] <= p["small_ft"]


def test_session_and_training_keep_picks(project, tmp_path):
    from pylogdigitizer.assisted_tracing import AssistedSession
    s = _session(project); r = _wrong_row(s)
    sp = s.check_spots()[0]; s.answer_spot(sp["id"], "skip")
    s.pin(r, float(xa(r))); s.add_stroke([(r + 300, float(xa(r + 300))), (r + 400, float(xa(r + 400)))])
    p = s.save_session(tmp_path / "s.json"); d = json.loads(p.read_text())
    assert d["version"] == 2 and len(d["strokes"]) == 1 and d["decisions"][0]["choice"] == "skip"
    s2 = AssistedSession.from_session_file(p)
    assert s2.anchors == s.anchors and s2.strokes == s.strokes and s2.decisions == s.decisions
    # reload re-traces in one pass; incremental local re-traces may differ by one DP column on a few rows
    d = np.abs(s2.T["x"] - s.T["x"]); assert d.max() <= s.T["fx"] and (d > 0).mean() < 0.03
    out = s.save_training(tmp_path / "corr")
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["picks"] == [list(a) for a in s.anchors] and meta["n_strokes"] == 1 and meta["decisions"][0]["choice"] == "skip"
