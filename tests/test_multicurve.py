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
    cal = 130 + 60 * np.sin(v / 90)
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


def test_app_candidate_picker(tmp_path):
    """Trace step with n_curves=2: the overlay carries both candidates, a pick re-traces with the
    other curve, undo restores the previous pick."""
    import time
    from PIL import Image
    from fastapi.testclient import TestClient
    from pylogdigitizer.app import create_app
    ink, v, g, cal = _two_curves(H=1600)
    d = tmp_path / "example_data" / "TWO_1"; d.mkdir(parents=True)
    Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)).save(d / "two.tif")
    H, W = ink.shape
    app = create_app(str(tmp_path)); app.state.testing = True; c = TestClient(app)
    def wait(r):
        assert r.status_code == 200, r.text
        jid = r.json()["job"]["id"]
        while (j := c.get(f"/shell/api/jobs/{jid}").json())["state"] == "running":
            time.sleep(0.05)
        assert j["state"] == "done", j
        return j
    sid = wait(c.post("/app/api/open", json=dict(scan="example_data/TWO_1/two.tif", well="TWO_1", curve="GR",
                                                  auto_layout=False, settings=dict(scale=[0, 150]))))["result"]["sid"]
    wait(c.post(f"/app/api/{sid}/depth", json={"mode": "labels", "labels": [[0, 1000], [H, 1160]]}))
    wait(c.post(f"/app/api/{sid}/section", json={"box": [0, W], "rows": [0, H]}))
    assert c.post(f"/app/api/{sid}/trace/pick", json={"index": 1}).status_code == 400      # no candidates yet
    assert c.post(f"/app/api/{sid}/trace", json={"n_curves": 9}).status_code == 400
    wait(c.post(f"/app/api/{sid}/trace", json={"mode": "conventional", "force": True, "n_curves": 2}))
    o = c.get(f"/app/api/{sid}/trace/options").json()
    assert o["n_curves"] == 2 and o["candidates"] == 2
    ov = c.get(f"/core/api/sessions/{sid}/overlay").json()
    cands = ov["candidates"]; first = o["picked"]; other = 1 - first
    assert len(cands) == 2 and cands[first]["picked"] and len(cands[other]["trace"]) > 100
    assert c.post(f"/app/api/{sid}/trace/pick", json={"index": 5}).status_code == 400
    wait(c.post(f"/app/api/{sid}/trace/pick", json={"index": other}))
    assert c.get(f"/app/api/{sid}/trace/options").json()["picked"] == other
    ov2 = c.get(f"/core/api/sessions/{sid}/overlay").json()
    xs = lambda t: np.array([p[1] for p in t])
    assert np.median(np.abs(xs(ov2["trace"])[:len(cands[other]["trace"])] - xs(cands[other]["trace"]))) < 3
    st = c.get(f"/app/api/{sid}/state").json(); assert st["undo"][-1] == f"pick curve {other + 1}"
    assert c.post(f"/app/api/{sid}/undo").status_code == 200
    assert c.get(f"/app/api/{sid}/trace/options").json()["picked"] == first


def test_auto_pick_score_prefers_track_filling_detailed_curve():
    base = dict(on_track=0.98, on_ink=0.95)
    gr_like = ct.auto_pick_score(dict(base, spread=0.6, rough=0.02, flat=0.15))
    caliper = ct.auto_pick_score(dict(base, spread=0.1, rough=0.003, flat=0.8))
    parked = ct.auto_pick_score(dict(base, on_track=0.3, spread=0.6, rough=0.02, flat=0.15))
    assert gr_like > 3 * caliper and gr_like > 2 * parked
