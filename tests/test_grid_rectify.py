# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Grid lattice detection, 2-D rectification, TPS, grid-line values and the rectified session path."""
import numpy as np
import pytest

from pylogdigitizer import grid_rectify as gr, grid_values as gv

pytest.importorskip("cv2")


def test_lattice_recovery_and_straightness():
    ink, true, _ = gr.synthetic_grid(seed=0)
    lat = gr.detect_lattice(ink)
    assert lat is not None and lat.X.shape == true.X.shape
    rms = np.sqrt(np.mean((lat.X - true.X) ** 2 + (lat.Y - true.Y) ** 2))
    assert rms < 1.5, rms
    raw = gr.straightness(ink, lat); rect = gr.straightness(gr.rectify_image(ink, lat, ink.shape), lat)
    assert rect["p95"] < 1.5 and rect["p95"] < raw["p95"] / 3, (raw, rect)


@pytest.mark.parametrize("seed", [1, 2])
def test_round_trip(seed):
    ink, true, _ = gr.synthetic_grid(seed=seed)
    rng = np.random.default_rng(seed)
    u = rng.uniform(true.u[0], true.u[-1], 500); v = rng.uniform(true.v[0], true.v[-1], 500)
    x, y = true.to_sheet(u, v)
    u2, v2 = true.to_rectified(x, y)
    assert np.max(np.hypot(u2 - u, v2 - v)) < 0.5


def test_straight_grid_is_identity():
    ink, true, _ = gr.synthetic_grid(amp=0.0, shear=0.0)
    lat = gr.detect_lattice(ink)
    assert lat.residual() < 0.6
    out = gr.rectify_image(ink, lat, ink.shape)
    assert (out != ink).mean() < 0.002


def test_tps_and_node_drag():
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 500, (20, 2)); dst = src + rng.normal(0, 3, src.shape)
    t = gr.tps_fit(src, dst)
    assert np.allclose(gr.tps_eval(t, src), dst, atol=1e-6)
    _, true, _ = gr.synthetic_grid()
    m = true.move_node(5, 5, true.X[5, 5] + 10, true.Y[5, 5], radius=2)
    assert m.X[5, 5] == pytest.approx(true.X[5, 5] + 10)
    assert 0 < m.X[5, 6] - true.X[5, 6] < 10 and m.X[5, 9] == true.X[5, 9]
    d = gr.Lattice.from_dict(m.to_dict()); assert np.allclose(d.X, m.X)


# ----------------------------------------------------------------------------- grid values
LX = np.linspace(10, 410, 11)


def test_linear_values_and_misread_flag():
    f = gv.fit_line_values(LX, [(10, 0), (210, 15), (410, 150), (90, 30)])   # 75 misread as 15
    assert f["kind"] == "linear" and f["values"] == [15.0 * k for k in range(11)]
    assert f["outliers"] == [dict(x=210.0, read=15.0, suggested=75.0)]
    assert any("disagree" in w for w in gv.check_divisions(f))


def test_reversed_and_log():
    f = gv.fit_line_values(LX, [(10, 140), (410, 40)], "linear")
    assert f["values"][0] == 140 and f["values"][-1] == 40 and f["values"][5] == 90
    f = gv.fit_line_values(np.linspace(0, 300, 4), [(0, 0.2), (100, 2), (300, 200)])
    assert f["kind"] == "log" and f["values"] == [0.2, 2.0, 20.0, 200.0]
    k = gv.knot_scale(f)
    assert k(150) == pytest.approx(np.sqrt(2 * 20)) and k.inverse(20) == pytest.approx(200)


def test_set_line_value():
    f = gv.fit_line_values(LX, [(10, 0), (410, 150)])
    g = gv.set_line_value(f, 10, 300)
    assert g["values"][0] == 0 and g["values"][-1] == 300


# ----------------------------------------------------------------------------- rectified session
H, W = 2400, 420


@pytest.fixture(scope="module")
def warped(tmp_path_factory):
    from PIL import Image
    ink, true, warp = gr.synthetic_grid(H=H, W=W, nv=11, nh=25, margin=20, amp=14, shear=0.012, seed=4)
    v = np.arange(0, H, 0.25)
    uc = true.u[0] + (true.u[-1] - true.u[0]) * (0.5 + 0.35 * np.sin(v / 140))
    x, y = warp(uc, v)
    gr._stamp(ink, x, y, 4)
    d = tmp_path_factory.mktemp("warp"); p = d / "warped.tif"
    Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)).save(p)
    val = 150 * 0.5 + 150 * 0.35 * np.sin(v / 140)
    o = np.argsort(y)   # true value at each sheet row (the warp also shifts rows slightly)
    truth = lambda rows: np.interp(np.asarray(rows, float), y[o], val[o])
    return p, true, truth


def _sess(p):
    from pylogdigitizer.assisted_tracing import AssistedSession
    s = AssistedSession(p, curve="GR", auto_layout=False, scale=(0, 150)).open()
    s.set_depth_anchors([(0, 1000), (H, 1240)]); s.set_section(box=(0, W), rows=(0, H))
    return s


def _err(s, truth):
    R = s.result(); m = np.isfinite(R["value"])
    # compare on the rows where the true curve exists away from the sheet ends
    k = m & (R["rows"] > 60) & (R["rows"] < H - 60)
    return float(np.median(np.abs(R["value"][k] - truth(R["rows"][k]))))


def test_session_rectified_trace_beats_raw(warped):
    p, true, truth = warped
    s = _sess(p)
    # raw: two-end scale on the track box (lines 0 and 10 at the box edges +-20 px)
    s.grid_values([(true.u[0], 0), (true.u[-1], 150)]); s.auto_trace(); raw = _err(s, truth)
    info = s.rectify_grid()
    assert len(info["lines"]) == 11
    fit = s.grid_values([(info["lines"][0], 0), (info["lines"][-1], 150)])
    assert fit["values"] == [15.0 * k for k in range(11)] and not fit["warnings"]
    s.auto_trace(); rect = _err(s, truth)
    assert rect < 1.0 and rect < raw / 2, (raw, rect)
    # sheet <-> panel mapping round-trips through the lattice
    r = np.array([500.0, 1500.0]); xp = np.array([100.0, 300.0])
    assert np.allclose(s.sheet_to_panel(r, s.panel_to_sheet(r, xp)), xp, atol=0.05)
    s.unrectify_grid(); assert s.grid_lattice is None


def test_app_grid_routes(warped, tmp_path):
    import shutil, time
    from fastapi.testclient import TestClient
    from pylogdigitizer.app import create_app
    p, true, truth = warped
    d = tmp_path / "example_data" / "WARP_1"; d.mkdir(parents=True); shutil.copy(p, d / "warped.tif")
    app = create_app(str(tmp_path)); app.state.testing = True; c = TestClient(app, base_url="http://127.0.0.1")
    def wait(r):
        assert r.status_code == 200, r.text
        jid = r.json()["job"]["id"]
        while (j := c.get(f"/shell/api/jobs/{jid}").json())["state"] == "running":
            time.sleep(0.05)
        assert j["state"] == "done", j
        return j
    sid = wait(c.post("/app/api/open", json=dict(scan="example_data/WARP_1/warped.tif", well="WARP_1", curve="GR",
                                                  auto_layout=False, settings=dict(scale=[0, 150]))))["result"]["sid"]
    wait(c.post(f"/app/api/{sid}/depth", json={"mode": "labels", "labels": [[0, 1000], [H, 1240]]}))
    wait(c.post(f"/app/api/{sid}/section", json={"box": [0, W], "rows": [0, H]}))
    assert c.get(f"/app/api/{sid}/grid").json()["rectified"] is False
    assert c.post(f"/app/api/{sid}/grid/preview", json={}).status_code == 400
    assert len(wait(c.post(f"/app/api/{sid}/grid/detect", json={}))["result"]["lines"]) == 11
    g = c.get(f"/app/api/{sid}/grid").json(); u = g["lattice"]["u"]
    f = c.post(f"/app/api/{sid}/grid/values", json={"labels": [[u[0], 0], [u[5], 15], [u[-1], 150]]}).json()
    assert f["values"][5] == 75 and f["outliers"][0]["suggested"] == 75 and f["warnings"]
    st = c.get(f"/app/api/{sid}/state").json(); assert st["undo"][-1] == "grid values"
    # header OCR numbers (fraction of track width, value) feed the grid-line values with no clicks
    fr = lambda x: (x + 0.5) / W
    row = dict(id="h1", origin="ocr", track="track1", mnemonic="GR", unit="GAPI", left=0, right=150, type="linear",
               backup_left=None, backup_right=None, points=[[fr(u[0]), 0], [fr(u[5]), 75], [fr(u[-1]), 150]])
    assert c.post(f"/app/api/{sid}/header", json={"rows": [row], "choice": "h1"}).status_code == 200
    f = c.post(f"/app/api/{sid}/grid/values", json={}).json()
    assert f["source"] == "header" and f["values"] == [15.0 * k for k in range(11)] and not f["outliers"]
    assert f["left"] == pytest.approx(150 * (-0.5 - u[0]) / (u[-1] - u[0]), abs=0.05)
    pv = c.post(f"/app/api/{sid}/grid/preview", json={"i": 3, "j": 4, "x": g["lattice"]["X"][3][4] + 5,
                                                     "y": g["lattice"]["Y"][3][4], "rows": [0, 400]}).json()
    assert pv["png"].startswith("data:image/png;base64,") and pv["lattice"]["X"][3][4] == pytest.approx(g["lattice"]["X"][3][4] + 5)
    assert c.get(f"/app/api/{sid}/grid").json()["lattice"]["X"][3][4] == g["lattice"]["X"][3][4]   # preview commits nothing
    wait(c.post(f"/app/api/{sid}/grid/node", json={"i": 3, "j": 4, "x": g["lattice"]["X"][3][4] + 5, "y": g["lattice"]["Y"][3][4]}))
    assert c.get(f"/app/api/{sid}/grid").json()["lattice"]["X"][3][4] == pytest.approx(g["lattice"]["X"][3][4] + 5)
    assert c.post(f"/app/api/{sid}/undo").status_code == 200
    assert c.get(f"/app/api/{sid}/grid").json()["lattice"]["X"][3][4] == pytest.approx(g["lattice"]["X"][3][4])
    assert c.post(f"/app/api/{sid}/grid/off").json()["rectified"] is False
    # Stage E: trace on the straightened, separated layer from the app
    wait(c.post(f"/app/api/{sid}/grid/detect", json={}))
    assert c.get(f"/app/api/{sid}/trace/options").json()["mask_other"] is False
    wait(c.post(f"/app/api/{sid}/trace", json={"mode": "conventional", "force": True, "mask_other": True}))
    assert c.get(f"/app/api/{sid}/trace/options").json()["mask_other"] is True
    st = c.get(f"/app/api/{sid}/state").json(); assert st["undo"][-1].startswith("trace")
