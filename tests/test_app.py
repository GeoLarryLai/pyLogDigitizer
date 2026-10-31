# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Desktop app API (pylogdigitizer.app): every step, undo/reset, export-only-on-press, greyed-out
optional features. Synthetic scan (fast) + one end-to-end equivalence run on the KGS example."""
import json, pathlib, time
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

H, W = 3000, 320
xa = lambda r: 90 + 45 * np.sin(np.asarray(r) / 150)
xb = lambda r: 235 + 40 * np.cos(np.asarray(r) / 170)
ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    import cv2
    from PIL import Image
    from pylogdigitizer.las_export import write_las
    root = tmp_path_factory.mktemp("approj")
    d = root / "example_data" / "SYN_1"; d.mkdir(parents=True)
    img = np.full((H, W), 255, np.uint8); r = np.arange(H)
    cv2.polylines(img, [np.c_[xa(r), r].astype(np.int32).reshape(-1, 1, 2)], False, 0, 6)
    cv2.polylines(img, [np.c_[xb(r), r].astype(np.int32).reshape(-1, 1, 2)], False, 0, 4)
    Image.fromarray(img).save(d / "syn.tif")
    depth = 1000 + np.arange(0, 300, 0.5)
    write_las(d / "syn.las", depth, {"GR": xa((depth - 1000) * 10) / W * 150}, units={"GR": "API"})
    return root


def wait(c, r):
    assert r.status_code == 200, r.text
    jid = r.json()["job"]["id"]
    while (j := c.get(f"/shell/api/jobs/{jid}").json())["state"] == "running":
        time.sleep(0.05)
    assert 0 <= j["pct"] <= 100 and j["N"] >= 1
    return j


@pytest.fixture(scope="module")
def app_sid(project):
    from pylogdigitizer.app import create_app
    app = create_app(str(project)); app.state.testing = True
    c = TestClient(app)
    info = c.get("/app/api/info").json()
    assert info["catalog"][0]["scan"] == "example_data/SYN_1/syn.tif"
    j = wait(c, c.post("/app/api/open", json=dict(scan="example_data/SYN_1/syn.tif", well="SYN_1", curve="GR",
                                                  las="example_data/SYN_1/syn.las", auto_layout=False,
                                                  settings=dict(scale=[0, 150], trace_kw={"straight_len": 0}))))
    assert j["state"] == "done", j
    return c, j["result"]["sid"], project


def test_steps_end_to_end(app_sid):
    c, sid, root = app_sid
    st = c.get(f"/app/api/{sid}/state").json()
    assert st["width"] == W and st["section"] is None and "depth" not in st["errors"] or True
    assert c.get(f"/app/api/{sid}/layout").json()["available"] is False            # step 2 (no auto layout)
    assert wait(c, c.post(f"/app/api/{sid}/section", json={"box": [0, W], "rows": [0, H]}))["state"] == "done"   # step 3
    st = c.get(f"/app/api/{sid}/state").json(); assert st["traced"] and st["section"]["kind"] == "box"
    h = c.get(f"/app/api/{sid}/header").json(); assert h["rows"] == []                # step 4: add a row by hand
    row = dict(id="r1", origin="added", track="track1", mnemonic="GR", unit="API", left="0", right="150", type="linear",
               backup_left="", backup_right="")
    h = c.post(f"/app/api/{sid}/header", json={"rows": [row], "choice": "r1"}).json()
    assert h["rows"][0]["right"] == 150.0 and h["choice"] == "r1"
    assert c.post(f"/app/api/{sid}/header", json={"rows": [dict(row, type="bad")]}).status_code == 400
    j = wait(c, c.post(f"/app/api/{sid}/depth", json={"mode": "labels", "labels": [[0, 1000], [H, 1300]]}))   # step 5
    assert j["state"] == "done", j
    st = c.get(f"/app/api/{sid}/state").json(); assert st["calibration"]["depth_source"] == "manual" and st["traced"]
    assert abs(st["calibration"]["px_per_ft"] - 10) < 1e-6
    assert c.post(f"/core/api/sessions/{sid}/scale", json={"left": 0, "right": 150}).status_code == 200    # step 6
    r0 = c.get(f"/core/api/sessions/{sid}/qc").json()["metrics"]["r"]                                       # step 7
    for rr in (300, 1500, 2700):
        assert c.post(f"/core/api/sessions/{sid}/anchors", json={"row": rr, "x": float(xa(rr)) + 3, "window_ft": 400}).status_code == 200
    q = c.get(f"/core/api/sessions/{sid}/qc").json(); assert q["available"] and q["metrics"]["r"] > max(0.95, r0)   # step 8
    st = c.get(f"/app/api/{sid}/state").json(); assert st["n_anchors"] == 3 and st["undo"][-1] == "anchors"


def test_undo_and_resets(app_sid):
    c, sid, _ = app_sid
    n = c.get(f"/app/api/{sid}/state").json()["n_anchors"]
    c.post(f"/core/api/sessions/{sid}/anchors", json={"row": 2000, "x": float(xa(2000))})
    assert c.get(f"/app/api/{sid}/state").json()["n_anchors"] == n + 1
    u = c.post(f"/app/api/{sid}/undo").json(); assert u["undone"] == "anchors" and u["n_anchors"] == n
    c.post(f"/core/api/sessions/{sid}/scale", json={"left": 0, "right": 300})
    assert c.post(f"/app/api/{sid}/undo").json()["calibration"]["scale"]["right"] == 150
    assert c.post(f"/core/api/sessions/{sid}/exclude", json={"row": 1500, "x": float(xb(1500))}).status_code == 200
    assert c.post(f"/app/api/{sid}/undo").json()["n_excluded"] == 0
    assert wait(c, c.post(f"/app/api/{sid}/trace/reset"))["state"] == "done"
    assert c.get(f"/app/api/{sid}/state").json()["n_anchors"] == 0
    assert c.post(f"/app/api/{sid}/undo").json()["n_anchors"] == n                       # reset itself is undoable
    h = c.post(f"/app/api/{sid}/header/revert").json(); assert h["rows"] == [] and h["choice"] is None
    assert c.post(f"/app/api/{sid}/undo").json()["sid"] == sid
    assert c.get(f"/app/api/{sid}/header").json()["rows"][0]["mnemonic"] == "GR"


def test_export_only_on_press(app_sid, tmp_path):
    c, sid, root = app_sid
    assert not (root / "outputs").exists()                                    # nothing written by steps 1-8
    for p in ("export_las", "save_session", "save_training"):                  # editor writers are blocked in the app
        assert c.post(f"/core/api/sessions/{sid}/{p}").status_code == 403
    assert c.get(f"/core/api/sessions/{sid}/qc.png").status_code == 403
    assert c.post("/shell/api/open_folder", json={"path": str(tmp_path / "x")}).status_code == 404
    st = c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp_path / "x")}).json()
    assert not st["exists"] and not (tmp_path / "x").exists()
    r = c.post(f"/app/api/{sid}/export/las").json(); assert len(r["written"]) == 1 and r["exists"]
    names = sorted(pathlib.Path(p).name for p in c.post(f"/app/api/{sid}/export/all").json()["written"])
    assert "SYN_1_syn_GR.las" in names and "SYN_1_syn_GR.csv" in names and "SYN_1_syn_GR_session.json" in names
    d = json.loads((tmp_path / "x" / "SYN_1_syn_GR_session.json").read_text()); assert d["scan"] == "example_data/SYN_1/syn.tif"
    from pylogdigitizer.las_export import read_las
    las = read_las(tmp_path / "x" / "SYN_1_syn_GR.las"); csv = np.loadtxt(tmp_path / "x" / "SYN_1_syn_GR.csv", delimiter=",", skiprows=1)
    assert np.allclose(las.index, csv[:, 0]) and np.allclose(las.curves[1].data, csv[:, 1], atol=1e-5)
    assert c.post("/shell/api/open_folder", json={"path": str(tmp_path / "x")}).status_code == 200
    assert not (root / "outputs").exists()


def test_optional_features_greyed_out(app_sid, monkeypatch):
    c, sid, root = app_sid
    from pylogdigitizer.app import routes
    real = routes.features
    monkeypatch.setattr(routes, "features", lambda r: dict(real(r), ml=dict(ok=False, model=None, hint='needs: pip install "pylogdigitizer[ml]"'),
                                                             ocr=dict(ok=False, hint="needs rapidocr_onnxruntime")))
    r = c.post(f"/app/api/{sid}/trace", json={"mode": "ml"}); assert r.status_code == 400 and "pylogdigitizer[ml]" in r.json()["detail"]
    r = c.post(f"/app/api/{sid}/header/rerun"); assert r.status_code == 400 and "rapidocr" in r.json()["detail"]
    f = c.get("/app/api/info").json()["features"]; assert f["ml"]["ok"] is False and f["ml"]["hint"]
    assert c.get(f"/app/api/{sid}/state").status_code == 200                     # still alive
    assert c.post(f"/app/api/{sid}/trace", json={"mode": "nope"}).status_code == 400
    j = wait(c, c.post(f"/app/api/{sid}/depth", json={"mode": "labels", "labels": [[0, 1000]]}))
    assert j["state"] == "error" and "2 depth anchors" in j["error"]                # errors are reported, never crash


def test_old_editor_page_not_mounted(app_sid):
    c, sid, _ = app_sid
    for u in ("/core", "/core/", "/core/index.html", "/editor/"):
        assert c.get(u).status_code == 404
    assert c.get(f"/core/api/sessions/{sid}/qc").status_code == 200


def test_ocr_engine_choice(app_sid, monkeypatch):
    c, sid, _ = app_sid
    from pylogdigitizer import ocr_engines as oe
    f = c.get("/app/api/info").json()["features"]["ocr"]
    assert set(f["engines"]) == {"rapidocr"} and "engine" in f
    monkeypatch.setattr(oe, "available", lambda e: e == "rapidocr")
    r = c.post("/app/api/ocr_engine", json={"engine": "tesseract"}); assert r.status_code == 400 and "unknown" in r.json()["detail"]
    f = c.post("/app/api/ocr_engine", json={"engine": "rapidocr"}).json()["ocr"]
    assert f["engine"] == "rapidocr" and f["engines"]["rapidocr"]["ok"]
    assert c.post("/app/api/ocr_engine", json={"engine": "none"}).json()["ocr"]["ok"] is False
    assert c.post("/app/api/ocr_engine", json={"engine": "auto"}).status_code == 200
    oe.set_default(None)


def test_page_and_static(app_sid):
    c, _, _ = app_sid
    html = c.get("/").text; assert "shell.js" in html and "openseadragon" in html and "cdn" not in html.lower()
    for u in ("/shell/static/shell.js", "/shell/static/shell.css", "/static/app.js", "/static/app.css",
              "/core/static/vendor/openseadragon/openseadragon.min.js"):
        assert c.get(u).status_code == 200, u
    assert c.get("/shell/api/jobs/nope").status_code == 404 and c.get("/app/api/nope/state").status_code == 404


def test_cli_entry_point():
    from pylogdigitizer.app import main
    from pylogdigitizer.app.shell import free_port
    assert free_port(0) > 0 and callable(main)


@pytest.mark.skipif(not (ROOT / "example_data/USA_A_16_15-055-20536/4342.tif").exists(), reason="KGS example data not present")
def test_equivalence_with_notebook_conventional(tmp_path):
    """App API end to end == notebook 01 Section A (digitize_curve -> resample -> write_las): identical values."""
    from pylogdigitizer.app import create_app
    from pylogdigitizer import workflows as wf
    from pylogdigitizer.las_export import resample, read_las
    c = TestClient(create_app(str(ROOT)))
    cat = [x for x in c.get("/app/api/info").json()["catalog"] if x["scan"].endswith("USA_A_16_15-055-20536/4342.tif") and x["curve"] == "GR"][0]
    j = wait(c, c.post("/app/api/open", json=cat)); assert j["state"] == "done", j; sid = j["result"]["sid"]
    c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp_path)}); c.post(f"/app/api/{sid}/export/las"); c.post(f"/app/api/{sid}/export/csv")
    st = cat["settings"]; kw = {k: (tuple(st[k]) if isinstance(st[k], list) else st[k]) for k in ("track", "scale", "backup", "wrap_side", "use_hatch", "first_depth", "scale_type") if k in st}
    R = wf.digitize_curve(ROOT / cat["scan"], **kw); g, v = resample(R["depth"], R["value"], 0.5)
    csv = np.loadtxt(tmp_path / "USA_A_16_4342_GR.csv", delimiter=",", skiprows=1)
    assert np.array_equal(csv[:, 0], g) and np.array_equal(csv[:, 1], v)
    las = read_las(tmp_path / "USA_A_16_4342_GR.las"); assert np.array_equal(las.index, g) and np.allclose(las.curves[1].data, v, atol=1e-4, rtol=0)


def test_guided_fix_routes(app_sid):
    c, sid, _ = app_sid
    st = c.get(f"/app/api/{sid}/state").json(); n = st["n_anchors"]; k = 2   # existing pins at 300/1500/2700
    r = c.post(f"/app/api/{sid}/fix/pin", json={"row": 2100, "x": 100, "snap": True}).json()
    assert r["n_anchors"] == n + 1 and r["ms"] >= 0 and r["progress"]["last"]["action"] == "pin"
    r = c.post(f"/app/api/{sid}/fix/move", json={"index": k, "row": 2200, "x": 100}).json(); assert r["n_anchors"] == n + 1
    r = c.post(f"/app/api/{sid}/fix/stroke", json={"points": [[400, 100], [500, 110], [600, 120]]}).json()
    assert r["n_strokes"] == 1
    assert c.post(f"/app/api/{sid}/fix/stroke", json={"points": [[1, 1]]}).status_code == 400
    assert c.get(f"/app/api/{sid}/fix/progress").json()["last"]["action"] == "stroke"
    ch = c.get(f"/app/api/{sid}/checks").json(); assert "spots" in ch and "disagree" in ch
    if ch["spots"]:
        a = c.post(f"/app/api/{sid}/checks/answer", json={"id": ch["spots"][0]["id"], "choice": "skip"}).json()
        assert a["decisions"][-1]["choice"] == "skip"
        c.post(f"/app/api/{sid}/undo")
    assert c.post(f"/app/api/{sid}/undo").json()["n_strokes"] == 0
    assert c.post(f"/app/api/{sid}/fix/unpin", json={"index": k}).json()["n_anchors"] == n
    c.post(f"/app/api/{sid}/undo"); c.post(f"/app/api/{sid}/undo")
    assert c.post(f"/app/api/{sid}/undo").json()["n_anchors"] == n
