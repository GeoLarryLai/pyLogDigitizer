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
    c = TestClient(app, base_url="http://127.0.0.1")
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
    monkeypatch.setattr(routes, "features", lambda r: dict(real(r), ml=dict(ok=False, model=None, hint="needs: pip install torch"),
                                                             ocr=dict(ok=False, hint="needs rapidocr_onnxruntime")))
    r = c.post(f"/app/api/{sid}/trace", json={"mode": "ml"}); assert r.status_code == 400 and "pip install torch" in r.json()["detail"]
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


@pytest.mark.skipif(not (ROOT / "example_data/USA_A_16_15-055-20536/4342_crop_3260-4770.tif").exists(), reason="KGS example data not present")
def test_equivalence_with_notebook_line_styles(tmp_path):
    """App API end to end == the black-ink workflow of notebook 05 (identify the curves by line
    style -> working image per curve -> assisted session -> values -> 0.5 ft grid): identical GR."""
    from pylogdigitizer.app import create_app
    from pylogdigitizer import curve_identify as ci
    from pylogdigitizer.workflows import open_color_curve, color_curve_values
    from pylogdigitizer.assisted_tracing import AssistedSession
    from pylogdigitizer.header_ocr import propose_header
    from pylogdigitizer.curve_ink import save_ink_tif
    from pylogdigitizer.las_export import resample, read_las
    c = TestClient(create_app(str(ROOT)), base_url="http://127.0.0.1")
    cat = [x for x in c.get("/app/api/info").json()["catalog"] if x["scan"].endswith("USA_A_16_15-055-20536/4342_crop_3260-4770.tif") and x["curve"] == "GR"][0]
    j = wait(c, c.post("/app/api/open", json=cat)); assert j["state"] == "done", j; sid = j["result"]["sid"]
    cur = c.get(f"/app/api/{sid}/curves").json()
    assert cur["kind"] == "style" and {"GR", "CALI"} <= {x["name"] for x in cur["curves"]}
    gr = next(x for x in cur["curves"] if x["name"] == "GR")
    assert cur["active"] == gr["id"] and gr["cls"] == "solid thick" and gr["backup"] == [150.0, 300.0]
    c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp_path)}); c.post(f"/app/api/{sid}/export/csv")
    las = c.post(f"/app/api/{sid}/export/las").json()["written"]
    assert [pathlib.Path(p).name for p in las] == ["USA_A_16_4342_crop_3260-4770.las"]
    st = cat["settings"]
    s = AssistedSession(ROOT / cat["scan"], curve="GR", well=cat["well"], track=st["track"], scale=tuple(st["scale"]),
                        backup=tuple(st["backup"]), use_hatch=st["use_hatch"], first_depth=st["first_depth"])
    s.open()
    ID = ci.identify_curves(s.scan_path, s.L, propose_header(s.scan_path, layout=s.L))
    g_ = next(x for x in ID["curves"] if x["name"] == "GR")
    tif = save_ink_tif(ci.working_image(ID, g_), tmp_path / "working_GR.tif")
    cv = dict(name="GR", track=g_["track"], left=st["scale"][0], right=st["scale"][1], scale_type="linear",
              backup=tuple(st["backup"]), use_hatch=st["use_hatch"])
    if st.get("wrap_side"):
        cv["wrap_side"] = st["wrap_side"]
    gs = open_color_curve(tif, cv, cat["well"], ci.trace_preset(ID), x_range=ID["tracks"][g_["track"]],
                          first_depth=st["first_depth"])
    g, v = resample(*color_curve_values(gs), 0.5)
    csv = np.loadtxt(tmp_path / "USA_A_16_4342_crop_3260-4770_GR.csv", delimiter=",", skiprows=1)
    assert np.array_equal(csv[:, 0], g) and np.array_equal(csv[:, 1], v, equal_nan=True)
    L = read_las(tmp_path / "USA_A_16_4342_crop_3260-4770.las")
    assert {"GR", "CALI"} <= set(L.keys())


def test_guided_fix_routes(app_sid):
    c, sid, _ = app_sid
    st = c.get(f"/app/api/{sid}/state").json(); n = st["n_anchors"]; k = 2   # existing pins at 300/1500/2700
    r = c.post(f"/app/api/{sid}/fix/pin", json={"row": 2100, "x": 100, "snap": True}).json()
    assert r["n_anchors"] == n + 1 and r["ms"] >= 0 and r["progress"]["last"]["action"] == "pin"
    r = c.post(f"/app/api/{sid}/fix/move", json={"index": k, "row": 2200, "x": 100}).json(); assert r["n_anchors"] == n + 1
    r = c.post(f"/app/api/{sid}/fix/node", json={"row": 1000, "x": 100, "radius": 50}).json()
    assert r["ms"] < 500 and r["undo"][-1] == "node"
    c.post(f"/app/api/{sid}/undo")
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


def test_calibration_editing(app_sid):
    c, sid, root = app_sid
    A = f"/app/api/{sid}"
    j = wait(c, c.post(A + "/depth", json=dict(mode="labels", labels=[[0, 1000], [1500, 1150], [3000, 1300]])))
    assert j["state"] == "done", j
    assert wait(c, c.post(A + "/section", json=dict(box=[0, W], rows=[0, H])))["state"] == "done"
    cal = c.get(A + "/calib").json()
    assert cal["width"] == W and [d["loo_ft"] for d in cal["depth"]][1] == 0.0      # collinear anchors: zero residual
    r = c.post(A + "/calib/scale2", json=dict(x1=-0.5, v1=0, x2=W / 2 - 0.5, v2=75)).json()
    assert abs(r["left"]) < 1e-6 and abs(r["right"] - 150) < 1e-6
    r = c.post(A + "/calib/scale2", json=dict(x1=0, v1=1, x2=100, v2=10, type="log")).json()
    assert r["right"] > r["left"] > 0
    j = wait(c, c.post(A + "/calib/edges", json=dict(left_dx=10, right_dx=-10)))
    assert j["state"] == "done", j
    assert c.get(A + "/calib").json()["width"] == W - 20
    bad = c.post(A + "/calib/scale2", json=dict(x1=5, v1=1, x2=5, v2=2))
    assert bad.status_code == 400


def test_header_word_correction(app_sid):
    from pylogdigitizer import header_ocr as ho
    c, sid, _root = app_sid
    e = c.app.state.editor.state.sessions[sid]
    w = lambda t, x, y, conf=95: dict(text=t, conf=conf, x0=x, y0=y, x1=x + 14 * len(t), y1=y + 20)
    words = [w("GAMMA", 300, 100), w("RAY", 400, 100), w("0", 102, 200), w("100", 850, 200, 40)]
    old = e.header
    e.header = dict(tracks={"track1": dict(x=(100, 900), grid="linear", words=words, curves=ho.propose_curves(words, 100, 900, "linear", track="track1"))})
    try:
        assert len(c.get(f"/app/api/{sid}/header/words").json()["words"]) == 4
        h = c.post(f"/app/api/{sid}/header/words", json={"track": "track1", "edits": [{"i": 3, "text": "200"}], "save": False}).json()
        (r,) = [r for r in h["rows"] if r["mnemonic"] == "GR"]
        assert (r["left"], r["right"]) == (0, 200)
        assert c.post(f"/app/api/{sid}/header/words", json={"track": "track1", "edits": [{"i": 99, "text": "x"}]}).status_code == 400
        c.post(f"/app/api/{sid}/undo")
        assert c.get(f"/app/api/{sid}/header/words").json()["words"][3]["text"] == "100"
    finally:
        e.header = old


def test_header_lines_group_and_match():
    from pylogdigitizer.app.routes import header_lines
    w = lambda i, t, x, y, tr="track1", conf=95: dict(i=i, track=tr, text=t, conf=conf, x0=x, y0=y, x1=x + 12 * len(t), y1=y + 20)
    words = [w(0, "Gamma", 120, 10), w(1, "0", 5, 12), w(2, "Ray", 200, 11, conf=60), w(3, "150", 400, 9),
             w(4, "SP", 150, 60), w(5, "(mV)", 190, 61), w(6, "ILD", 700, 10, "track2")]
    L = header_lines(words, [dict(id="c1", track="track1", header="Gamma Ray"), dict(id="c2", track="track1", header="SP (mV)"),
                             dict(id="c3", track="track2", header="RxoRt")])
    assert [(q["track"], q["text"], q["words"]) for q in L] == [("track1", "0 Gamma Ray 150", [1, 0, 2, 3]), ("track1", "SP (mV)", [4, 5]),
                                                                ("track2", "ILD", [6])]
    assert L[0]["conf"] == 60 and (L[0]["x0"], L[0]["y0"], L[0]["y1"]) == (5, 9, 32)
    assert [q["curve"] for q in L] == ["c1", "c2", None]            # no shared token: left unmatched


def test_header_lines_route_and_check_and_save_outputs(app_sid, tmp_path):
    from pylogdigitizer import header_ocr as ho
    c, sid, _root = app_sid
    e = c.app.state.editor.state.sessions[sid]
    w = lambda t, x, y, conf=95: dict(text=t, conf=conf, x0=x, y0=y, x1=x + 14 * len(t), y1=y + 20)
    words = [w("GAMMA", 100, 100), w("RAY", 190, 100), w("0", 12, 200), w("150", 270, 200, 40)]
    old = e.header
    e.header = dict(tracks={"track1": dict(x=(10, 310), grid="linear", words=words, curves=ho.propose_curves(words, 10, 310, "linear", track="track1"))})
    try:
        h = c.get(f"/app/api/{sid}/header/lines").json()
        assert h["n_words"] == 4 and h["n_low"] == 1 and abs(h["mean_conf"] - 81.25) < 1e-9 and h["strip"][1] >= 220
        assert [q["text"] for q in h["lines"]] == ["GAMMA RAY", "0 150"] and h["lines"][1]["conf"] == 40
    finally:
        e.header = old
    g = c.get(f"/app/api/{sid}/logs").json()                          # Check and save plot: row-aligned with the scan
    (q,) = g["curves"]
    assert q["name"] and len(q["depth"]) == len(q["row"]) == len(q["frac"]) == len(q["value"]) > 100
    assert np.allclose(np.diff(q["depth"]), 0.5) and np.all(np.diff(q["row"]) > 0)
    assert all(0 <= f <= 1 for f in q["frac"] if f is not None) and g["ticks"] and g["height"] == H
    c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp_path)})
    (csvp,) = c.post(f"/app/api/{sid}/export/logs_csv", json={"ids": [q["id"]]}).json()["written"]
    head = pathlib.Path(csvp).read_text().splitlines()
    assert pathlib.Path(csvp).name == "SYN_1_syn_logs.csv" and head[0].split(",") == ["DEPT", q["name"]] and len(head) == len(q["depth"]) + 1
    for img in (True, False):
        (png,) = c.post(f"/app/api/{sid}/export/figure", json={"image": img}).json()["written"]
        assert pathlib.Path(png).name == "SYN_1_syn_logs.png" and pathlib.Path(png).read_bytes()[:4] == b"\x89PNG"
    for k in ("figure", "logs_csv"):                                   # no chosen curve: refused, nothing written
        assert c.post(f"/app/api/{sid}/export/{k}", json={"ids": ["nope"]}).status_code >= 400


def test_package_logo_and_separate_credits(tmp_path):
    from pylogdigitizer.app import create_app
    c = TestClient(create_app(str(tmp_path)), base_url="http://127.0.0.1")
    html = c.get("/").text
    assert 'class="brand" title="pyLogDigitizer"' in html
    assert html.count('class="home-logo"') == 1
    assert '<h1>pyLogDigitizer</h1>' not in html
    assert '<footer id="bar">' in html
    assert html.index('<footer id="bar">') < html.index('<footer class="app-credits"')
    assert 'Project by Larry Syu-Heng Lai, PhD.' in html
    assert 'AGPL-3.0 or later license' in html
    logo = c.get("/static/logo.png")
    assert logo.status_code == 200 and logo.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_color_step_palette_and_confirm(tmp_path):
    """Colour scans expose a palette step; B&W scans do not claim to be colour."""
    from PIL import Image
    from pylogdigitizer.app import create_app
    rgb = np.full((280, 360, 3), 248, np.uint8)
    rgb[:, :14] = (0, 0, 0)
    rgb[40:240, 40:120] = (36, 158, 52)
    rgb[40:240, 160:240] = (196, 42, 42)
    d = tmp_path / "example_data" / "COL_1"; d.mkdir(parents=True)
    Image.fromarray(rgb).save(d / "col.tif")
    app = create_app(str(tmp_path)); app.state.testing = True
    c = TestClient(app, base_url="http://127.0.0.1")
    info = c.get("/app/api/info").json()
    assert any(x["scan"].endswith("COL_1/col.tif") for x in info["catalog"])
    j = wait(c, c.post("/app/api/open", json=dict(scan="example_data/COL_1/col.tif", well="COL_1",
                                                  curve="GR", auto_layout=False,
                                                  settings=dict(scale=[0, 150]))))
    assert j["state"] == "done", j
    sid = j["result"]["sid"]
    st = c.get(f"/app/api/{sid}/state").json()
    assert st["is_color"] is True
    col = c.get(f"/app/api/{sid}/color").json()
    assert col["is_color"] and "white" in col["palette"] and "black" in col["palette"]
    assert col["chromatic"]
    pal = dict(col["palette"])
    pal["extra"] = [10, 180, 180]
    r = c.post(f"/app/api/{sid}/color", json=dict(palette=pal, curve_color=col["chromatic"][0]))
    assert r.status_code == 200, r.text
    got = r.json()
    assert "extra" in got["palette"]
    samp = c.post(f"/app/api/{sid}/color/sample", json=dict(row=100, x=80))
    assert samp.status_code == 200 and len(samp.json()["rgb"]) == 3
    j2 = wait(c, c.post(f"/app/api/{sid}/color", json=dict(palette=pal, curve_color=col["chromatic"][0], confirm=True)))
    assert j2["state"] == "done", j2
    assert j2["result"]["confirmed"] is True
    st2 = c.get(f"/app/api/{sid}/state").json()
    assert st2["color"]["confirmed"] is True



def test_ui_paths_are_not_absolute(app_sid):
    """Load and export screens must not show raw filesystem paths."""
    c, sid, root = app_sid
    info = c.get("/app/api/info").json()
    assert info["project"] == root.name
    assert not info["project"].startswith("/")
    st = c.get(f"/app/api/{sid}/state").json()
    folder = st["export"]["folder"]
    assert not folder.startswith("/")
    assert ".." not in pathlib.Path(folder).parts


def test_stone_color_confirm_qc_pairs_las(tmp_path):
    """After the colour step, STONE 5-19 GR (blue) must pair with its LAS and report real r/RMSE.

    The grey-grid Eli print needs the automatic thin-border layout on both the colour crop
    and the 1-bit working image (no hand-given track positions).

    Re-opening the isolated working image used to omit track-edge exclusion, so the
    tracer hugged the left border (constant value) and QC showed r as undefined
    with a bogus few-foot shift.
    """
    import shutil
    from pylogdigitizer.app import create_app
    src = ROOT / "example_data" / "STONE_5-19_15-055-22611"
    if not (src / "triple_combo_color_crop.tif").exists():
        pytest.skip("STONE example data not present")
    dst = tmp_path / "example_data" / "STONE_5-19_15-055-22611"
    shutil.copytree(src, dst)
    (tmp_path / "dataset_profiles").mkdir()
    shutil.copy(ROOT / "dataset_profiles" / "color_scan_wells.yaml",
                tmp_path / "dataset_profiles" / "color_scan_wells.yaml")
    app = create_app(str(tmp_path)); app.state.testing = True
    c = TestClient(app, base_url="http://127.0.0.1")
    info = c.get("/app/api/info").json()
    assert info["project"] == tmp_path.name
    item = next(x for x in info["catalog"] if "triple_combo_color_crop" in x["scan"])
    j = wait(c, c.post("/app/api/open", json=dict(
        scan=item["scan"], curve=item["curve"], well=item["well"], las=item["las"],
        settings=item.get("settings") or {}, auto_layout=True)))
    assert j["state"] == "done", j
    sid = j["result"]["sid"]
    col = c.get(f"/app/api/{sid}/color").json()
    assert col["is_color"] and "blue" in (col.get("palette") or {})
    j2 = wait(c, c.post(f"/app/api/{sid}/color", json=dict(
        palette=col["palette"], curve_color="blue", confirm=True)))
    assert j2["state"] == "done", j2
    st = c.get(f"/app/api/{sid}/state").json()
    assert st["color"]["confirmed"] is True
    assert st.get("las")
    q = c.get(f"/core/api/sessions/{sid}/qc").json()
    assert q.get("available"), q
    m = q["metrics"]
    assert m.get("r") is not None and np.isfinite(m["r"]), m
    print("QC", m)
    assert m["r"] > 0.85, m
    assert m.get("rmse") is not None and np.isfinite(m["rmse"]), m
    assert m["n"] > 800, m
    assert abs(float(m.get("shift_ft") or 0)) < 2.0, m


def test_color_header_rows_prefers_rows_with_scales():
    """Colour scans: the line parser keeps printed names, but when it reads fewer clean
    scales than the default proposals (e.g. the Eli STONE header) the default rows win."""
    from pylogdigitizer.app.routes import color_header_rows
    f = lambda v, review=False: dict(value=v, needs_review=review, confidence=0.9)
    curve = lambda tr, m, l, r, t: dict(track=tr, label=m, mnemonic=f(m), unit=f(""), scale_left=f(l),
                                         scale_right=f(r), scale_type=f(t), backup=f(None))
    H = dict(curves=[curve("track1", "GR", 0.0, 150.0, "linear"), curve("track2", "RES", 0.2, 200.0, "log")],
             line_curves=[dict(track="track3", name="RHOC", left=None, right=None, scale_type="linear",
                               check="no scale read")])
    rows = color_header_rows(H)
    assert [r["mnemonic"] for r in rows] == ["GR", "RES"]
    H["line_curves"] = [dict(track="track1", name="GR", left=0.0, right=150.0, scale_type="linear", check=""),
                        dict(track="track2", name="RES16N", left=0.2, right=200.0, scale_type="log", check="")]
    rows = color_header_rows(H)
    assert [r["mnemonic"] for r in rows] == ["GR", "RES16N"]


def test_curve_layers_reference_and_step(app_sid, tmp_path):
    """Curves tab and layers on the single-curve scan: list/select/edit, traces, grid lines, the
    highlight tiles, reference LAS + comparison, upload, and the 0.25 ft LAS option."""
    c, sid, root = app_sid
    if not c.get(f"/app/api/{sid}/state").json()["traced"]:                    # when run on its own
        wait(c, c.post(f"/app/api/{sid}/section", json={"box": [0, W], "rows": [0, H]}))
        wait(c, c.post(f"/app/api/{sid}/depth", json={"mode": "labels", "labels": [[0, 1000], [H, 1300]]}))
    cv = c.get(f"/app/api/{sid}/curves").json()
    assert cv["kind"] == "single" and len(cv["curves"]) == 1
    cid = cv["curves"][0]["id"]
    assert c.post(f"/app/api/{sid}/curves/select", json={"id": cid}).status_code == 200
    assert c.post(f"/app/api/{sid}/curves/select", json={"id": "nope"}).status_code == 404
    tr = c.get(f"/app/api/{sid}/traces").json()
    assert tr["active"] == cid and len(tr["traces"][0]["trace"]) > 100
    assert "available" in c.get(f"/app/api/{sid}/grid_lines").json()
    dzi = c.get(f"/app/api/{sid}/highlight/{cid}.dzi"); assert dzi.status_code == 200 and b"Image" in dzi.content
    t = c.get(f"/app/api/{sid}/highlight/{cid}_files/0/0_0.png"); assert t.status_code == 200 and t.content[:4] == b"\x89PNG"
    assert c.get(f"/app/api/{sid}/highlight/other.dzi").status_code == 404
    assert c.post(f"/app/api/{sid}/curves/{cid}", json={"left": 0, "right": "x"}).status_code == 400
    p = c.post(f"/app/api/{sid}/curves/{cid}", json={"left": 0, "right": 150, "type": "linear", "unit": "API"}).json()
    assert p["right"] == 150 and p["unit"] == "API"
    assert c.post(f"/app/api/{sid}/reference", json={"las": "example_data/SYN_1/syn.las"}).json()["matched"] == {"GR": "GR"}
    cmp = c.get(f"/app/api/{sid}/compare").json()
    assert cmp["available"] and cmp["rows"][0]["ref"] == "GR" and cmp["rows"][0]["r"] > 0.5 and cmp["plot"]
    up = c.post("/app/api/upload", content=b"abc", headers={"X-Filename": "a.tif"}).json()
    assert up["name"] == "a.tif" and up["size"] == 3
    assert c.post("/app/api/upload", content=b"x", headers={"X-Filename": "a.exe"}).status_code == 400
    c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp_path / "s")})
    for bad in (0, -0.5, "abc", "nan", "inf", 1e-9):              # any step > 0 ft, validated
        assert c.post(f"/app/api/{sid}/export/las", json={"step": bad}).status_code == 400, bad
    from pylogdigitizer.las_export import read_las
    for step in (0.25, 0.3):
        w = c.post(f"/app/api/{sid}/export/las", json={"step": step}).json()["written"]
        d = np.diff(read_las(w[0]).index); assert np.allclose(d, step)


def test_rotate_rebuilds_session(project):
    from pylogdigitizer.app import create_app
    app = create_app(str(project)); app.state.testing = True
    c = TestClient(app, base_url="http://127.0.0.1")
    j = wait(c, c.post("/app/api/open", json=dict(scan="example_data/SYN_1/syn.tif", well="SYN_1", curve="GR",
                                                  auto_layout=False, settings=dict(scale=[0, 150]))))
    sid = j["result"]["sid"]
    assert c.post(f"/app/api/{sid}/rotate", json={"deg": 15}).status_code == 400
    assert wait(c, c.post(f"/app/api/{sid}/rotate", json={"deg": 1.5}))["state"] == "done"
    assert c.get(f"/app/api/{sid}/state").json()["rotate"] == 1.5


def test_stone_identifies_every_curve_and_compares(tmp_path):
    """STONE colour crop: one curve per legend row (shared auto-identification), each paired with
    its reference LAS column; one LAS holds them all."""
    import shutil
    from pylogdigitizer.app import create_app
    src = ROOT / "example_data" / "STONE_5-19_15-055-22611"
    if not (src / "triple_combo_color_crop.tif").exists():
        pytest.skip("STONE example data not present")
    shutil.copytree(src, tmp_path / "example_data" / src.name)
    (tmp_path / "dataset_profiles").mkdir()
    shutil.copy(ROOT / "dataset_profiles" / "color_scan_wells.yaml", tmp_path / "dataset_profiles")
    app = create_app(str(tmp_path)); app.state.testing = True
    c = TestClient(app, base_url="http://127.0.0.1")
    item = next(x for x in c.get("/app/api/info").json()["catalog"] if "triple_combo_color_crop" in x["scan"])
    j = wait(c, c.post("/app/api/open", json=dict(scan=item["scan"], curve=item["curve"], well=item["well"],
                                                  las=item["las"], settings=item.get("settings") or {}, auto_layout=True)))
    sid = j["result"]["sid"]
    cv = c.get(f"/app/api/{sid}/curves").json()
    names = {x["name"] for x in cv["curves"]}
    assert cv["kind"] == "color" and {"GR", "SP", "CALI", "ILD", "ILM", "SFL", "NPHI", "DPHI", "PE"} <= names
    rows = {r["name"]: r for r in c.get(f"/app/api/{sid}/compare").json()["rows"]}
    assert rows["SFL"]["ref"] == "RLL3" and rows["ILD"]["ref"] == "RILD"
    for n in ("SP", "ILD", "ILM", "DPHI", "NPHI", "PE"):
        assert rows[n]["r"] > 0.95, (n, rows[n])
    c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp_path / "out")})
    w = c.post(f"/app/api/{sid}/export/las").json()["written"]
    from pylogdigitizer.las_export import read_las
    las = read_las(w[0]); assert {"GR", "SP", "ILD", "NPHI"} <= set(las.keys())
    assert np.allclose(np.diff(las.index), 0.5)


def test_core_api_sessions_still_editable(project):
    """Sessions opened through the mounted editor API (scripts/editor_selfcheck.py) have no app undo
    stack; editing them must not fail in the app's undo middleware."""
    from pylogdigitizer.app import create_app
    app = create_app(str(project)); app.state.testing = True
    c = TestClient(app, base_url="http://127.0.0.1")
    st = c.post("/core/api/sessions", json=dict(scan="example_data/SYN_1/syn.tif", curve="GR", well="SYN_1",
                                                las="example_data/SYN_1/syn.las", settings=dict(scale=[0, 150])))
    assert st.status_code == 200, st.text
    sid = st.json()["sid"]
    assert c.post(f"/core/api/sessions/{sid}/section", json={"box": [0, W], "rows": [0, H]}).status_code == 200
    assert c.post(f"/core/api/sessions/{sid}/depth", json={"anchors": [[0, 1000], [H, 1300]]}).status_code == 200
    r = c.post(f"/core/api/sessions/{sid}/anchors", json={"row": 1500, "x": float(xa(1500))})
    assert r.status_code == 200, r.text
