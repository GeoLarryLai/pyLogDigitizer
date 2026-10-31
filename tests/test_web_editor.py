# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Web editor + assisted tracing + training-data format (synthetic scan, fast)."""
import json
import numpy as np
import pytest

H, W = 3000, 320
xa = lambda r: 90 + 45 * np.sin(np.asarray(r) / 150)          # curve A (thicker, the tracer's choice)
xb = lambda r: 235 + 40 * np.cos(np.asarray(r) / 170)         # curve B (distractor the tracer prefers)
WANT = xa                                                    # the target curve (in the LAS)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    import cv2
    from PIL import Image
    from pylogdigitizer.las_export import write_las
    root = tmp_path_factory.mktemp("proj")
    d = root / "example_data" / "SYN_1"; d.mkdir(parents=True)
    img = np.full((H, W), 255, np.uint8); r = np.arange(H)
    cv2.polylines(img, [np.c_[xa(r), r].astype(np.int32).reshape(-1, 1, 2)], False, 0, 6)
    cv2.polylines(img, [np.c_[xb(r), r].astype(np.int32).reshape(-1, 1, 2)], False, 0, 4)
    Image.fromarray(img).save(d / "syn.tif")
    depth = 1000 + np.arange(0, 300, 0.5)                      # 10 px/ft
    write_las(d / "syn.las", depth, {"GR": xa((depth - 1000) * 10) / W * 150}, units={"GR": "API"})
    return root


def _session(project, **kw):
    from pylogdigitizer.assisted_tracing import AssistedSession
    s = AssistedSession(project / "example_data/SYN_1/syn.tif", curve="GR", well="SYN_1",
                        las_path=project / "example_data/SYN_1/syn.las", auto_layout=False, scale=(0, 150),
                        trace_kw={"straight_len": 0}, **kw).open()
    s.set_depth_anchors([(0, 1000), (H, 1300)])
    s.set_section(box=(0, W), rows=(0, H))
    return s


def test_assisted_seed_and_anchor(project):
    s = _session(project)
    s.auto_trace()                                             # conventional: jumps onto the distractor B in places
    q0 = s.qc()["metrics"]["r"]
    assert q0 < 0.9                                            # partly on the wrong curve
    for row in (300, 1500, 2700):                              # three clicks on curve B
        s.add_anchor(row, float(WANT(row)) + 3, window_ft=400)   # imprecise click, snapped to ink
    R = s.result()
    assert (np.abs(R["x_panel"] - WANT(R["rows"])) < 6).mean() > 0.9
    q1 = s.qc()["metrics"]["r"]
    assert q1 > 0.95
    s.remove_anchor(0); assert len(s.anchors) == 2
    s2 = _session(project); s2.auto_trace(seed=(100, float(WANT(100))))   # seed click
    R = s2.result(); assert (np.abs(R["x_panel"] - WANT(R["rows"])) < 6)[R["rows"] < 400].mean() > 0.9


def test_local_retrace_does_not_move_far_rows(project):
    s = _session(project); s.auto_trace()
    before = s.T["x"].copy()
    s.add_anchor(1500, float(WANT(1500)), window_ft=20)          # +-200 px window
    a, b = s.last_retrace["sheet_rows"]
    out = (s.T["rows"] < a) | (s.T["rows"] > b)
    assert np.array_equal(before[out], s.T["x"][out]) and b - a <= 420


def test_session_save_replay(project, tmp_path):
    from pylogdigitizer.assisted_tracing import AssistedSession
    s = _session(project); s.auto_trace(seed=(100, float(WANT(100))))
    for row in (900, 1700, 2500):
        s.add_anchor(row, float(WANT(row)), window_ft=400)
    p = s.save_session(tmp_path / "sess.json")
    s2 = AssistedSession.from_session_file(p)
    assert s2.anchors == s.anchors and s2.section == s.section
    assert np.abs(s2.result()["x_panel"] - WANT(s2.result()["rows"])).mean() < 6


def test_training_data_roundtrip(project, tmp_path):
    from pylogdigitizer import ml_training_data as mtd
    s = _session(project); s.auto_trace(seed=(100, float(WANT(100))))
    d = s.save_training(tmp_path / "corr")
    assert sorted(f.name for f in d.iterdir()) == sorted(mtd.FILES)
    smp = mtd.load_correction(d)
    assert smp["image"].shape == (H, W) and smp["mask"].shape == (H, W) and smp["image"].dtype == np.uint8
    assert len(smp["trace"]["depth_ft"]) == len(s.result()["depth"])
    assert np.allclose(smp["trace"]["value"], s.result()["value"], atol=1e-3)
    assert smp["anchors"] == [list(a) for a in s.anchors]
    assert smp["metadata"]["well"] == "SYN_1" and smp["metadata"]["format_version"] == mtd.FORMAT_VERSION
    assert smp["calibration"]["scale"]["right"] == 150
    # mask is on the traced curve
    r = smp["trace"]["panel_row"].astype(int); x = smp["trace"]["x_panel"].astype(int)
    assert smp["mask"][r, np.clip(x, 0, W - 1)].mean() > 0.95
    assert mtd.list_corrections(tmp_path / "corr") == [d]
    tiles = list(mtd.corrections_to_tiles(tmp_path / "corr", size=256))
    assert tiles and tiles[0][1].shape == (256, 256)
    with pytest.raises(FileNotFoundError, match="mask.png"):
        (d / "mask.png").unlink(); mtd.load_correction(d)


def test_ml_training_data_helpers():
    from pylogdigitizer import ml_training_data as mtd
    sp = mtd.split_by_well([f"w{i}" for i in range(20)], (0.7, 0.15, 0.15))
    assert sum(map(len, sp.values())) == 20 and not set(sp["train"]) & set(sp["test"])
    m = mtd.las_to_pixel_labels((100, 50), lambda d: d - 1000, lambda v: v, np.arange(1000, 1100.0), np.full(100, 25.0))
    assert m[:, 25].sum() >= 95


@pytest.fixture(scope="module")
def client(project):
    from fastapi.testclient import TestClient
    from pylogdigitizer.web_editor import create_app
    return TestClient(create_app(str(project)))


def test_api_end_to_end(client, project):
    pytest.importorskip("matplotlib")
    cat = client.get("/api/catalog").json()
    assert any(c["scan"] == "example_data/SYN_1/syn.tif" for c in cat)
    assert "OpenSeadragon" in client.get("/static/vendor/openseadragon/openseadragon.min.js").text[:2000] or \
        client.get("/static/vendor/openseadragon/openseadragon.min.js").status_code == 200
    assert "editor.js" in client.get("/").text
    st = client.post("/api/sessions", json=dict(scan="example_data/SYN_1/syn.tif", las="example_data/SYN_1/syn.las",
                                                curve="GR", well="SYN_1", auto_layout=False,
                                                settings={"scale": [0, 150], "trace_kw": {"straight_len": 0}})).json()
    sid = st["sid"]; S = f"/api/sessions/{sid}"
    assert st["errors"] == {} and st["section"] is None and st["width"] == W
    assert "<Image" in client.get(S + "/image.dzi").text
    t = client.get(S + "/image_files/12/0_0.png"); assert t.status_code == 200 and t.content[:4] == b"\x89PNG"
    assert client.get(S + "/image_files/0/0_0.png").status_code == 200
    assert client.get(S + "/image_files/12/99_99.png").status_code == 404
    r = client.post(S + "/section", json={"box": [0, W], "rows": [0, H]}).json()   # trace works before depth calibration
    assert r["traced"] and r["qc"] is None and r["calibration"]["px_per_ft"] is None
    assert client.post(S + "/export_las").status_code == 400                         # ...but LAS export needs depth
    st = client.post(S + "/depth", json={"anchors": [[0, 1000], [H, 1300]]}).json()
    assert abs(st["calibration"]["px_per_ft"] - 10) < 1e-6
    st = client.post(S + "/section", json={"box": [0, W], "rows": [0, H], "seed": [100, float(WANT(100))]}).json()
    assert st["traced"] and st["section"]["kind"] == "box"
    for row in (900, 1700, 2500):
        st = client.post(S + "/anchors", json={"row": row, "x": float(WANT(row)), "window_ft": 400}).json()
    assert st["n_anchors"] == 4 and st["qc"]["r"] > 0.95
    ov = client.get(S + "/overlay").json(); assert len(ov["anchors"]) == 4 and len(ov["trace"]) > 100
    q = client.get(S + "/qc").json(); assert q["available"] and q["metrics"]["n"] > 100
    assert client.get(S + "/qc.png").content[:4] == b"\x89PNG"
    st = client.post(S + "/scale", json={"left": 0, "right": 300}).json(); assert st["calibration"]["scale"]["right"] == 300
    las = client.post(S + "/export_las").json(); assert (project / las["path"]).exists()
    assert "~A" in client.get(las["url"]).text
    tr = client.post(S + "/save_training").json(); assert (project / tr["path"] / "metadata.json").exists()
    ss = client.post(S + "/save_session").json()
    assert json.loads((project / ss["path"]).read_text())["scan"] == "example_data/SYN_1/syn.tif"
    assert client.delete(S + "/anchors/0").json()["n_anchors"] == 3
    assert client.delete(S + "/anchors/9").status_code == 404


def test_api_errors(client):
    assert client.post("/api/sessions", json={"scan": "../../etc/passwd"}).status_code in (400, 404)
    assert client.get("/api/sessions/nope").status_code == 404


def test_server_binds_localhost_only():
    from pylogdigitizer import web_editor
    assert web_editor.HOST == "127.0.0.1"


def test_cli_has_editor():
    from pylogdigitizer.cli import main
    with pytest.raises(SystemExit) as e:
        main(["editor", "--help"])
    assert e.value.code == 0


def test_exclude_stroke_and_backup_interval(project):
    s = _session(project, backup=(150, 300))
    s.auto_trace()
    R = s.result(); wrong = np.abs(R["x_panel"] - xb(R["rows"])) < 6     # rows on the distractor B
    assert wrong.any()
    k = int(np.flatnonzero(wrong)[len(np.flatnonzero(wrong)) // 2]); row = float(R["rows"][k])
    a, b = s.exclude_stroke(row, float(xb(row)), panel_x=True)          # "not this curve" on B
    assert b > a and len(s.excluded) == 1
    R = s.result(); assert (np.abs(R["x_panel"] - xb(R["rows"])) < 6).mean() < wrong.mean()
    s.remove_exclusions(); assert s.excluded == [] and np.array_equal(s.T["cost"], s.T["cost0"])
    v0 = s.result()["value"].copy()
    s.set_backup_interval(1000, 1500)                                   # read rows 1000-1500 on 150-300
    R = s.result(); m = (R["rows"] >= 1000) & (R["rows"] <= 1500) & ~R["off_scale"]
    assert np.allclose(R["value"][m], v0[m] + 150, atol=1e-6) and np.array_equal(R["value"][~m], v0[~m])
    d = s.session_dict(); assert d["backup_intervals"] == [[1000.0, 1500.0]] and d["excluded"] == []


def test_api_corrections(client):
    st = client.post("/api/sessions", json=dict(scan="example_data/SYN_1/syn.tif", las="example_data/SYN_1/syn.las",
                                                curve="GR", auto_layout=False, settings={"scale": [0, 150], "trace_kw": {"straight_len": 0}})).json()
    S = f"/api/sessions/{st['sid']}"
    client.post(S + "/depth", json={"anchors": [[0, 1000], [H, 1300]]})
    client.post(S + "/section", json={"box": [0, W], "rows": [0, H]})
    assert client.post(S + "/backup_interval", json={"rows": [0, 100]}).status_code == 400    # no backup scale yet
    client.post(S + "/scale", json={"left": 0, "right": 150, "backup": [150, 300]})
    st = client.post(S + "/backup_interval", json={"rows": [0, 100]}).json(); assert st["n_backup_intervals"] == 1
    assert client.post(S + "/backup_interval/clear").json()["n_backup_intervals"] == 0
    assert client.post(S + "/exclude", json={"row": 5, "x": 1}).status_code == 400            # no ink there
    assert client.post(S + "/retrace_all").json()["traced"]


def test_api_proposed_header(client, monkeypatch):
    """The proposed-header endpoint returns header_ocr proposals unapplied (OCR is faked here)."""
    from pylogdigitizer import header_ocr
    import pylogdigitizer.web_editor.app as appmod
    f = lambda v, c: dict(value=v, confidence=c, needs_review=c < 0.75)
    fake = dict(header_rows=[0, 100], runtime_s=0.1, backend="rapidocr", status="proposed", scan="syn.tif",
                curves=[dict(track="track1", label="GAMMA RAY", status="proposed", mnemonic=f("GR", 0.9),
                             unit=f("GAPI", 0.9), scale_left=f(0, 0.9), scale_right=f(150, 0.9),
                             scale_type=f("linear", 0.9), backup=f([150, 300], 0.6))])
    calls = []
    monkeypatch.setattr(header_ocr, "propose_header", lambda *a, **k: calls.append(1) or fake)
    monkeypatch.setattr(appmod, "header_ocr_available", lambda: True)
    st = client.post("/api/sessions", json=dict(scan="example_data/SYN_1/syn.tif", curve="GR", auto_layout=False,
                                                settings={"scale": [0, 150]})).json()
    S = f"/api/sessions/{st['sid']}"
    h = client.get(S + "/proposed_header").json()
    assert h["status"] == "proposed" and h["curves"][0]["mnemonic"]["value"] == "GR"
    assert h["curves"][0]["backup"]["needs_review"]
    client.get(S + "/proposed_header"); assert len(calls) == 1          # cached per session
    assert client.get(S).json()["calibration"]["scale"]["right"] == 150  # nothing applied
