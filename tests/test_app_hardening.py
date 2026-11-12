# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Desktop app hardening: a busy session never freezes the server, the export folder round trip,
export names and columns, colour header edits, uploads, host/origin checks and locked reads."""
import pathlib, threading, time
from urllib.parse import quote
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

H, W = 2000, 320
xa = lambda r: 90 + 45 * np.sin(np.asarray(r) / 150)
LOCAL = "http://127.0.0.1"


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    import cv2
    from PIL import Image
    from pylogdigitizer.las_export import write_las
    root = tmp_path_factory.mktemp("hardproj")
    d = root / "example_data" / "SYN_1"; d.mkdir(parents=True)
    img = np.full((H, W), 255, np.uint8); r = np.arange(H)
    cv2.polylines(img, [np.c_[xa(r), r].astype(np.int32).reshape(-1, 1, 2)], False, 0, 6)
    Image.fromarray(img).save(d / "syn.tif")
    depth = 1000 + np.arange(0, 200, 0.5)
    write_las(d / "syn.las", depth, {"GR": xa((depth - 1000) * 10) / W * 150}, units={"GR": "API"})
    return root


def wait(c, r):
    assert r.status_code == 200, r.text
    jid = r.json()["job"]["id"]
    while (j := c.get(f"/shell/api/jobs/{jid}").json())["state"] == "running":
        time.sleep(0.05)
    assert j["state"] == "done", j
    return j


@pytest.fixture(scope="module")
def sess(project):
    from pylogdigitizer.app import create_app
    app = create_app(str(project)); app.state.testing = True
    with TestClient(app, base_url=LOCAL) as c:            # one event loop for every request (like uvicorn)
        j = wait(c, c.post("/app/api/open", json=dict(scan="example_data/SYN_1/syn.tif", well="SYN_1", curve="GR",
                                                      las="example_data/SYN_1/syn.las", auto_layout=False,
                                                      settings=dict(scale=[0, 150], trace_kw={"straight_len": 0}))))
        sid = j["result"]["sid"]
        wait(c, c.post(f"/app/api/{sid}/section", json={"box": [0, W], "rows": [0, H]}))
        wait(c, c.post(f"/app/api/{sid}/depth", json={"mode": "labels", "labels": [[0, 1000], [H, 1200]]}))
        yield c, app, sid, app.state.editor.state.sessions[sid]


def _in_thread(fn):
    out = {}
    t = threading.Thread(target=lambda: out.update(r=fn()), daemon=True); t.start()
    return t, out


def test_busy_session_does_not_freeze_the_server(sess):
    """1. An editor edit waiting for a running job's lock must not block other requests."""
    c, app, sid, e = sess
    e.lock.acquire()
    try:
        t, out = _in_thread(lambda: c.post(f"/core/api/sessions/{sid}/scale", json={"left": 0, "right": 150}))
        time.sleep(0.3); assert t.is_alive()                      # waiting for the lock ...
        t2, out2 = _in_thread(lambda: c.get("/app/api/info"))    # ... while the server still answers
        t2.join(10); assert not t2.is_alive() and out2["r"].status_code == 200
    finally:
        e.lock.release()
    t.join(30); assert not t.is_alive() and out["r"].status_code == 200


def test_export_folder_outside_project_round_trip(sess, tmp_path):
    """2. The state keeps the absolute folder; the label is only shown; Open folder uses the real path."""
    c, app, sid, e = sess
    out = tmp_path / "outside"
    st = c.post(f"/app/api/{sid}/export/folder", json={"path": str(out)}).json()
    assert st["folder"] == "outside" and pathlib.Path(st["folder_abs"]) == out.resolve()
    x = c.get(f"/app/api/{sid}/state").json()["export"]
    assert pathlib.Path(x["folder_abs"]) == out.resolve()
    out.mkdir()
    assert c.post("/shell/api/open_folder", json={"path": x["folder_abs"]}).status_code == 200


def test_empty_ids_refused(sess, tmp_path):
    """3. ``ids: []`` (None ticked in the plot picker) writes nothing."""
    c, app, sid, e = sess
    c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp_path)})
    for k in ("logs_csv", "figure"):
        assert c.post(f"/app/api/{sid}/export/{k}", json={"ids": []}).status_code == 400
    assert not list(tmp_path.iterdir())


def test_safe_export_names(sess, tmp_path):
    """4. Curve names with / \\ : never nest, escape or collide in export file names."""
    from pylogdigitizer.app.routes import safe_name
    assert safe_name("RES/16N") == "RES_16N" and safe_name("a\\b:c") == "a_b_c" and safe_name("../") == "curve"
    c, app, sid, e = sess
    cid = e.active; old = e.curves[0]["name"]
    try:
        c.post(f"/app/api/{sid}/curves/{cid}", json={"name": "RES/16N"})
        c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp_path)})
        r = c.post(f"/app/api/{sid}/export/csv"); assert r.status_code == 200, r.text
        (p,) = r.json()["written"]
        assert pathlib.Path(p).parent == tmp_path.resolve() and pathlib.Path(p).name == "SYN_1_syn_RES_16N.csv"
    finally:
        c.post(f"/app/api/{sid}/curves/{cid}", json={"name": old})


def test_unique_curve_keys_in_las_and_logs_csv(sess, tmp_path):
    """5. Two curves with the same base name both reach the LAS and the logs CSV."""
    from pylogdigitizer.app.routes import unique_keys
    assert unique_keys(["GR", "GR", "ILD:a", "ILD:b", "GR_2"]) == ["GR", "GR_2", "ILD", "ILD_2", "GR_2_2"]
    c, app, sid, e = sess
    e.curves.append(dict(e.curves[0], id="dup"))
    try:
        c.post(f"/app/api/{sid}/export/folder", json={"path": str(tmp_path)})
        (las,) = c.post(f"/app/api/{sid}/export/las").json()["written"]
        from pylogdigitizer.las_export import read_las
        L = read_las(las); name = e.curves[0]["name"]
        assert {name, f"{name}_2"} <= set(L.keys())
        (p,) = c.post(f"/app/api/{sid}/export/logs_csv").json()["written"]
        assert pathlib.Path(p).read_text().splitlines()[0].split(",") == ["DEPT", name, f"{name}_2"]
    finally:
        e.curves.pop()


def test_colour_header_word_edit_keeps_line_rows(sess, monkeypatch):
    """6. A word edit on a line-parsed (colour) header rebuilds the table with color_header_rows."""
    from pylogdigitizer.app import routes
    from pylogdigitizer import header_ocr
    c, app, sid, e = sess
    sentinel = [dict(name="LINE_ROW", left=0, right=150)]
    monkeypatch.setattr(routes, "color_header_rows", lambda h: sentinel)
    monkeypatch.setattr(header_ocr, "apply_word_edits", lambda h, t, ed: h)
    old = (getattr(e, "header", None), e.app["header_rows"], e.app["header_ocr"])
    e.header = dict(parse="lines", tracks={"track1": dict(words=[], curves=[])}, curves=[])
    try:
        r = c.post(f"/app/api/{sid}/header/words", json={"track": "track1", "edits": [], "save": False})
        assert r.status_code == 200 and r.json()["rows"] == sentinel
    finally:
        e.header, e.app["header_rows"], e.app["header_ocr"] = old
        e.undo.pop()


def test_upload_non_ascii_name_size_cap_and_run_folder(sess):
    """7. URL-encoded file names, a size cap (413, nothing left behind) and one folder per app run."""
    c, app, sid, e = sess
    name = "Bohrloch_ü_測井.tif"
    up = c.post("/app/api/upload", content=b"abc", headers={"X-Filename": quote(name)}).json()
    p = pathlib.Path(up["path"])
    assert up["name"] == name and p.read_bytes() == b"abc" and p.is_relative_to(app.state.run_dir)
    app.state.upload_max = 10
    try:
        assert c.post("/app/api/upload", content=b"x" * 20, headers={"X-Filename": "big.tif"}).status_code == 413
    finally:
        del app.state.upload_max
    assert not any(q.name.startswith("big.tif") for q in p.parent.iterdir())


def test_host_and_origin_checks(project, sess):
    """8. Only local Host names; writes only from the app's own origin; no paths outside the project."""
    from pylogdigitizer.app import create_app
    c, app, sid, e = sess
    assert TestClient(create_app(str(project))).get("/app/api/info").status_code == 400     # Host: testserver
    assert c.get("/app/api/info", headers={"host": "evil.example"}).status_code == 400
    bad = c.post(f"/app/api/{sid}/export/folder", json={"path": "/tmp"}, headers={"origin": "http://evil.example"})
    assert bad.status_code == 403
    ok = c.post(f"/app/api/{sid}/curves/select", json={"id": e.active}, headers={"origin": LOCAL})
    assert ok.status_code == 200
    assert c.post("/app/api/open", json={"scan": "/etc/hostname"}).status_code == 400
    assert c.post("/app/api/open", json={"scan": "../outside.tif"}).status_code == 400


@pytest.mark.parametrize("path", ["traces", "compare", "curves", "trace/options"])
def test_reads_wait_for_the_lock(sess, path):
    """9. Reads of curves and traces wait for a running job instead of returning torn data."""
    c, app, sid, e = sess
    e.lock.acquire()
    try:
        t, out = _in_thread(lambda: c.get(f"/app/api/{sid}/{path}"))
        time.sleep(0.3); assert t.is_alive()
    finally:
        e.lock.release()
    t.join(30); assert out["r"].status_code == 200, out["r"].text


def test_curves_edit_undo_snapshot_under_lock(sess):
    """9. The curves_edit undo snapshot is taken under the lock (never during a job)."""
    c, app, sid, e = sess
    n0 = len(e.undo.items); cid = e.active
    e.lock.acquire()
    try:
        t, out = _in_thread(lambda: c.post(f"/app/api/{sid}/curves/{cid}", json={"unit": "API"}))
        time.sleep(0.3); assert t.is_alive() and len(e.undo.items) == n0
    finally:
        e.lock.release()
    t.join(30); assert out["r"].status_code == 200 and len(e.undo.items) == n0 + 1


def test_bad_export_step_is_400(sess):
    """Nice to have 2: a non-numeric LAS step is a 400, not a 500."""
    c, app, sid, e = sess
    assert c.post(f"/app/api/{sid}/export/las", json={"step": "x"}).status_code == 400
