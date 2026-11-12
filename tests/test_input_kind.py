# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Input-type detection (scan / vector / mixed), the exact vector-path workflow and its resampling,
the mixed-PDF header, and the app's import step. Tiny PDFs are generated here with reportlab
(tests only; no example data)."""
from __future__ import annotations

import time

import numpy as np
import pytest

reportlab = pytest.importorskip("reportlab")
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from pylogdigitizer import detect_input_kind  # noqa: E402
from pylogdigitizer.pdf_vector import digitize_vector, mixed_header, resample_path, read_vector_page  # noqa: E402

PAGE = (300.0, 500.0)
TOP, BOT = 400.0, 50.0                  # plot frame (pt)
D0, D1 = 1000.0, 1070.0                 # depth at TOP / BOT
T1, T2 = (40.0, 140.0), (140.0, 260.0)  # track borders
y_of = lambda d: TOP + (np.asarray(d, float) - D0) / (D1 - D0) * (BOT - TOP)
gr = lambda d: 75 + 50 * np.sin((np.asarray(d) - D0) / 7.0)            # 0-150 on track 1
rt = lambda d: 10 ** (0.5 + 1.2 * np.cos((np.asarray(d) - D0) / 11.0))  # 0.2-200 log on track 2


def _frame_and_text(c, header=True):
    c.setLineWidth(0.5); c.setStrokeColorRGB(0, 0, 0)
    for x in (10.0, T1[0], T1[1], T2[1]):
        c.line(x, BOT, x, TOP)
    c.setFont("Helvetica", 7)
    for d in range(1000, 1071, 10):
        c.drawString(14, float(y_of(d)) - 2.5, str(d))
    if header:
        c.setStrokeColorRGB(0.9, 0, 0); c.line(T1[0], 420, T1[1], 420)
        c.drawString(70, 428, "GR"); c.drawString(T1[0] + 1, 410, "0"); c.drawString(80, 410, "API"); c.drawString(T1[1] - 13, 410, "150")
        c.setStrokeColorRGB(0, 0, 0.9); c.line(T2[0], 420, T2[1], 420)
        c.drawString(185, 428, "RT"); c.drawString(T2[0] + 1, 410, "0.2"); c.drawString(190, 410, "ohm.m"); c.drawString(T2[1] - 13, 410, "200")


def _curve(c, xs, ys, rgb):
    c.setStrokeColorRGB(*rgb); c.setLineWidth(0.8)
    p = c.beginPath(); p.moveTo(xs[0], ys[0])
    for x, y in zip(xs[1:], ys[1:]):
        p.lineTo(x, y)
    c.drawPath(p, stroke=1, fill=0)


def _draw_vector_log(c):
    dd = np.linspace(D0, D1, 400)
    _frame_and_text(c)
    _curve(c, T1[0] + gr(dd) / 150 * (T1[1] - T1[0]), y_of(dd), (0.9, 0, 0))
    lx = (np.log10(rt(dd)) - np.log10(0.2)) / (np.log10(200) - np.log10(0.2))
    _curve(c, T2[0] + lx * (T2[1] - T2[0]), y_of(dd), (0, 0, 0.9))


def _rotated_pdf(path, turn, page_rotate=0):
    """The same log drawn turned ``turn`` degrees counter-clockwise on its sheet (a sideways log,
    through the page transform), optionally with a page /Rotate so a viewer shows it upright."""
    W, H = PAGE
    size, (tx, ty) = {0: (PAGE, (0, 0)), 90: ((H, W), (H, 0)), 180: (PAGE, (W, H)), 270: ((H, W), (0, W))}[turn]
    # reportlab takes the page size as shown (after /Rotate) and swaps it for the MediaBox
    c = canvas.Canvas(str(path), pagesize=size[::-1] if page_rotate % 180 else size)
    if page_rotate:
        c.setPageRotation(page_rotate)
    c.translate(tx, ty); c.rotate(turn)
    _draw_vector_log(c)
    c.showPage(); c.save()
    return path


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinds")
    dd = np.linspace(D0, D1, 400)
    # vector: frame, depth numbers, header and two curve paths
    c = canvas.Canvas(str(d / "vector.pdf"), pagesize=PAGE)
    _frame_and_text(c)
    _curve(c, T1[0] + gr(dd) / 150 * (T1[1] - T1[0]), y_of(dd), (0.9, 0, 0))
    lx = (np.log10(rt(dd)) - np.log10(0.2)) / (np.log10(200) - np.log10(0.2))
    _curve(c, T2[0] + lx * (T2[1] - T2[0]), y_of(dd), (0, 0, 0.9))
    c.showPage(); c.save()
    # scan: one page-sized picture, nothing else
    from PIL import Image
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((500, 300)) > 0.02).astype(np.uint8) * 255).convert("L")
    c = canvas.Canvas(str(d / "scan.pdf"), pagesize=PAGE)
    c.drawImage(ImageReader(img), 0, 0, width=PAGE[0], height=PAGE[1]); c.showPage(); c.save()
    # mixed: the same picture with the header and depth numbers as PDF text on top
    c = canvas.Canvas(str(d / "mixed.pdf"), pagesize=PAGE)
    c.drawImage(ImageReader(img), 0, 0, width=PAGE[0], height=PAGE[1])
    c.setFont("Helvetica", 7)
    for dpt in range(1000, 1071, 10):
        c.drawString(14, float(y_of(dpt)) - 2.5, str(dpt))
    c.drawString(70, 428, "GR"); c.drawString(T1[0] + 1, 410, "0"); c.drawString(80, 410, "API"); c.drawString(T1[1] - 13, 410, "150")
    c.showPage(); c.save()
    img.save(d / "scan.tif"); img.save(d / "scan.png"); img.convert("RGB").save(d / "scan.jpg")
    return d


@pytest.mark.parametrize("name", ["scan.tif", "scan.png", "scan.jpg"])
def test_images_are_scans(pdfs, name):
    k = detect_input_kind(pdfs / name)
    assert k.kind == "scan" and k.method == "trace" and k.evidence["size_px"] == [300, 500]


def test_detect_vector_scan_mixed(pdfs):
    v = detect_input_kind(pdfs / "vector.pdf")
    assert v.kind == "vector" and v.method == "vector"
    assert v.evidence["n_curve_paths"] >= 2 and v.evidence["text_chars"] > 20 and v.evidence["image_coverage"] == 0
    s = detect_input_kind(pdfs / "scan.pdf")
    assert s.kind == "scan" and s.evidence["image_coverage"] > 0.95 and s.evidence["text_chars"] == 0
    m = detect_input_kind(pdfs / "mixed.pdf")
    assert m.kind == "mixed" and m.method == "mixed" and m.evidence["image_coverage"] > 0.95 and m.evidence["text_chars"] > 20
    assert set(v.as_dict()) >= {"kind", "method", "method_label", "evidence", "notes"}
    with pytest.raises(ValueError):
        detect_input_kind(pdfs / "vector.pdf", page=3)


def test_vector_reads_exact_paths(pdfs, tmp_path):
    vp = read_vector_page(pdfs / "vector.pdf", 0)
    assert len(vp.borders) == 4 and vp.depth_fit is not None
    names = {h.mnemonic: h for h in vp.headers}
    assert names["GR"].left == 0 and names["GR"].right == 150 and names["GR"].unit == "API"
    assert names["RT"].kind == "log" and names["RT"].left == 0.2 and names["RT"].right == 200
    log = digitize_vector(pdfs / "vector.pdf", step=0.5)
    assert log.info["reader"] == "vector"
    by = {t.mnemonic: t for t in log.traces}
    assert by["GR"].info["matched_by"] in ("colour", "track")
    # every path point maps back to the drawn value (to PDF number precision): nothing traced or smoothed
    t = by["GR"]; ok = np.isfinite(t.depth)
    assert np.nanmax(np.abs(t.value[ok] - gr(t.depth[ok]))) < 0.2
    r = by["RT"]; ok = np.isfinite(r.depth)
    assert np.nanmax(np.abs(np.log10(r.value[ok]) - np.log10(rt(r.depth[ok])))) < 0.01
    # depth from the printed numbers: the path ends sit at 1000 and 1070 ft
    assert np.nanmin(t.depth) == pytest.approx(1000.0, abs=0.1) and np.nanmax(t.depth) == pytest.approx(1070.0, abs=0.1)
    assert 1000.0 <= log.depth[0] <= 1000.5 and np.allclose(np.diff(log.depth), 0.5)
    out = log.to_las(tmp_path / "v.las")
    from pylogdigitizer.las_export import read_las
    df = read_las(out).df()
    assert {"GR", "RT"} <= set(df.columns)
    assert np.allclose(df["GR"].values, gr(df.index.values), atol=0.3)


@pytest.mark.parametrize("step", [0.1, 0.25, 0.5, 2.0])
def test_resample_path_any_step(step):
    d = np.linspace(100, 200, 501); v = 3 * d
    g, out = resample_path(d, v, step)
    assert np.allclose(np.diff(g), step) and g[0] == pytest.approx(100) and g[-1] == pytest.approx(200)
    assert np.allclose(out, 3 * g)


def test_resample_path_breaks_and_order():
    d = np.r_[np.linspace(0, 10, 50), np.nan, np.linspace(30, 40, 50)]
    v = np.r_[np.ones(50), np.nan, 2 * np.ones(50)]
    g, out = resample_path(d[::-1], v[::-1], 0.5)          # path drawn bottom-up: order does not matter
    assert np.all(np.isnan(out[(g > 10.5) & (g < 29.5)]))   # the break between pieces stays null
    assert np.allclose(out[g <= 10], 1) and np.allclose(out[g >= 30], 2)
    with pytest.raises(ValueError):
        resample_path(d, v, 0)


def test_mixed_header_in_picture_pixels(pdfs):
    H = mixed_header(pdfs / "mixed.pdf")
    assert H["image_size"] == (300, 500)
    h = next(h for h in H["headers"] if h["mnemonic"] == "GR")
    assert (h["left"], h["right"], h["unit"]) == (0, 150, "API")
    # picture is 1 px per pt here, row 0 at the page top
    rows = dict((d, r) for r, d in H["depth_labels"])
    assert rows[1000.0] == pytest.approx(PAGE[1] - TOP, abs=1.5) and rows[1070.0] == pytest.approx(PAGE[1] - BOT, abs=1.5)


def test_vector_without_layout_falls_back_or_says_why(pdfs):
    with pytest.raises(ValueError):
        digitize_vector(pdfs / "scan.pdf", fallback=False)


cal = lambda d: 11 + 3 * np.sin((np.asarray(d) - D0) / 5.0)             # 6-16 in, dashed, track 1


def _shared_track_pdf(path):
    """A layout like many service-company prints: curves drawn as thousands of two-point segments,
    a dashed curve, two curves sharing a track told apart by colour and dash, the name above each
    scale line with the coloured bar in between, scale numbers of neighbouring tracks touching at
    the border, and the plot frame drawn as one joined outline."""
    dd = np.linspace(D0, D1, 400)
    c = canvas.Canvas(str(path), pagesize=PAGE)
    _frame_and_text(c, header=False)
    c.setLineWidth(0.5); c.setStrokeColorRGB(0, 0, 0)
    c.line(10, TOP, T2[1], TOP); c.line(10, BOT, T2[1], BOT)            # meets the borders: one outline
    c.setFont("Helvetica", 7)
    for name, unit, lo, hi, y, rgb, dash, x0, x1 in (
            ("GR", "API", "0", "150", 446, (0.9, 0, 0), None, *T1),
            ("CALI", "in", "6", "16", 424, (0, 0, 0.9), (4, 2), *T1),
            ("RT", "ohm.m", "0.2", "200", 446, (0, 0.6, 0), None, *T2)):
        c.setStrokeColorRGB(*rgb); c.setLineWidth(1.5); c.setDash(*(dash or ()))
        c.line(x0, y + 4, x1, y + 4); c.setDash()
        c.drawCentredString((x0 + x1) / 2, y + 8, name)
        c.drawString(x0 + 0.3, y - 6, lo); c.drawCentredString((x0 + x1) / 2, y - 6, unit); c.drawRightString(x1, y - 6, hi)
    def segments(xs, ys, rgb):                     # one path per two-point segment
        c.setStrokeColorRGB(*rgb); c.setLineWidth(0.8)
        for i in range(len(xs) - 1):
            c.line(xs[i], ys[i], xs[i + 1], ys[i + 1])
    segments(T1[0] + gr(dd) / 150 * (T1[1] - T1[0]), y_of(dd), (0.9, 0, 0))
    c.setDash(4, 2); _curve(c, T1[0] + (cal(dd) - 6) / 10 * (T1[1] - T1[0]), y_of(dd), (0, 0, 0.9)); c.setDash()
    lx = (np.log10(rt(dd)) - np.log10(0.2)) / (np.log10(200) - np.log10(0.2))
    segments(T2[0] + lx * (T2[1] - T2[0]), y_of(dd), (0, 0.6, 0))
    c.showPage(); c.save()
    return path


def test_segmented_dashed_and_shared_track_curves(tmp_path):
    from pylogdigitizer.pdf_vector.vector_log import chain_strokes
    from pylogdigitizer.pdf_vector.extract import Stroke
    segs = [Stroke(pts=np.array([[i, i * i], [i + 1, (i + 1) ** 2]], float), color=(1, 0, 0), width=1.0) for i in range(5)]
    segs.insert(2, Stroke(pts=np.array([[0, 0], [9, 9]], float), color=(0, 0, 1), width=1.0))   # another style between
    joined = chain_strokes(segs[::-1])                       # order and direction do not matter
    red = [s for s in joined if s.color == (1, 0, 0)]
    assert len(joined) == 2 and len(red) == 1 and len(red[0].pts) == 6
    f = _shared_track_pdf(tmp_path / "shared.pdf")
    vp = read_vector_page(f, 0)
    by = {h.mnemonic: h for h in vp.headers}
    assert set(by) == {"GR", "CALI", "RT"}
    assert (by["GR"].right, by["RT"].left, by["RT"].right) == (150, 0.2, 200)     # touching numbers kept apart
    assert by["CALI"].dash and not by["GR"].dash and by["RT"].kind == "log"
    log = digitize_vector(f, step=0.5)
    t = {x.mnemonic: x for x in log.traces}
    assert set(t) == {"GR", "CALI", "RT"} and log.info["reader"] == "vector"
    assert t["GR"].info["n_paths"] == 1 and t["GR"].info["matched_by"] == "colour"   # 399 segments, one path
    assert not any("no scale printed" in n for n in log.info["notes"])               # frame outline is not a curve
    for k, fn in (("GR", gr), ("CALI", cal)):
        ok = np.isfinite(t[k].depth)
        assert np.nanmax(np.abs(t[k].value[ok] - fn(t[k].depth[ok]))) < 0.2, k
    ok = np.isfinite(t["RT"].depth)
    assert np.nanmax(np.abs(np.log10(t["RT"].value[ok]) - np.log10(rt(t["RT"].depth[ok])))) < 0.01


def test_no_curves_error_says_why(pdfs):
    with pytest.raises(ValueError, match="no vector curves.*\\("):
        digitize_vector(pdfs / "scan.pdf")                     # fallback finds nothing either: say why


# ------------------------------------------------------------------ app import step
fastapi = pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def app_client(pdfs, tmp_path_factory):
    import shutil
    from fastapi.testclient import TestClient
    from pylogdigitizer.app import create_app
    root = tmp_path_factory.mktemp("vapp"); d = root / "example_data" / "VEC_1"; d.mkdir(parents=True)
    shutil.copy(pdfs / "vector.pdf", d / "vector.pdf"); shutil.copy(pdfs / "scan.tif", d / "scan.tif")
    app = create_app(str(root))
    return TestClient(app, base_url="http://127.0.0.1"), root


def _wait(c, r):
    assert r.status_code == 200, r.text
    jid = r.json()["job"]["id"]
    while (j := c.get(f"/shell/api/jobs/{jid}").json())["state"] == "running":
        time.sleep(0.05)
    return j


def test_app_detect_and_vector_open(app_client):
    c, root = app_client
    d = c.get("/app/api/detect", params={"scan": "example_data/VEC_1/vector.pdf"}).json()
    assert d["kind"] == "vector" and d["path"] == "example_data/VEC_1/vector.pdf"
    assert c.get("/app/api/detect", params={"scan": "example_data/VEC_1/scan.tif"}).json()["kind"] == "scan"
    assert c.get("/app/api/detect", params={"scan": "/etc/passwd"}).status_code == 400       # path limits kept
    j = _wait(c, c.post("/app/api/open", json=dict(scan="example_data/VEC_1/vector.pdf", method="auto")))
    assert j["state"] == "done", j
    sid = j["result"]["sid"]; assert j["result"]["input"] == "vector"
    st = c.get(f"/app/api/{sid}/state").json()
    assert st["input"]["used"] == "vector" and st["n_curves"] == 2 and st["traced"]
    names = {q["name"] for q in c.get(f"/app/api/{sid}/curves").json()["curves"]}
    assert names == {"GR", "RT"}
    logs = c.get(f"/app/api/{sid}/logs").json()                # Check and save: same plot data as for scans
    assert len(logs["tracks"]) == 2                              # each curve plotted in its own track
    gr_log = next(q for q in logs["curves"] if q["name"] == "GR")
    dd = np.array(gr_log["depth"]); vv = np.array(gr_log["value"], float)
    assert np.nanmax(np.abs(vv - gr(dd))) < 0.5
    r = _wait(c, c.post(f"/app/api/{sid}/trace", json={}))       # nothing to trace
    assert r["state"] == "error" and "tracing not needed" in r["error"]
    w = c.post(f"/app/api/{sid}/export/las", json={"step": 0.25}).json()
    assert w["written"][0].endswith(".las")
    # any typed step > 0 (validated), the same for scans (tests/test_app.py)
    r3 = c.post(f"/app/api/{sid}/export/las", json={"step": 0.3}); assert r3.status_code == 200, r3.text; w = r3.json()
    from pylogdigitizer.las_export import read_las
    las = read_las(root / w["written"][0]) if not w["written"][0].startswith("/") else read_las(w["written"][0])
    assert np.allclose(np.diff(las.df().index.values), 0.3)
    for bad in (0, -1, "abc", "nan", "inf", 1e-9):
        assert c.post(f"/app/api/{sid}/export/las", json={"step": bad}).status_code == 400, bad
    # audit2 hardening still in place
    bad = c.post("/app/api/open", json=dict(scan="example_data/VEC_1/vector.pdf"), headers={"origin": "http://evil.example"})
    assert bad.status_code == 403
    assert c.post("/app/api/open", json=dict(scan="example_data/VEC_1/scan.tif", method="vector")).status_code == 400
    assert c.post("/app/api/open", json=dict(scan="example_data/VEC_1/vector.pdf", method="nope")).status_code == 400
    # scanned / mixed PDFs are read from page 1 only
    assert c.post("/app/api/open", json=dict(scan="example_data/VEC_1/vector.pdf", method="trace", page=1)).status_code == 400


# ------------------------------------------------------------------ rotated / sideways vector pages
@pytest.mark.parametrize("turn,page_rotate", [(90, 0), (180, 0), (270, 0), (90, 270), (270, 90), (0, 90)])
def test_rotated_vector_pages_read_exactly(tmp_path, turn, page_rotate):
    pdf = _rotated_pdf(tmp_path / f"rot{turn}_{page_rotate}.pdf", turn, page_rotate)
    k = detect_input_kind(pdf)
    assert k.kind == "vector"
    vp = read_vector_page(pdf, 0)
    assert vp.rotation == turn                       # read in the frame where the log is upright
    assert (vp.width, vp.height) == pytest.approx(PAGE)
    assert len(vp.borders) == 4 and vp.depth_fit is not None
    names = {h.mnemonic: h for h in vp.headers}
    assert (names["GR"].left, names["GR"].right, names["GR"].unit) == (0, 150, "API")
    assert names["RT"].kind == "log"
    log = digitize_vector(pdf, step=0.5, fallback=False)  # the exact reader, not the older fallback
    assert log.info["reader"] == "vector"
    by = {t.mnemonic: t for t in log.traces}
    t = by["GR"]; ok = np.isfinite(t.depth)
    assert np.nanmax(np.abs(t.value[ok] - gr(t.depth[ok]))) < 0.2
    assert np.nanmin(t.depth) == pytest.approx(1000.0, abs=0.1) and np.nanmax(t.depth) == pytest.approx(1070.0, abs=0.1)
    r = by["RT"]; ok = np.isfinite(r.depth)
    assert np.nanmax(np.abs(np.log10(r.value[ok]) - np.log10(rt(r.depth[ok])))) < 0.01


@pytest.mark.parametrize("turn", [90, 270])
def test_rotated_page_drawn_upright(tmp_path, turn):
    from pylogdigitizer.pdf_vector.render import render_page
    pdf = _rotated_pdf(tmp_path / f"r{turn}.pdf", turn)
    vp = read_vector_page(pdf, 0)
    img = render_page(pdf, 0, scale=1.0, rotation=vp.rotation)
    assert img.shape[:2] == (500, 300)               # portrait again: header on top, depth down
    # the red GR path sits in track 1 of the upright drawing, where the reader put it
    red = (img[..., 0] > 180) & (img[..., 1] < 90) & (img[..., 2] < 90)
    cols = np.nonzero(red[int(PAGE[1] - TOP) + 5:int(PAGE[1] - BOT) - 5].any(0))[0]
    assert cols.min() >= T1[0] - 2 and cols.max() <= T1[1] + 2


def test_sideways_text_on_upright_frame_still_counts():
    from pylogdigitizer.pdf_vector.extract import TextRun, frame_transform, runs_in_frame
    M = frame_transform((0, 0, 500, 300), 90)
    # "1000" written upward (angle 90) at user (100, 50) -> in the turned frame it reads left to right
    r = TextRun(x=100, y=50, x1=100, text="1000", size=7, angle=90, ux1=100, uy1=64)
    (q,) = runs_in_frame([r], M, 90)
    assert q.text == "1000" and q.x1 - q.x == pytest.approx(14)
    # the same run seen from the upright (0) frame becomes a stand-in centred on the glyphs
    (q0,) = runs_in_frame([r], np.eye(3), 0)
    assert q0.x < 100 < q0.x1 + 1 and 50 < q0.y + 0.36 * 7 < 64


def test_app_opens_sideways_vector_page(app_client, tmp_path):
    c, root = app_client
    _rotated_pdf(root / "example_data" / "VEC_1" / "sideways.pdf", 90, 270)
    j = _wait(c, c.post("/app/api/open", json=dict(scan="example_data/VEC_1/sideways.pdf")))
    assert j["state"] == "done", j
    sid = j["result"]["sid"]
    st = c.get(f"/app/api/{sid}/state").json()
    assert st["input"]["used"] == "vector" and st["n_curves"] == 2
    assert (st["width"], st["height"]) == (900, 1500)               # drawn upright at 3 px per pt
    logs = c.get(f"/app/api/{sid}/logs").json()
    q = next(q for q in logs["curves"] if q["name"] == "GR")
    dd = np.array(q["depth"]); vv = np.array(q["value"], float)
    assert np.nanmax(np.abs(vv - gr(dd))) < 0.5
