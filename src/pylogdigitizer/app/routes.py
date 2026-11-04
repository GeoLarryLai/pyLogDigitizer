# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""routes: the pyLogDigitizer steps of the desktop app (/app/api/...).

The interactive editing routes are the existing editor's (``web_editor.app``), mounted at
/core (its API and static files only; the old editor page is not served) and sharing the same sessions, so there is one implementation of every digitizing action
(all logic in ``assisted_tracing`` / ``workflows``). This module only adds what the step-by-step
app needs on top: background jobs with progress, precomputation, undo snapshots, editable OCR
tables and the export guard (the editor's own write routes are blocked inside the app).

Steps: 1 load · 2 deskew · 3 section · 4 header OCR · 5 depth · 6 scale · 7 trace · 8 QC · 9 export
"""
from __future__ import annotations
from pathlib import Path
import csv, importlib.util, json, re, shutil, uuid

import numpy as np
from fastapi import Body, HTTPException, Request
from fastapi.responses import JSONResponse

from pylogdigitizer import curve_tracing as ct, depth_calibration as dc
from pylogdigitizer.assisted_tracing import AssistedSession, SessionError, UNITS
from pylogdigitizer.depth_calibration import CalibrationError
from pylogdigitizer.track_grid_detection import LayoutError
from pylogdigitizer.web_editor.app import _clean, _Entry, catalog, SETTING_KEYS
from pylogdigitizer.app.shell import ExportGuard, UndoStack

EXPECTED = (SessionError, CalibrationError, LayoutError, ValueError, FileNotFoundError)
SNAP_ATTRS = ("dm", "depth_anchors", "depth_checks", "depth_source", "section", "pan", "width", "_prob_map", "anchors",
              "excluded", "excluded_ids", "anchor_window_ft", "backup_intervals", "values", "wraps", "scale", "track",
              "trace_kw", "first_depth", "label_interval", "errors", "depth_section", "depth_sections",
              "strokes", "decisions", "change_log", "_spots", "grid_lattice", "_pan_raw", "grid_fit")
# editor routes that write files: blocked in the app (the Export step is the only writer)
BLOCKED = re.compile(r"^/core/api/sessions/[^/]+/(export_las|download_las|save_training|save_session|qc\.png)")
MUTATING = re.compile(r"^/core/api/sessions/([^/]+)/(section|depth|scale|trace|anchors.*|retrace_all|exclude.*|backup_interval.*)$")


# ---------------------------------------------------------------- optional features
def features(root: Path) -> dict:
    """What is installed. Missing optional features are greyed out in the page with ``hint``."""
    has = lambda m: importlib.util.find_spec(m) is not None
    from pylogdigitizer import ocr_engines
    eng = ocr_engines.status(); ocr = eng["default"] is not None
    model = next((p for p in (root / "models" / "kansas_gr_unet.pt", root / "models" / "kansas_kgs" / "unet.pt") if p.exists()), None)
    ml_pkgs = has("torch")                     # ml_models is plain torch (no smp needed)
    return dict(
        ocr=dict(ok=ocr, engine=eng["default"], engines={k: eng[k] for k in ocr_engines.ENGINES},
                 hint=None if ocr else "no OCR engine: " + ocr_engines.HINTS["rapidocr"]),
        ml=dict(ok=bool(ml_pkgs and model), model=str(model) if model else None,
                hint=None if (ml_pkgs and model) else ("needs: pip install torch" if not ml_pkgs
                                                       else "no U-Net checkpoint in models/ (train one in notebook 02)")),
        plot=dict(ok=has("matplotlib"), hint=None if has("matplotlib") else "QC plot PNG needs: pip install matplotlib"),
        backend=ct.backend())


# ---------------------------------------------------------------- undo snapshots
def snapshot(e: _Entry):
    s = e.s; d = {}
    for k in SNAP_ATTRS:
        v = getattr(s, k)
        d[k] = dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)
    d["T"] = dict(s.T) if s.T is not None else None
    d["app"] = json.loads(json.dumps(e.app.get("header_rows")))
    h = getattr(e, "header", None)          # header OCR words/proposals (small): word corrections are undoable
    d["header"] = (dict(h, tracks={k: dict(t, words=[dict(w) for w in t["words"]], curves=list(t["curves"])) for k, t in h["tracks"].items()},
                        curves=list(h.get("curves", []))) if isinstance(h, dict) and "tracks" in h else h)
    return d


def restore(e: _Entry, d) -> None:
    s = e.s
    for k in SNAP_ATTRS:
        setattr(s, k, d[k])
    s.T = d["T"]
    if d["app"] is not None:
        e.app["header_rows"] = d["app"]
    if "header" in d:
        e.header = d["header"]


def push_undo(e: _Entry, label: str) -> None:
    e.undo.push(label, snapshot(e))


# ---------------------------------------------------------------- header OCR rows (editable)
def header_rows(h: dict | None) -> list[dict]:
    rows = []
    for c in (h or {}).get("curves", []):
        v = lambda k: c[k]["value"]; bk = v("backup")
        rows.append(dict(id=uuid.uuid4().hex[:6], origin="ocr", track=c["track"], mnemonic=v("mnemonic"), unit=v("unit"),
                         left=v("scale_left"), right=v("scale_right"), type=v("scale_type") or "linear",
                         backup_left=bk[0] if bk else None, backup_right=bk[1] if bk else None,
                         review=[k for k in ("mnemonic", "unit", "scale_left", "scale_right", "scale_type", "backup") if c[k]["needs_review"]],
                         confidence=round(float(min(c[k]["confidence"] for k in ("scale_left", "scale_right"))), 2),
                         label=c.get("label", ""), points=(c.get("evidence") or {}).get("scale_points", [])))
    return rows


def wrap_side(left, right, bl, br) -> str:
    return "right" if abs(bl - right) <= abs(br - left) else "left"


def install(app, root: Path, runner) -> None:
    sessions: dict = app.state.editor.state.sessions
    guard_default = root / "outputs" / "app"

    def get(sid: str) -> _Entry:
        if sid not in sessions:
            raise HTTPException(404, f"unknown session {sid} (app restarted? load the scan again)")
        return sessions[sid]

    def locked(e: _Entry, fn):
        with e.lock:
            try:
                return fn(e.s)
            except EXPECTED as ex:
                raise HTTPException(400, f"{type(ex).__name__}: {ex}")

    def job(e: _Entry | None, title: str, fn) -> dict:
        def body(j):
            if e is None:
                return fn(j)
            with e.lock:
                return fn(j)
        j = runner.submit(title, body, sync=getattr(app.state, "sync_jobs", False))
        return dict(job=j.as_dict())

    def state(e: _Entry) -> dict:
        s = e.s; st = app.state.editor.state.state_fn(e)
        st.update(sid=e.sid, undo=e.undo.labels()[-8:], export=e.guard.state(), header_choice=e.app.get("header_choice"),
                  initial_scale=e.app["initial_scale"], name=e.app["name"])
        st.update(depth_sections=s.depth_sections, depth_section=s.depth_section)
        if s.dm is not None and s.depth_anchors:
            a = np.array(s.depth_anchors, float)
            st["residuals"] = _clean(np.c_[a[:, 1], dc.residuals(s.depth_anchors, s.dm)].tolist()) if len(a) > 2 else []
        return st

    # middleware: undo snapshots before editor edits; block editor file writes inside the app
    @app.middleware("http")
    async def guard_mw(request: Request, call_next):
        path = request.url.path
        if path.startswith("/core") and not path.startswith(("/core/api/", "/core/static/")):
            return JSONResponse({"detail": "not part of the app (the old editor UI: run `pylogdigitizer editor`)"}, status_code=404)
        if BLOCKED.match(path):
            return JSONResponse({"detail": "writing files is done in the Export step of the app"}, status_code=403)
        m = MUTATING.match(path)
        if m and request.method in ("POST", "DELETE") and m.group(1) in sessions:
            e = sessions[m.group(1)]
            with e.lock:
                push_undo(e, m.group(2).split("/")[0])
        return await call_next(request)

    @app.get("/app/api/info")
    def info():
        return dict(project=str(root), catalog=catalog(root), features=features(root))

    # ---------------- step 1: load (precomputes everything the later steps open with)
    @app.post("/app/api/open")
    def open_scan(body: dict = Body(...)):
        def p(rel):
            if not rel:
                return None
            q = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            if not q.exists():
                raise HTTPException(404, f"not found: {rel}")
            return q
        scan, las = p(body.get("scan")), p(body.get("las"))
        st = dict(body.get("settings") or {}); kw = {k: st[k] for k in SETTING_KEYS if k in st}
        for k in ("scale", "backup", "x_range", "rows"):
            if kw.get(k) is not None:
                kw[k] = tuple(kw[k])
        feats = features(root)
        if not feats["ocr"]["ok"]:
            kw["use_ocr"] = False
        curve = body.get("curve") or "GR"; well = body.get("well") or scan.parent.name

        def work(j):
            N = 7
            s = AssistedSession(scan, curve=curve, well=well, las_path=las, auto_layout=bool(body.get("auto_layout", True)), **kw)
            s.open(progress=lambda i, n, m: j.progress(i, N, m))
            sid = uuid.uuid4().hex[:10]; e = _Entry(s); e.sid = sid; e.undo = UndoStack()
            base = f"{s.well}_{Path(s.scan_path).stem}_{curve.replace(':', '_')}"
            e.guard = ExportGuard(guard_default / base)
            e.app = dict(name=base, initial_scale=dict(s.scale), header_rows=None, header_ocr=None, header_choice=None,
                         settings=st, label_values=None)
            j.progress(3, N, "conventional trace of the section")
            if s.pan is not None:
                s.auto_trace()
            j.progress(4, N, "depth-label values for review")
            if s.L is not None:
                e.app["label_values"] = list(s.L._ocr_values) if s.L._ocr_values is not None else None
            j.progress(5, N, "header OCR (track names, units, scales)" if feats["ocr"]["ok"] else "header OCR skipped (no OCR engine installed)")
            if feats["ocr"]["ok"] and s.L is not None:
                try:
                    from pylogdigitizer import header_ocr
                    e.header = header_ocr.propose_header(s.scan_path, layout=s.L)
                    e.app["header_ocr"] = header_rows(e.header); e.app["header_rows"] = json.loads(json.dumps(e.app["header_ocr"]))
                except Exception as ex:      # header OCR is advisory: never blocks the workflow
                    s.errors["header"] = f"{type(ex).__name__}: {ex}"
            j.progress(6, N, "image tiles")
            e.tiles.level(max(0, e.tiles.max_level - 6))
            sessions[sid] = e
            return dict(sid=sid)
        return job(None, "Load scan", work)

    @app.get("/app/api/{sid}/state")
    def get_state(sid: str):
        return state(get(sid))

    # ---------------- step 2/3 figures: deskew angles, followed boundaries, depth labels
    @app.get("/app/api/{sid}/layout")
    def layout(sid: str):
        e = get(sid); L = e.s.L
        if L is None:
            return dict(available=False, height=e.s.shape[0], width=e.s.shape[1])
        nb = len(L.skew_deg); H = L.ink.shape[0]
        vals = e.app.get("label_values")
        return _clean(dict(available=True, width=L.ink.shape[1], height=H, probe_row=L.probe_row, label_mode=L.label_mode,
                           skew=[[(i + 0.5) * H / nb, float(a)] for i, a in enumerate(L.skew_deg)],
                           edges={n: [[float(r), float(x)] for r, x in zip(L.centers, L.edges[:, j])] for j, n in enumerate(L.edge_names)},
                           labels=[[float(l["row"]), (vals[i] if vals and i < len(vals) else None)] for i, l in enumerate(L.labels)],
                           timings=L.timings, tracks=e.s.tracks()))

    # ---------------- step 3: section (job: re-traces so step 7 opens with a trace)
    @app.post("/app/api/{sid}/section")
    def section(sid: str, body: dict = Body(...)):
        e = get(sid); push_undo(e, "section")
        def work(j):
            s = e.s; j.progress(0, 2, "cutting the section")
            s.set_section(track=body.get("track"), box=tuple(body["box"]) if body.get("box") else None,
                          depth_range=tuple(body["depth_range"]) if body.get("depth_range") else None,
                          rows=tuple(body["rows"]) if body.get("rows") else None)
            j.progress(1, 2, "conventional trace of the new section"); s.auto_trace(); return dict(ok=True)
        return job(e, "Section", work)

    # ---------------- step 4: header OCR (editable table)
    @app.get("/app/api/{sid}/header")
    def header(sid: str):
        e = get(sid)
        return _clean(dict(rows=e.app["header_rows"] or [], ocr=e.app["header_ocr"] or [], choice=e.app.get("header_choice"),
                           runtime_s=(e.header or {}).get("runtime_s"), error=e.s.errors.get("header")))

    @app.post("/app/api/{sid}/header")
    def header_save(sid: str, body: dict = Body(...)):
        e = get(sid); push_undo(e, "header")
        rows = body.get("rows", [])
        for r in rows:
            for k in ("left", "right", "backup_left", "backup_right"):
                if r.get(k) in ("", None):
                    r[k] = None
                else:
                    r[k] = float(r[k])
            if r.get("type") not in ("linear", "log"):
                raise HTTPException(400, f"type must be linear or log (row {r.get('mnemonic')})")
        e.app["header_rows"] = rows
        if "choice" in body:
            e.app["header_choice"] = body["choice"]
        return header(sid)

    @app.get("/app/api/{sid}/header/words")
    def header_words(sid: str):
        """All OCR words of the header, per track: [{track, i, text, conf, x0, y0, x1, y1, corrected}]."""
        e = get(sid)
        out = [dict(track=nm, i=i, **{k: w.get(k) for k in ("text", "conf", "x0", "y0", "x1", "y1", "corrected", "ocr_text")})
               for nm, t in ((e.header or {}).get("tracks") or {}).items() for i, w in enumerate(t["words"])]
        return _clean(dict(words=out))

    @app.post("/app/api/{sid}/header/words")
    def header_words_edit(sid: str, body: dict = Body(...)):
        """{track, edits: [{i, text}]}: correct OCR words (typed text = full confidence), re-fit that track's
        curves/scales, refresh the header table and keep the (crop, text) pairs under <project>/ocr_corrections/."""
        e = get(sid)
        if not e.header:
            raise HTTPException(400, "no header OCR yet")
        push_undo(e, "header words")
        from pylogdigitizer import header_ocr
        try:
            header_ocr.apply_word_edits(e.header, str(body["track"]), list(body.get("edits") or []))
        except (KeyError, IndexError) as ex:
            e.undo.pop(); raise HTTPException(400, str(ex))
        e.app["header_ocr"] = header_rows(e.header); e.app["header_rows"] = json.loads(json.dumps(e.app["header_ocr"]))
        L = e.s.L
        if L is not None and body.get("save", True):
            try:
                header_ocr.save_word_corrections(root / "ocr_corrections", L.ink, [w for w in e.header["tracks"][body["track"]]["words"]
                                                 if w.get("corrected") and not w.get("saved")], scan=str(e.s.scan_path), offset=(L.x_offset, L.y_offset))
                for w in e.header["tracks"][body["track"]]["words"]:
                    if w.get("corrected"):
                        w["saved"] = True
            except Exception as ex:          # keeping crops is best effort; never blocks the edit
                e.s.errors["ocr_corrections"] = f"{type(ex).__name__}: {ex}"
        return header(sid)

    @app.post("/app/api/{sid}/header/revert")
    def header_revert(sid: str):
        e = get(sid); push_undo(e, "header revert")
        e.app["header_rows"] = json.loads(json.dumps(e.app["header_ocr"] or [])); e.app["header_choice"] = None
        return header(sid)

    @app.post("/app/api/ocr_engine")
    def ocr_engine(body: dict = Body(...)):
        """{engine: 'auto'|'rapidocr'|'none'} - engine for header OCR and depth labels from now on
        (press Re-run OCR / Re-run automatic to re-read). A missing engine is refused with its install hint."""
        from pylogdigitizer import ocr_engines
        eng = body.get("engine", "auto")
        try:
            if eng not in (None, "auto"):
                ocr_engines.resolve(eng)
            ocr_engines.set_default(eng)
        except (RuntimeError, ValueError) as ex:
            raise HTTPException(400, str(ex))
        return features(root)

    @app.post("/app/api/{sid}/header/rerun")
    def header_rerun(sid: str):
        e = get(sid)
        if not features(root)["ocr"]["ok"]:
            raise HTTPException(400, features(root)["ocr"]["hint"])
        def work(j):
            from pylogdigitizer import header_ocr
            j.progress(0, 1, "reading the track headers")
            e.header = header_ocr.propose_header(e.s.scan_path, layout=e.s.L)
            e.app["header_ocr"] = header_rows(e.header); e.app["header_rows"] = json.loads(json.dumps(e.app["header_ocr"]))
            return dict(n=len(e.app["header_rows"]))
        return job(e, "Header OCR", work)

    # ---------------- step 5a: which log section (only when the sheet holds several and the header does not say)
    @app.post("/app/api/{sid}/depth_section")
    def depth_section(sid: str, body: dict = Body(...)):
        """{index: n} - use log section n for the depth scale, then re-cut and re-trace the track."""
        e = get(sid); push_undo(e, "depth section")
        def work(j):
            s = e.s; j.progress(0, 2, "depth calibration of the chosen section")
            s.set_depth_section(int(body["index"]))
            e.app["label_values"] = list(s.L._ocr_values) if s.L._ocr_values is not None else None
            if s.track in s.L.track_names and (s.section is None or s.section["kind"] == "track"):
                j.progress(1, 2, "tracing the chosen section")
                s.set_section(track=s.section["track"] if s.section else s.track); s.errors.pop("section", None); s.auto_trace()
            return dict(ok=True)
        return job(e, "Depth section", work)

    # ---------------- step 5: depth calibration (job: re-calibrates then re-traces)
    @app.post("/app/api/{sid}/depth")
    def depth(sid: str, body: dict = Body(...)):
        """{mode:'auto', first_depth, label_interval} | {mode:'labels', labels:[[row, depth],..]} (edited OCR values,
        piecewise linear) | {mode:'reset'} (automatic calibration as on load)."""
        e = get(sid); push_undo(e, "depth")
        mode = body.get("mode", "auto")
        def work(j):
            s = e.s; j.progress(0, 2, "depth calibration")
            if mode == "labels":
                pts = [(float(r), float(d)) for r, d in body.get("labels", []) if d is not None and d != ""]
                s.set_depth_anchors(pts)
                e.app["label_values"] = None if s.L is None else [dict((round(r), d) for r, d in pts).get(round(l["row"])) for l in s.L.labels]
                if body.get("extra_labels"):
                    e.app["extra_labels"] = body["extra_labels"]
            elif mode == "reset":
                st = e.app["settings"]
                s.set_depth_labels(st.get("first_depth"), st.get("label_interval"))
                e.app["label_values"] = list(s.L._ocr_values) if s.L is not None and s.L._ocr_values is not None else None
            else:
                fd, li = body.get("first_depth"), body.get("label_interval")
                s.set_depth_labels(None if fd in ("", None) else float(fd), None if li in ("", None) else float(li))
                e.app["label_values"] = list(s.L._ocr_values) if s.L is not None and s.L._ocr_values is not None else None
            if s.pan is not None and s.section and s.section["kind"] == "track" and s.L is not None:
                j.progress(1, 2, "re-tracing with the new depth scale")
                s.set_section(track=s.section["track"]); s.auto_trace()
            elif s.pan is not None:
                j.progress(1, 2, "re-tracing"); s.auto_trace()
            return dict(ok=True)
        return job(e, "Depth calibration", work)

    # ---------------- calibration editing: track edges, manual scale, depth residuals
    @app.get("/app/api/{sid}/calib")
    def calib(sid: str):
        e = get(sid)
        def f(s):
            return dict(section=s.section, width=s.width, scale=s.scale, edge_edits=getattr(s, "edge_edits", []),
                        grid_x=s.grid_lines() if s.pan is not None else [], depth=s.depth_residuals())
        return _clean(locked(e, f))

    @app.post("/app/api/{sid}/calib/edges")
    def calib_edges(sid: str, body: dict = Body(...)):
        """{left_dx, right_dx}: move the track boundaries (sheet px), re-cut and re-trace."""
        e = get(sid); push_undo(e, "track edges")
        def work(j):
            j.progress(0, 1, "re-cutting the track"); e.s.adjust_track_edges(float(body.get("left_dx", 0)), float(body.get("right_dx", 0)))
            e.s.auto_trace(); return dict(ok=True)
        return job(e, "Track edges", work)

    @app.post("/app/api/{sid}/calib/scale2")
    def calib_scale2(sid: str, body: dict = Body(...)):
        """{x1, v1, x2, v2, type}: two panel columns (e.g. grid lines) with known values -> track scale."""
        e = get(sid); push_undo(e, "scale")
        l, r = locked(e, lambda s: s.scale_from_points(float(body["x1"]), float(body["v1"]), float(body["x2"]), float(body["v2"]),
                                                       body.get("type", "linear")))
        return _clean(dict(left=l, right=r))

    # ---------------- grid: straighten (rectify) the section, edit the lattice, value every grid line
    def _grid(s, window=None):
        lat = getattr(s, "grid_lattice", None)
        return dict(rectified=lat is not None, lattice=lat.to_dict() if lat is not None else None,
                    fit=getattr(s, "grid_fit", None), width=s.width, rows=s.pan.shape[0] if s.pan is not None else 0)

    @app.get("/app/api/{sid}/grid")
    def grid_get(sid: str):
        return _clean(locked(get(sid), _grid))

    @app.post("/app/api/{sid}/grid/detect")
    def grid_detect(sid: str, body: dict = Body(default={})):
        """Detect and straighten the grid of the current section, then re-trace (job)."""
        e = get(sid); push_undo(e, "straighten grid")
        def work(j):
            j.progress(0, 2, "finding grid lines"); info = e.s.rectify_grid(lattice=body.get("lattice"))
            j.progress(1, 2, "tracing in the straightened panel"); e.s.auto_trace(); return _clean(info)
        return job(e, "Straighten grid", work)

    @app.post("/app/api/{sid}/grid/off")
    def grid_off(sid: str):
        e = get(sid); push_undo(e, "unstraighten grid")
        def f(s):
            s.unrectify_grid(); s.auto_trace(); return _grid(s)
        return _clean(locked(e, f))

    @app.post("/app/api/{sid}/grid/preview")
    def grid_preview(sid: str, body: dict = Body(...)):
        """{i, j, x, y, radius, rows:[r0, r1]}: move lattice node (i, j) to raw-panel (x, y) and return the
        straightened window as a PNG data URL (nothing is committed). Without i/j: the current lattice."""
        import base64, io
        from PIL import Image
        from pylogdigitizer import grid_rectify as gr
        e = get(sid)
        def f(s):
            lat = getattr(s, "grid_lattice", None)
            if lat is None:
                raise HTTPException(400, "straighten the grid first")
            if body.get("i") is not None:
                lat = lat.move_node(int(body["i"]), int(body["j"]), float(body["x"]), float(body["y"]), int(body.get("radius", 1)))
            raw = s._pan_raw; r0, r1 = body.get("rows") or (0, min(raw.shape[0], 1500))
            r0, r1 = int(max(0, r0)), int(min(raw.shape[0], r1))
            im = gr.rectify_image(raw, lat, raw.shape, window=(r0, r1, 0, raw.shape[1]))
            buf = io.BytesIO(); Image.fromarray(np.where(im, 0, 255).astype(np.uint8)).save(buf, "PNG")
            return dict(png="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), rows=[r0, r1],
                        lattice=lat.to_dict())
        return _clean(locked(e, f))

    @app.post("/app/api/{sid}/grid/node")
    def grid_node(sid: str, body: dict = Body(...)):
        """{i, j, x, y, radius}: commit a node drag (raw-panel px), re-cut and re-trace (job)."""
        e = get(sid); push_undo(e, "grid node")
        def work(j):
            lat = e.s.grid_lattice
            if lat is None:
                raise HTTPException(400, "straighten the grid first")
            lat = lat.move_node(int(body["i"]), int(body["j"]), float(body["x"]), float(body["y"]), int(body.get("radius", 1)))
            e.s.rectify_grid(lattice=lat); e.s.auto_trace(); return dict(ok=True)
        return job(e, "Grid node", work)

    @app.post("/app/api/{sid}/grid/values")
    def grid_values(sid: str, body: dict = Body(default={})):
        """{labels:[[panel_x, value], ..], kind, apply}: value every grid line, flag labels that disagree,
        and (apply) set the track scale from the fit. No labels: values implied by the current scale."""
        e = get(sid)
        if body.get("apply", True):
            push_undo(e, "grid values")
        labels, kind = body.get("labels"), body.get("kind")
        src = "labels" if labels else "scale"
        if not labels:   # the chosen header row's OCR'd scale numbers, placed by their x in the track
            row = next((r for r in e.app.get("header_rows") or [] if r.get("id") == e.app.get("header_choice")), None)
            pts = (row or {}).get("points") or []
            if len(pts) >= 2:
                w = e.s.width
                labels = [(f * w - 0.5, v) for f, v in pts]; kind = kind or row.get("type"); src = "header"
        out = locked(e, lambda s: s.grid_values(labels, kind, bool(body.get("apply", True))))
        return _clean(dict(out, source=src))

    # ---------------- step 7: trace modes (job)
    @app.post("/app/api/{sid}/trace")
    def trace(sid: str, body: dict = Body(default={})):
        e = get(sid); mode = body.get("mode", "conventional"); push_undo(e, f"trace {mode}")
        if mode not in ("conventional", "ml", "blend"):
            raise HTTPException(400, "mode must be conventional, ml or blend")
        f = features(root)
        if mode != "conventional" and not f["ml"]["ok"]:
            raise HTTPException(400, f["ml"]["hint"])
        def work(j):
            s = e.s
            if mode == "conventional" and s.trace_kw.get("cost_mode", "conventional") == "conventional" and not body.get("force"):
                j.progress(0, 1, "conventional trace"); s.auto_trace(anchors_panel=s.anchors or None, excluded=s.excluded or None, strokes=s.strokes or None)
            else:
                j.progress(0, 1, f"{mode} cost map + trace" + (" (U-Net)" if mode != "conventional" else ""))
                s.set_cost_mode(mode, ml_model=f["ml"]["model"] if mode != "conventional" else None)
            return dict(ok=True)
        return job(e, "Trace", work)

    @app.post("/app/api/{sid}/retrace_all")
    def retrace(sid: str):
        e = get(sid); push_undo(e, "re-trace all")
        return job(e, "Re-trace", lambda j: (j.progress(0, 1, "global re-trace with all anchors"), e.s.retrace_all(), dict(ok=True))[-1])

    @app.post("/app/api/{sid}/trace/reset")
    def trace_reset(sid: str):
        e = get(sid); push_undo(e, "reset trace")
        def work(j):
            s = e.s; s.backup_intervals = []; s.trace_kw.pop("cost_mode", None); s._prob_map = None
            j.progress(0, 1, "fresh conventional trace"); s.auto_trace(); return dict(ok=True)
        return job(e, "Reset trace", work)

    # ---------------- step 7: guided fixing (click where wrong, guide strokes, check spots, progress)
    import time as _time

    def guided(e: _Entry, label: str, fn) -> dict:
        """Run one guided edit (app undo snapshot first) and return what the page needs to redraw."""
        push_undo(e, label); t = _time.perf_counter()
        try:
            out = locked(e, fn)
        except Exception:
            e.undo.pop()                      # a refused edit leaves nothing to undo
            raise
        s = e.s
        return _clean(dict(result=list(out) if isinstance(out, tuple) else None, ms=round(1000 * (_time.perf_counter() - t)),
                           retrace=getattr(s, "last_retrace", None), progress=s.progress(), n_anchors=len(s.anchors),
                           n_strokes=len(s.strokes), undo=e.undo.labels()[-8:]))

    @app.post("/app/api/{sid}/fix/pin")
    def fix_pin(sid: str, body: dict = Body(...)):
        """{row, x}: the trace must pass here (click on the real curve where the trace is wrong)."""
        e = get(sid)
        return guided(e, "pin", lambda s: s.pin(float(body["row"]), float(body["x"]), snap=bool(body.get("snap", True))))

    @app.post("/app/api/{sid}/fix/move")
    def fix_move(sid: str, body: dict = Body(...)):
        e = get(sid)
        return guided(e, "move pin", lambda s: s.move_pin(int(body["index"]), float(body["row"]), float(body["x"])))

    @app.post("/app/api/{sid}/fix/unpin")
    def fix_unpin(sid: str, body: dict = Body(...)):
        e = get(sid)
        return guided(e, "remove pin", lambda s: s.unpin(int(body["index"])))

    @app.post("/app/api/{sid}/fix/stroke")
    def fix_stroke(sid: str, body: dict = Body(...)):
        """{points: [[row, x], ..]}: guide stroke dragged along a hard stretch."""
        e = get(sid)
        return guided(e, "guide stroke", lambda s: s.add_stroke([(float(r), float(x)) for r, x in body["points"]]))

    @app.post("/app/api/{sid}/fix/circle")
    def fix_circle(sid: str, body: dict = Body(...)):
        """{row, x, radius, through?}: re-trace only inside the cursor circle; nothing outside moves."""
        e = get(sid)
        return guided(e, "circle re-trace", lambda s: s.circle_retrace(float(body["row"]), float(body["x"]),
                                                                      radius_px=float(body.get("radius", 60)), through=bool(body.get("through", False))))

    @app.post("/app/api/{sid}/fix/unstroke")
    def fix_unstroke(sid: str, body: dict = Body(...)):
        e = get(sid)
        return guided(e, "remove stroke", lambda s: s.remove_stroke(int(body["index"])))

    @app.get("/app/api/{sid}/fix/progress")
    def fix_progress(sid: str):
        e = get(sid)
        return _clean(locked(e, lambda s: s.progress()))

    @app.get("/app/api/{sid}/checks")
    def checks(sid: str):
        """Spots to confirm (crossovers / wrap-arounds) + where the trace disagrees with itself."""
        e = get(sid)
        def f(s):
            if s.T is None:
                raise SessionError("no trace yet")
            sp = s.check_spots(); dis = s.disagreement; rows = s.T["rows"]
            from pylogdigitizer import scale_wraparound as sw
            segs = [[float(rows[a]), float(rows[min(b, len(rows) - 1)])] for a, b in sw.detect_wraps(dis > 0, 1)]
            return dict(spots=sp, disagree=segs, decisions=s.decisions)
        return _clean(locked(e, f))

    @app.post("/app/api/{sid}/checks/answer")
    def checks_answer(sid: str, body: dict = Body(...)):
        """{id, choice}: choice = a candidate id ('left'/'right'/'off'/'inside') or 'skip'."""
        e = get(sid)
        return guided(e, "answer", lambda s: s.answer_spot(int(body["id"]), str(body["choice"])))

    # ---------------- undo
    @app.post("/app/api/{sid}/undo")
    def undo(sid: str):
        e = get(sid)
        with e.lock:
            it = e.undo.pop()
            if it is None:
                raise HTTPException(400, "nothing to undo")
            restore(e, it[1])
        return dict(undone=it[0], **state(e))

    # ---------------- step 9: export (the only writer)
    @app.post("/app/api/{sid}/export/folder")
    def export_folder(sid: str, body: dict = Body(...)):
        e = get(sid); e.guard.set_folder(body.get("path") or guard_default / e.app["name"]); return e.guard.state()

    @app.post("/app/api/{sid}/export/{kind}")
    def export(sid: str, kind: str):
        e = get(sid); s = e.s; name = e.app["name"]; out = []
        kinds = ["las", "csv", "qc_png", "session"] if kind == "all" else [kind]
        f = features(root)
        for k in kinds:
            if k == "las":
                out.append(locked(e, lambda s: e.guard.write(name + ".las", lambda p: s.export_las(p, srvc="pyLogDigitizer app"))))
            elif k == "csv":
                def wcsv(p):
                    g, v = s.resampled(); c = s.curve.split(":")[0]
                    with open(p, "w", newline="") as fh:
                        w = csv.writer(fh); w.writerow(["DEPT", c])
                        for a, b in zip(g, v):
                            w.writerow([repr(float(a)), repr(float(b))])
                    return p
                out.append(locked(e, lambda s: e.guard.write(name + ".csv", wcsv)))
            elif k == "qc_png":
                if not f["plot"]["ok"] or locked(e, lambda s: s.qc()) is None:
                    if kind == "all":
                        continue
                    raise HTTPException(400, f["plot"]["hint"] or "no paired LAS for this scan: nothing to compare")
                out.append(locked(e, lambda s: e.guard.write(name + "_qc.png", lambda p: (s.qc(str(p)), p)[1])))
            elif k == "session":
                def wses(p):
                    d = s.session_dict()
                    for kk in ("scan", "las"):
                        if d.get(kk) and Path(d[kk]).resolve().is_relative_to(root):
                            d[kk] = Path(d[kk]).resolve().relative_to(root).as_posix()
                    d["header_rows"] = e.app.get("header_rows")
                    Path(p).write_text(json.dumps(_clean(d), indent=1)); return p
                out.append(locked(e, lambda s: e.guard.write(name + "_session.json", wses)))
            elif k == "training":      # correction sample for ML training (project folder, not in "all")
                out.append(locked(e, lambda s: s.save_training(root / "training_data" / "corrections")))
            else:
                raise HTTPException(404, f"unknown export {k}")
        return dict(e.guard.state(), written=[str(p) for p in out], all_written=e.guard.state()["written"])
