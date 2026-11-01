# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""app: FastAPI core API of the editor, served by ``pylogdigitizer-app`` under /core (no page of its own). Thin layer over assisted_tracing.AssistedSession.

Routes (JSON unless noted; {sid} = session id from POST /api/sessions):
  GET  /api/catalog                        scans in the project (example_data + wells config presets)
  POST /api/sessions                       open a scan {scan, curve, well, las, settings{...}, auto_layout}
  GET  /api/sessions/{sid}                 state (tracks, calibration, scale, section, errors, metrics)
  GET  /api/sessions/{sid}/image.dzi       Deep Zoom descriptor (XML); tiles: /image_files/{level}/{col}_{row}.png
  POST /api/sessions/{sid}/section         {track} or {box:[x0,x1]}, optional {rows:[r0,r1]} / {depth_range:[d0,d1]}
  POST /api/sessions/{sid}/depth           {anchors:[[row,depth],..]} or {first_depth, label_interval}
  POST /api/sessions/{sid}/scale           {left,right,type,backup,wrap_side,use_hatch}
  POST /api/sessions/{sid}/trace           {seed:[row,x]} optional - full conventional trace
  POST /api/sessions/{sid}/anchors         {row,x,snap,window_ft} - add anchor + local re-trace
  POST /api/sessions/{sid}/retrace_all     global re-trace with all anchors / exclusions
  POST /api/sessions/{sid}/exclude         {row,x} "not this curve": drop that stroke, re-trace its rows
  POST /api/sessions/{sid}/exclude/clear   undo all exclusions
  POST /api/sessions/{sid}/backup_interval {rows:[r0,r1]} read these rows on the backup scale (missed wrap); /clear
  DELETE /api/sessions/{sid}/anchors/{i}   remove anchor i + local re-trace
  GET  /api/sessions/{sid}/overlay         trace polyline / anchors (?r0=&r1=)
  GET  /api/sessions/{sid}/qc              metrics vs paired LAS; /qc.png = QC plot (PNG)
  POST /api/sessions/{sid}/export_las      write LAS -> outputs/editor/  ;  GET /download_las
  POST /api/sessions/{sid}/save_training   write a correction sample -> training_data/corrections/
  POST /api/sessions/{sid}/save_session    write replayable session JSON -> outputs/editor/sessions/
"""
from __future__ import annotations
from pathlib import Path
import threading, time, uuid

import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pylogdigitizer.assisted_tracing import AssistedSession, SessionError
from pylogdigitizer.depth_calibration import CalibrationError, ocr_available as header_ocr_available
from pylogdigitizer.track_grid_detection import LayoutError
from pylogdigitizer.web_editor.tiles import TilePyramid
from pylogdigitizer import curve_tracing as ct

STATIC = Path(__file__).parent / "static"
SETTING_KEYS = ("track", "scale", "scale_type", "backup", "wrap_side", "use_hatch", "first_depth", "label_interval",
                "use_ocr", "x_range", "rows", "trace_kw")


def _clean(o):
    """JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def catalog(root: Path) -> list[dict]:
    """Scans under example_data/: presets from dataset_profiles/*_wells.yaml, then other TIFFs."""
    import yaml
    out, seen = [], set()
    for cfgp in sorted((root / "dataset_profiles").glob("*_wells.yaml")):
        cfg = yaml.safe_load(cfgp.read_text()); data = cfg.get("data_dir", "example_data")
        for w in cfg.get("wells", []):
            for r in w.get("runs", []):
                st = dict(cfg.get("defaults", {})); st.update({k: r[k] for k in SETTING_KEYS if k in r})
                scan = f"{data}/{w['dir']}/{r['scan']}"
                out.append(dict(id=f"{w['name']}|{r['scan']}|{r['curve']}", well=w["name"], scan=scan, curve=r["curve"],
                                las=f"{data}/{w['dir']}/{w['las']}" if w.get("las") else None, settings=st,
                                scale_source=r.get("scale_source", ""), profile=cfg.get("profile", "")))
                seen.add(scan)
    ex = root / "example_data"
    for p in sorted(ex.rglob("*.tif")) if ex.exists() else []:
        rel = p.relative_to(root).as_posix()
        if rel not in seen:
            out.append(dict(id=rel, well=p.parent.name, scan=rel, curve="GR", las=None, settings={}, scale_source="", profile=""))
    return out


class _Entry:
    def __init__(self, s: AssistedSession):
        self.s = s; self.lock = threading.Lock(); self.tiles = TilePyramid(s.ink); self.t = time.time()
        self.header: dict | None = None     # cached header_ocr proposals


def build_app(project_dir: str = ".") -> FastAPI:
    root = Path(project_dir).resolve()
    out_dir = root / "outputs" / "editor"; train_dir = root / "training_data" / "corrections"
    app = FastAPI(title="pyLogDigitizer editor", docs_url="/api/docs", redoc_url=None)
    app.state.root, app.state.sessions = root, {}
    sessions: dict[str, _Entry] = app.state.sessions

    def inside(rel: str | None) -> Path | None:
        if not rel:
            return None
        p = (root / rel).resolve()
        if root not in p.parents and p != root:
            raise HTTPException(400, f"path outside the project: {rel}")
        if not p.exists():
            raise HTTPException(404, f"not found: {rel}")
        return p

    def get(sid: str) -> _Entry:
        if sid not in sessions:
            raise HTTPException(404, f"unknown session {sid} (server restarted? open the scan again)")
        return sessions[sid]

    def run(e: _Entry, fn):
        with e.lock:
            try:
                return fn(e.s)
            except (SessionError, CalibrationError, LayoutError, ValueError) as ex:
                raise HTTPException(400, f"{type(ex).__name__}: {ex}")

    def state(e: _Entry, with_qc: bool = True) -> dict:
        s = e.s
        st = dict(scan=s.scan_path, well=s.well, curve=s.curve, las=s.las_path, width=s.shape[1], height=s.shape[0],
                  tracks=s.tracks(), errors=s.errors, section=s.section, calibration=s.calibration(),
                  depth_checks={k: v for k, v in s.depth_checks.items() if isinstance(v, (int, float, str, bool))},
                  n_anchors=len(s.anchors), n_excluded=len(s.excluded), n_strokes=len(getattr(s, "strokes", [])), n_decisions=len(getattr(s, "decisions", [])), n_backup_intervals=len(s.backup_intervals), traced=s.T is not None, backend=ct.backend(),
                  labels=[float(l["row"]) for l in (s.L.labels if s.L else [])])
        if s.T is not None and s.dm is not None:
            d = s.dm(np.array(s.section["rows"], float)); st["depth_top_base"] = [float(d[0]), float(d[1])]
            st["low_conf_intervals"] = len(ct.low_confidence_intervals(s.T["conf"]))
            st["n_wraps"] = len(s.wraps)
        if with_qc and s.T is not None:
            q = s.qc(); st["qc"] = q["metrics"] if q else None
        return _clean(st)

    app.state.state_fn = state          # reused by the desktop app (pylogdigitizer.app)

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/api/catalog")
    def api_catalog():
        return catalog(root)

    @app.post("/api/sessions")
    def open_session(body: dict = Body(...)):
        scan = inside(body.get("scan")); las = inside(body.get("las")) if body.get("las") else None
        st = dict(body.get("settings") or {})
        kw = {k: st[k] for k in SETTING_KEYS if k in st}
        for k in ("scale", "backup", "x_range", "rows"):
            if k in kw and kw[k] is not None:
                kw[k] = tuple(kw[k])
        s = AssistedSession(scan, curve=body.get("curve", "GR"), well=body.get("well", ""), las_path=las,
                            auto_layout=bool(body.get("auto_layout", True)), **kw)
        try:
            s.open()
        except Exception as ex:   # unreadable file etc.
            raise HTTPException(400, f"{type(ex).__name__}: {ex}")
        sid = uuid.uuid4().hex[:10]; sessions[sid] = e = _Entry(s)
        if body.get("trace", True) and s.pan is not None:
            run(e, lambda s: s.auto_trace())
        return dict(sid=sid, **state(e))

    @app.get("/api/sessions/{sid}")
    def get_state(sid: str):
        return state(get(sid))

    @app.get("/api/sessions/{sid}/image.dzi")
    def dzi(sid: str):
        return Response(get(sid).tiles.dzi_xml(), media_type="application/xml")

    @app.get("/api/sessions/{sid}/image_files/{level}/{name}")
    def tile(sid: str, level: int, name: str):
        e = get(sid)
        try:
            c, r = name.split(".")[0].split("_")
            return Response(e.tiles.tile_png(level, int(c), int(r)), media_type="image/png",
                            headers={"Cache-Control": "max-age=3600"})
        except (KeyError, ValueError):
            raise HTTPException(404, "no such tile")

    @app.post("/api/sessions/{sid}/section")
    def section(sid: str, body: dict = Body(...)):
        e = get(sid)
        def f(s):
            s.set_section(track=body.get("track"), box=tuple(body["box"]) if body.get("box") else None,
                          depth_range=tuple(body["depth_range"]) if body.get("depth_range") else None,
                          rows=tuple(body["rows"]) if body.get("rows") else None)
            if body.get("trace", True):
                s.auto_trace(seed=tuple(body["seed"]) if body.get("seed") else None)
        run(e, f); return state(e)

    @app.post("/api/sessions/{sid}/depth")
    def depth(sid: str, body: dict = Body(...)):
        e = get(sid)
        if body.get("anchors"):
            run(e, lambda s: s.set_depth_anchors(body["anchors"]))
        else:
            run(e, lambda s: s.set_depth_labels(body.get("first_depth"), body.get("label_interval"), body.get("use_ocr")))
        return state(e)

    @app.post("/api/sessions/{sid}/scale")
    def scale(sid: str, body: dict = Body(...)):
        e = get(sid)
        run(e, lambda s: s.set_scale(float(body["left"]), float(body["right"]), body.get("type", "linear"),
                                     body.get("backup"), body.get("wrap_side", "right"), body.get("use_hatch")))
        return state(e)

    @app.get("/api/sessions/{sid}/proposed_header")
    def proposed_header(sid: str):
        """header_ocr proposals for this scan (cached). Proposals only: the front end fills the
        scale form with a chosen proposal, applied with "Apply scale"."""
        e = get(sid)
        if e.header is None:
            from pylogdigitizer import header_ocr
            if not header_ocr_available():
                raise HTTPException(400, "no OCR engine installed (pip install rapidocr_onnxruntime)")
            def f(s):
                return header_ocr.propose_header(s.scan_path, layout=s.L)
            e.header = run(e, f)
        h = e.header
        return _clean(dict(header_rows=h["header_rows"], runtime_s=h["runtime_s"], backend=h["backend"],
                           status=h["status"], curves=h["curves"]))

    @app.post("/api/sessions/{sid}/trace")
    def trace(sid: str, body: dict = Body(default={})):
        e = get(sid)
        run(e, lambda s: s.auto_trace(seed=tuple(body["seed"]) if body.get("seed") else None))
        return state(e)

    @app.post("/api/sessions/{sid}/anchors")
    def add_anchor(sid: str, body: dict = Body(...)):
        e = get(sid)
        a = run(e, lambda s: s.add_anchor(float(body["row"]), float(body["x"]), bool(body.get("snap", True)),
                                          float(body.get("window_ft", 150.0)), bool(body.get("panel_x", False))))
        return dict(anchor=list(a), retrace=getattr(e.s, "last_retrace", None), **state(e))

    @app.delete("/api/sessions/{sid}/anchors/{i}")
    def del_anchor(sid: str, i: int):
        e = get(sid)
        if not 0 <= i < len(e.s.anchors):
            raise HTTPException(404, f"no anchor {i}")
        run(e, lambda s: s.remove_anchor(i)); return state(e)

    @app.post("/api/sessions/{sid}/anchors/clear")
    def clear_anchors(sid: str):
        e = get(sid); run(e, lambda s: s.clear_anchors()); return state(e)

    @app.post("/api/sessions/{sid}/retrace_all")
    def retrace_all(sid: str):
        e = get(sid); run(e, lambda s: s.retrace_all()); return state(e)

    @app.post("/api/sessions/{sid}/exclude")
    def exclude(sid: str, body: dict = Body(...)):
        e = get(sid)
        ext = run(e, lambda s: s.exclude_stroke(float(body["row"]), float(body["x"]), bool(body.get("panel_x", False))))
        return dict(excluded_rows=list(ext), **state(e))

    @app.post("/api/sessions/{sid}/exclude/clear")
    def exclude_clear(sid: str):
        e = get(sid); run(e, lambda s: s.remove_exclusions()); return state(e)

    @app.post("/api/sessions/{sid}/backup_interval")
    def backup_interval(sid: str, body: dict = Body(...)):
        e = get(sid); run(e, lambda s: s.set_backup_interval(float(body["rows"][0]), float(body["rows"][1]))); return state(e)

    @app.post("/api/sessions/{sid}/backup_interval/clear")
    def backup_clear(sid: str):
        e = get(sid); run(e, lambda s: s.clear_backup_intervals()); return state(e)

    @app.get("/api/sessions/{sid}/overlay")
    def overlay(sid: str, r0: float | None = None, r1: float | None = None, max_points: int = 6000):
        e = get(sid)
        return _clean(e.s.overlay((r0, r1) if r0 is not None and r1 is not None else None, max_points))

    @app.get("/api/sessions/{sid}/qc")
    def qc(sid: str):
        e = get(sid); q = run(e, lambda s: s.qc())
        if not q:
            return dict(available=False, reason="no paired LAS / curve, or no trace yet")
        c = q["compare"]; st = max(1, len(c["grid"]) // 3000)
        return _clean(dict(available=True, metrics=q["metrics"], depth=c["grid"][::st], ref=c["ref"][::st],
                           digitized=c["test_aligned"][::st]))

    @app.get("/api/sessions/{sid}/qc.png")
    def qc_png(sid: str):
        e = get(sid); out_dir.mkdir(parents=True, exist_ok=True); p = out_dir / f"qc_{sid}.png"
        q = run(e, lambda s: s.qc(str(p)))
        if not q or not p.exists():
            raise HTTPException(404, "no QC available (paired LAS missing?)")
        return FileResponse(p, media_type="image/png")

    @app.post("/api/sessions/{sid}/export_las")
    def export_las(sid: str):
        e = get(sid); out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{e.s.well}_{Path(e.s.scan_path).stem}_{e.s.curve.replace(':', '_')}_assisted.las"
        p = run(e, lambda s: s.export_las(out_dir / name))
        return dict(path=str(p.relative_to(root)), url=f"/api/sessions/{sid}/download_las?name={name}")

    @app.get("/api/sessions/{sid}/download_las")
    def download_las(sid: str, name: str):
        p = (out_dir / Path(name).name)
        if not p.exists():
            raise HTTPException(404, "export first")
        return FileResponse(p, filename=p.name, media_type="text/plain")

    @app.post("/api/sessions/{sid}/save_training")
    def save_training(sid: str):
        e = get(sid); p = run(e, lambda s: s.save_training(train_dir))
        return dict(path=str(p.relative_to(root)))

    @app.post("/api/sessions/{sid}/save_session")
    def save_session(sid: str):
        e = get(sid); s = e.s
        p = out_dir / "sessions" / f"{s.well}_{Path(s.scan_path).stem}_{s.curve.replace(':', '_')}.json"
        d = s.session_dict()
        for k in ("scan", "las"):     # store project-relative paths (portable)
            if d.get(k) and Path(d[k]).resolve().is_relative_to(root):
                d[k] = Path(d[k]).resolve().relative_to(root).as_posix()
        import json
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(_clean(d), indent=1))
        return dict(path=str(p.relative_to(root)))

    @app.delete("/api/sessions/{sid}")
    def close(sid: str):
        sessions.pop(sid, None); return dict(closed=sid)

    return app
