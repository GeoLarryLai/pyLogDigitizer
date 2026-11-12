# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""routes: the pyLogDigitizer desktop app (/app/api/...).

The interactive editing routes are the existing editor's (``web_editor.app``), mounted at
/core (its API and static files only; the old editor page is not served) and sharing the same sessions, so there is one implementation of every digitizing action
(all logic in ``assisted_tracing`` / ``workflows``). This module only adds what the step-by-step
app needs on top: background jobs with progress, precomputation, undo snapshots, editable OCR
tables and the export guard (the editor's own write routes are blocked inside the app).

Tabs: Open scan · Tracks and depth · Curves · Trace and fix · Check and save.

On load every curve of the sheet is identified (``curve_identify``: by colour, by line style, or
both), gets its own working image and its own assisted session, and is traced. The selected
curve's session is ``e.s``, so every editing route acts on the selected curve; the viewer always
shows the original scan.

Input type (0.4.0): the Open scan tab first runs ``detect_input_kind`` (``GET /app/api/detect``) and
shows the kind (image scan, vector PDF or mixed) and the recommended method, which can be
overridden (``method`` in ``POST /app/api/open``). A vector PDF page is read exactly from its paths
(``app.vector_entry``): one read-only curve session per path, no tracing, and the page drawing as
the viewer image; the tracks, curves and trace steps are hidden and Check and save / export are
the same as for scans. A mixed PDF is traced from its image, with the header names and scales
taken from the PDF text.
"""
from __future__ import annotations
from pathlib import Path
import asyncio, csv, importlib.util, json, re, shutil, tempfile, uuid
from urllib.parse import unquote, urlsplit

import numpy as np
from fastapi import Body, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from pylogdigitizer import curve_tracing as ct, depth_calibration as dc
from pylogdigitizer.assisted_tracing import AssistedSession, SessionError, UNITS
from pylogdigitizer.depth_calibration import CalibrationError
from pylogdigitizer.track_grid_detection import LayoutError
from pylogdigitizer.web_editor.app import _clean, _Entry, catalog, SETTING_KEYS
from pylogdigitizer.web_editor.tiles import TilePyramid
from pylogdigitizer.app.shell import ExportGuard, UndoStack


def _display_path(path, root: Path) -> str:
    """Short project-relative path for the UI (never a raw absolute filesystem path)."""
    raw = Path(path) if path is not None else Path(root)
    try:
        resolved = raw.expanduser().resolve()
        root_r = Path(root).resolve()
        if resolved == root_r:
            return root_r.name
        return str(resolved.relative_to(root_r))
    except Exception:
        return raw.name or str(raw)


def safe_name(text, default: str = "curve") -> str:
    """A file-name-safe form of user text (curve, well and scan names): anything but letters, digits,
    ``.``, ``-`` and ``_`` becomes ``_``, so ``/``, ``\\`` and ``:`` can never nest or escape a path."""
    return re.sub(r"[^\w.-]+", "_", str(text or "")).strip("._") or default


def unique_keys(names) -> list[str]:
    """One output column per curve: the base mnemonic (before any ``:``), with ``_2``, ``_3`` ... added
    to repeats so two curves never overwrite each other in the LAS or the logs CSV."""
    seen: dict[str, int] = {}; out = []
    for n in names:
        base = str(n).split(":")[0].strip() or "CURVE"; k = base; i = 1
        while k in seen:
            i += 1; k = f"{base}_{i}"
        seen[k] = 1; out.append(k)
    return out


UPLOAD_MAX = 1 << 30          # 1 GiB per uploaded file
EXPECTED = (SessionError, CalibrationError, LayoutError, ValueError, FileNotFoundError)
SNAP_ATTRS = ("dm", "depth_anchors", "depth_checks", "depth_source", "section", "pan", "width", "_prob_map", "anchors",
              "excluded", "excluded_ids", "anchor_window_ft", "backup_intervals", "hidden_intervals", "values", "wraps", "scale", "track",
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
    model = next((p for p in (root / "models" / "kansas_kgs" / "unet.pt",) if p.exists()), None)
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
    d["curve_id"] = getattr(e, "active", None)
    d["app"] = json.loads(json.dumps(e.app.get("header_rows")))
    h = getattr(e, "header", None)          # header OCR words/proposals (small): word corrections are undoable
    d["header"] = (dict(h, tracks={k: dict(t, words=[dict(w) for w in t["words"]], curves=list(t["curves"])) for k, t in h["tracks"].items()},
                        curves=list(h.get("curves", []))) if isinstance(h, dict) and "tracks" in h else h)
    return d


def restore(e: _Entry, d) -> None:
    if d.get("curve_id") and d["curve_id"] != getattr(e, "active", None):
        select_curve(e, d["curve_id"])
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


def header_lines(words: list[dict], curves: list[dict]) -> list[dict]:
    """Group header OCR words (sheet coords, per track) into printed lines and match each line to the
    curve whose header label it is. Returns [{track, text, conf, x0, y0, x1, y1, words: [i], curve}]."""
    norm = lambda t: re.sub(r"\s+", " ", str(t or "")).strip().lower()
    lines = []
    for tr in sorted({w["track"] for w in words}):
        ws = sorted((w for w in words if w["track"] == tr), key=lambda w: (w["y0"] + w["y1"]) / 2)
        cur = []
        for w in ws:
            yc, h = (w["y0"] + w["y1"]) / 2, max(1.0, w["y1"] - w["y0"])
            if cur and abs(yc - np.mean([(q["y0"] + q["y1"]) / 2 for q in cur])) > 0.6 * h:
                lines.append(cur); cur = []
            cur.append(w)
        if cur:
            lines.append(cur)
    out = []
    for ln in lines:
        ln = sorted(ln, key=lambda w: w["x0"])
        out.append(dict(track=ln[0]["track"], text=" ".join(str(w["text"]) for w in ln),
                        conf=float(min(w["conf"] or 0 for w in ln)), x0=min(w["x0"] for w in ln), y0=min(w["y0"] for w in ln),
                        x1=max(w["x1"] for w in ln), y1=max(w["y1"] for w in ln), words=[w["i"] for w in ln], curve=None))
    for c in curves:                     # each curve's header label -> the line with the most shared tokens
        lab = set(norm(c.get("header")).split())
        best = max(((len(lab & set(norm(L["text"]).split())) / max(1, len(lab)), k) for k, L in enumerate(out)
                    if L["curve"] is None and L["track"] == c.get("track")), default=(0, None))
        if lab and best[1] is not None and best[0] >= 0.5:
            out[best[1]]["curve"] = c["id"]
    return out


def scale_fraction(v: np.ndarray, left: float, right: float, typ: str) -> np.ndarray:
    """Values -> position across their track (0 = left edge, 1 = right edge) on the printed scale."""
    v = np.asarray(v, float)
    if typ == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log10(v) - np.log10(left)) / (np.log10(right) - np.log10(left))
    return (v - left) / (right - left)


def wrap_side(left, right, bl, br) -> str:
    return "right" if abs(bl - right) <= abs(br - left) else "left"


def line_header_rows(rows: list) -> list:
    """Editable header table rows from :func:`propose_curves_from_lines` (printed names, not CWLS)."""
    import uuid as _uuid
    out = []
    for c in rows:
        bk = c.get("backup")
        review = ["scale"] if c.get("check") or None in (c.get("left"), c.get("right")) else []
        out.append(dict(id=_uuid.uuid4().hex[:6], origin="ocr-lines", track=c.get("track"),
                        mnemonic=c.get("name"), unit=c.get("unit") or "",
                        left=c.get("left"), right=c.get("right"), type=c.get("scale_type") or "linear",
                        backup_left=bk[0] if bk else None, backup_right=bk[1] if bk else None,
                        color=c.get("color") or "black", review=sorted(set(review)),
                        confidence=1.0 if not review else 0.5,
                        label=c.get("label") or "", points=[], check=c.get("check") or ""))
    return out


def color_header_rows(H: dict) -> list:
    """Header table for a colour scan: the line parser's rows (printed names kept) when they
    read at least as many clean scales as the default proposals, else the default rows."""
    line_rows = line_header_rows(H.get("line_curves") or [])
    def_rows = header_rows(H)
    clean = lambda rows: sum(1 for r in rows if r.get("left") is not None and r.get("right") is not None
                             and not r.get("check") and "scale" not in (r.get("review") or []))
    if line_rows and clean(line_rows) >= clean(def_rows):
        return line_rows
    return def_rows or line_rows


# ---------------------------------------------------------------- curves (one session per identified curve)
class MaskPyramid(TilePyramid):
    """DZI tiles of one curve's ink as a transparent dark layer (the highlight layer: dark grey on the\n    greyscale scan, so it stands out without competing with the coloured traces)."""

    def __init__(self, mask: np.ndarray, rgb=(17, 17, 17), tile_size: int = 512):
        import cv2
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # thin pen lines stay visible zoomed out
        super().__init__(np.where(mask, 255, 0).astype(np.uint8), tile_size)
        self.rgb = rgb

    def tile_png(self, lv: int, col: int, row: int) -> bytes:
        import io
        from PIL import Image
        img = self.level(min(lv, self.max_level))
        a = img[row * self.tile:(row + 1) * self.tile, col * self.tile:(col + 1) * self.tile]
        if a.size == 0:
            raise KeyError((lv, col, row))
        t = np.zeros(a.shape + (4,), np.uint8)
        t[..., :3] = self.rgb
        t[..., 3] = np.clip(a.astype(np.int32) * 3, 0, 235).astype(np.uint8)
        b = io.BytesIO(); Image.fromarray(t, "RGBA").save(b, "PNG", compress_level=1)
        return b.getvalue()


def curve_public(c: dict) -> dict:
    s = c.get("s")
    out = {k: c.get(k) for k in ("id", "track", "cls", "kind", "pattern", "color", "coverage", "name", "unit",
                                 "left", "right", "type", "backup", "header", "review", "ref")}
    if s is not None:
        out.update(left=s.scale["left"], right=s.scale["right"], type=s.scale["type"], backup=s.scale.get("backup"),
                   wrap_side=s.scale.get("wrap_side"), traced=s.T is not None, n_anchors=len(s.anchors),
                   errors={k: v for k, v in s.errors.items() if k in ("wrap_note", "layout_note", "section", "depth")})
    return out


def select_curve(e: _Entry, cid: str) -> None:
    c = next((c for c in getattr(e, "curves", []) if c["id"] == cid), None)
    if c is None:
        raise HTTPException(404, f"unknown curve {cid}")
    e.active = cid
    e.s = c["s"]
    e.app["name"] = f"{safe_name(e.base.well, 'well')}_{safe_name(Path(e.app['scan_stem']).stem, 'scan')}_{safe_name(c['name'])}"
    e.hl = None


def curve_series(c: dict):
    """(depth, value) of one curve: the session values, with off-scale stretches on identified
    curves handled as in the notebooks (wrapped -> backup scale, clipped -> null)."""
    s = c["s"]
    if c.get("kind") == "single":
        R = s.result()
        return np.asarray(R["depth"], float), np.asarray(R["value"], float)
    from pylogdigitizer.workflows import color_curve_values
    return color_curve_values(s)


def logs_table(e, ids=None, step: float = 0.5) -> dict:
    """Every traced curve (or ``ids``) on its depth grid, with the sheet row of each depth and the
    position across its track on the printed scale: the Check and save plot and the logs figure,
    drawn height-aligned with the scan."""
    from pylogdigitizer.curve_identify import track_ranges
    from pylogdigitizer.las_export import resample
    L = e.base.L
    tracks = [dict(name=k, left=float(v[0]), right=float(v[1])) for k, v in track_ranges(L).items()] if L is not None else []
    if L is None:                         # vector page: the tracks are the curves' header extents on the drawing
        seen = {}
        for c in e.curves:
            if c.get("track_px") and c.get("track") not in seen:
                seen[c["track"]] = dict(name=c["track"], left=c["track_px"][0], right=c["track_px"][1])
        tracks = sorted(seen.values(), key=lambda t: t["left"])
    out, d_lo, d_hi = [], np.inf, -np.inf
    for c in e.curves:
        s = c["s"]
        if (ids is not None and c["id"] not in ids) or s.T is None or s.dm is None:
            continue
        R = s.result(); g, v = resample(*curve_series(c), step)
        dd, rr = np.asarray(R["depth"], float), np.asarray(R["rows"], float)
        o = np.argsort(dd); row = np.interp(g, dd[o], rr[o])
        sc = s.scale
        out.append(dict(id=c["id"], name=c["name"], color=c["color"], track=c.get("track"), unit=c.get("unit") or "",
                        type=sc["type"], left=sc["left"], right=sc["right"], depth=g, value=v, row=row,
                        frac=scale_fraction(v, sc["left"], sc["right"], sc["type"])))
        d_lo, d_hi = min(d_lo, g[0]), max(d_hi, g[-1])
    ticks = []
    if out:
        ref = max(out, key=lambda q: len(q["depth"]))
        for d in np.arange(np.ceil(d_lo / 10) * 10, d_hi + 1e-6, 10.0):
            ticks.append([float(d), float(np.interp(d, ref["depth"], ref["row"])), bool(d % 50 == 0)])
    return dict(tracks=tracks, curves=out, ticks=ticks, width=int(e.tiles.w), height=int(e.tiles.h))


MAX_LAS_SAMPLES = 2_000_000


def check_las_step(step: float, e=None) -> float:
    """A typed LAS depth step: a finite number of feet > 0, and not so small that the LAS would hold
    more than :data:`MAX_LAS_SAMPLES` rows."""
    import math
    if not math.isfinite(step) or step <= 0:
        raise HTTPException(400, "LAS step must be a number of feet greater than 0")
    if e is not None:
        lo, hi = np.inf, -np.inf
        for c in e.curves:
            d = np.asarray(c["s"].result()["depth"], float) if c["s"].T is not None else np.array([])
            if d.size and np.isfinite(d).any():
                lo, hi = min(lo, np.nanmin(d)), max(hi, np.nanmax(d))
        if np.isfinite(lo) and (hi - lo) / step > MAX_LAS_SAMPLES:
            raise HTTPException(400, f"LAS step {step:g} ft is too small for {hi - lo:.0f} ft of log "
                                     f"(more than {MAX_LAS_SAMPLES:,} rows)")
    return step


def match_ref_curve(name: str, columns) -> str | None:
    """LAS column for a curve name: the same mnemonic, else the same canonical mnemonic."""
    cols = list(columns)
    if name in cols:
        return name
    key = lambda t: re.sub(r"[^A-Z0-9]", "", str(t).upper())          # FL_COND_AMB = "FL COND AMB", GAM_NAT = "GAM(NAT)"
    same = [c for c in cols if key(c) == key(name)]
    if len(same) == 1:
        return same[0]
    if re.match(r"TRACK\d+_", name):                     # unnamed curve: nothing to compare with yet
        return None
    from pylogdigitizer.header_ocr import CURVES, match_mnemonic
    want = match_mnemonic(name)[0]
    if want is None:
        return None
    aliases = {a.upper() for a in CURVES[want]["aliases"]}          # exact aliases first (RLL3 keeps its digit)
    hits = [c for c in cols if c.upper() in aliases] or [c for c in cols if match_mnemonic(c) == (want, 1.0)]
    return hits[0] if hits else None


def grid_lines(ink: np.ndarray, tracks: dict, body_rows) -> dict:
    """Printed grid of the sheet for the grid layer: vertical lines per track (x) and horizontal
    lines (rows), from long straight ink runs."""
    r0, r1 = int(body_rows[0]), int(body_rows[1])
    out = dict(v={}, h=[])
    for t, (xl, xr) in tracks.items():
        x0, x1 = int(xl) + 3, int(xr) - 3
        if x1 - x0 < 10:
            continue
        col = ink[r0:r1, x0:x1].mean(0)
        xs = np.flatnonzero(col > 0.6)
        groups = np.split(xs, np.flatnonzero(np.diff(xs) > 2) + 1) if len(xs) else []
        out["v"][t] = [float(x0 + g.mean()) for g in groups if len(g)]
    if tracks:
        xl, xr = next(iter(tracks.values()))
        row = ink[r0:r1, int(xl) + 3:int(xr) - 3].mean(1)
        ys = np.flatnonzero(row > 0.6)
        groups = np.split(ys, np.flatnonzero(np.diff(ys) > 2) + 1) if len(ys) else []
        out["h"] = [float(r0 + g.mean()) for g in groups if len(g)]
    return out


def _palette_public(pal: dict) -> dict:
    return {str(k): [int(v[0]), int(v[1]), int(v[2])] for k, v in pal.items()}


def install(app, root: Path, runner) -> None:
    sessions: dict = app.state.editor.state.sessions
    guard_default = root / "outputs" / "app"
    # one temporary folder per app run for uploads and rotated copies, removed when the process exits
    run_dir = Path(tempfile.mkdtemp(prefix="pyLogDigitizer_run_")).resolve(); app.state.run_dir = run_dir
    import atexit
    atexit.register(shutil.rmtree, run_dir, True)

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

    def export_state(e: _Entry) -> dict:
        """Export folder for the page: ``folder`` is the short label shown, ``folder_abs`` the real path
        (posted back by Open folder; the label is never turned back into a path)."""
        ex = e.guard.state(); ex["folder_abs"] = ex["folder"]; ex["folder"] = _display_path(ex["folder"], root)
        return ex

    def locked_push(e: _Entry, label: str) -> None:
        with e.lock:
            push_undo(e, label)

    def state(e: _Entry) -> dict:
        s = e.s; st = app.state.editor.state.state_fn(e)
        ex = export_state(e)
        st.update(sid=e.sid, undo=e.undo.labels()[-8:], export=ex, header_choice=e.app.get("header_choice"),
                  initial_scale=e.app["initial_scale"], name=e.app["name"])
        st.update(depth_sections=e.base.depth_sections, depth_section=e.base.depth_section, rotate=e.app.get("rotate", 0.0),
                  active=getattr(e, "active", None), kind=(getattr(e, "ident", None) or {}).get("kind", "single"),
                  n_curves=len(getattr(e, "curves", [])), scan_name=e.app.get("scan_stem"), las=e.base.las_path and _display_path(e.base.las_path, root),
                  skew=[float(a) for a in e.base.L.skew_deg] if e.base.L is not None else [])
        inp = e.app.get("input") or {}
        st.update(input=dict(kind=inp.get("kind"), used=inp.get("used", "scan"), method=inp.get("method", "auto"),
                             page=inp.get("page", 0), n_pages=inp.get("n_pages", 1),
                             method_label=inp.get("method_label"), evidence=inp.get("evidence"), notes=inp.get("notes")),
                  vector=e.app.get("vector"), mixed=e.app.get("mixed"))
        col = e.app.get("color") or {}
        st.update(is_color=bool(col.get("is_color")),
                  color=dict(is_color=bool(col.get("is_color")), confirmed=bool(col.get("confirmed")),
                             palette=_palette_public(col["palette"]) if col.get("palette") else {},
                             curve_color=col.get("curve_color"),
                             chromatic=[n for n in (col.get("palette") or {}) if n not in ("white", "black")],
                             body_rows=list(col["body_rows"]) if col.get("body_rows") else None,
                             reviews=col.get("reviews") or {},
                             source=col.get("source")))
        if s.dm is not None and s.depth_anchors:
            a = np.array(s.depth_anchors, float)
            st["residuals"] = _clean(np.c_[a[:, 1], dc.residuals(s.depth_anchors, s.dm)].tolist()) if len(a) > 2 else []
        return st

    # middleware: same-origin writes only; undo snapshots before editor edits; block editor file writes inside the app
    @app.middleware("http")
    async def guard_mw(request: Request, call_next):
        path = request.url.path
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin is not None and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        if path.startswith("/core") and not path.startswith(("/core/api/", "/core/static/")):
            return JSONResponse({"detail": "not part of the app (the old editor UI: run `pylogdigitizer editor`)"}, status_code=404)
        if BLOCKED.match(path):
            return JSONResponse({"detail": "writing files is done in the Export step of the app"}, status_code=403)
        m = MUTATING.match(path)
        e = sessions.get(m.group(1)) if m else None
        if e is not None and request.method in ("POST", "DELETE") and hasattr(e, "undo"):   # app sessions only
            # the lock may be held for seconds by a running job: wait in a worker thread, never on the event loop
            await run_in_threadpool(locked_push, e, m.group(2).split("/")[0])
        return await call_next(request)

    @app.get("/app/api/info")
    def info():
        return dict(project=_display_path(root, root), catalog=catalog(root), features=features(root))

    # ---------------- Open scan: load, identify every curve, trace each on its own working image
    def resolve(rel):
        """A scan or LAS path from the page: project-relative, or a file uploaded in this run."""
        if not rel:
            return None
        q = Path(str(rel)).expanduser(); q = (q if q.is_absolute() else root / q).resolve()
        if not any(q == b or q.is_relative_to(b) for b in (root, run_dir)):
            raise HTTPException(400, f"path outside the project: {rel}")
        if not q.exists():
            raise HTTPException(404, f"not found: {rel}")
        return q

    def identify_into(e: _Entry, j, n0: int, N: int, palette=None) -> None:
        """Identify every curve (colour / line style / both), build its working image and its
        assisted session, trace it. Without a track layout the loaded session is the one curve."""
        from pylogdigitizer import curve_identify as ci
        from pylogdigitizer.workflows import open_color_curve
        s = e.base
        if s.L is None or not s.L.track_names:
            e.curves = [dict(id="c1", track=s.track, cls="", kind="single", pattern="", color="#d62728", coverage=None,
                             name=s.curve, unit="", header="", review=[], ref=None, s=s)]
            select_curve(e, "c1")
            return
        col = e.app["color"]; rgb = None
        if col.get("is_color"):
            from pylogdigitizer.scan_io import load_rgb
            rgb = load_rgb(col["source"])
        j.progress(n0, N, "identifying the curves by colour and line style")
        ID = ci.identify_curves(s.scan_path, s.L, e.header, rgb=rgb, palette=palette or col.get("palette"))
        e.ident = ID
        if rgb is not None:
            col.update(palette=ID["palette"], names=list(ID["names"]), body_rows=list(ID["body_rows"]))
        e.work_dir = Path(tempfile.mkdtemp(prefix="work_", dir=run_dir))
        st = e.app["settings"]; preset = ci.trace_preset(ID)
        preset.update({k: v for k, v in (st.get("trace_kw") or {}).items() if k not in preset})
        cols = None
        if s.las_path:
            from pylogdigitizer.las_export import read_las
            cols = list(read_las(s.las_path).df().columns)
        curves = ID["curves"]
        N = max(N, n0 + 2 + len(curves))
        for k, c in enumerate(curves):
            j.progress(n0 + 1 + k, N, f"tracing {c['name']} ({c['cls']}, {c['track']}) on its own working image")
            from pylogdigitizer.curve_ink import save_ink_tif
            tif = save_ink_tif(ci.working_image(ID, c), e.work_dir / f"working_{c['id']}.tif")
            cv = dict(name=c["name"], track=c["track"], left=c["left"], right=c["right"], scale_type=c["type"],
                      backup=c.get("backup"))
            if c["name"] == e.app["curve"]:           # the preset of the catalogue entry (dataset profile)
                for key in ("backup", "wrap_side", "use_hatch"):
                    if st.get(key) is not None:
                        cv[key] = tuple(st[key]) if key == "backup" else st[key]
                if st.get("scale") is not None:
                    cv["left"], cv["right"] = st["scale"]
            scaled = "scale" not in c["review"]
            cs = open_color_curve(tif, cv, s.well, dict(preset), x_range=ID["tracks"][c["track"]],
                                  first_depth=s.first_depth, label_interval=s.label_interval,
                                  auto_backup=scaled, las_path=s.las_path)
            c["ref"] = match_ref_curve(c["name"], cols) if cols else None
            cs.ref_curve = c["ref"]
            c["s"] = cs
        e.curves = curves
        want = next((c for c in curves if c["name"] == e.app["curve"]), curves[0] if curves else None)
        if want is None:
            raise ValueError("no curves found on this scan")
        select_curve(e, want["id"])

    def build_entry(scan: Path, las, body: dict, j, sid: str | None = None, rotate: float = 0.0) -> _Entry:
        st = dict(body.get("settings") or {}); kw = {k: st[k] for k in SETTING_KEYS if k in st}
        for k in ("scale", "backup", "x_range", "rows"):
            if kw.get(k) is not None:
                kw[k] = tuple(kw[k])
        feats = features(root)
        if not feats["ocr"]["ok"]:
            kw["use_ocr"] = False
        curve = body.get("curve") or "GR"; well = body.get("well") or scan.parent.name
        src = scan
        if rotate:
            src = rotated_copy(scan, rotate)
        N0 = 6
        s = AssistedSession(src, curve=curve, well=well, las_path=las, auto_layout=bool(body.get("auto_layout", True)), **kw)
        s.open(progress=lambda i, n, m: j.progress(i, N0 + 4, m))
        sid = sid or uuid.uuid4().hex[:10]; e = _Entry(s); e.sid = sid; e.undo = UndoStack(); e.base = s
        e.guard = ExportGuard(guard_default / f"{safe_name(s.well, 'well')}_{safe_name(scan.stem, 'scan')}")
        e.app = dict(name="", scan_stem=scan.name, curve=curve, initial_scale=dict(s.scale), header_rows=None, header_ocr=None,
                     header_choice=None, settings=st, label_values=None, rotate=float(rotate), body=dict(body),
                     color=dict(is_color=False, confirmed=False, source=str(src)))
        labels = names = rgb = None
        try:
            from pylogdigitizer.scan_io import is_color_scan, load_rgb
            from pylogdigitizer.scan_color import suggest_palette, classify_colors
            if is_color_scan(src):
                rgb = load_rgb(src)
                pal = suggest_palette(rgb)
                labels, names = classify_colors(rgb, pal)
                e.app["color"] = dict(is_color=True, confirmed=True, source=str(src), palette=pal,
                                      curve_color=None, body_rows=None, reviews={})
        except Exception as ex:
            s.errors["color"] = f"{type(ex).__name__}: {ex}"
        e.tiles = TilePyramid(rgb if rgb is not None else s.ink)
        if s.L is not None:
            e.app["label_values"] = list(s.L._ocr_values) if s.L._ocr_values is not None else None
        j.progress(N0 - 1, N0 + 4, "header OCR (curve names, units, scales)" if feats["ocr"]["ok"] else "header OCR skipped (no OCR engine installed)")
        if feats["ocr"]["ok"] and s.L is not None:
            try:
                from pylogdigitizer import header_ocr
                e.header = header_ocr.propose_header(s.scan_path, layout=s.L, labels=labels, label_names=names)
                e.app["header_ocr"] = header_rows(e.header); e.app["header_rows"] = json.loads(json.dumps(e.app["header_ocr"]))
            except Exception as ex:      # header OCR is advisory: never blocks the workflow
                s.errors["header"] = f"{type(ex).__name__}: {ex}"
        if s.pan is not None and (s.L is None or not s.L.track_names):
            s.auto_trace()
        n_cur = 0
        if s.L is not None and s.L.track_names:
            n_cur = 12
        identify_into(e, j, N0, N0 + 4 + n_cur)
        if e.app["color"].get("is_color"):
            e.app["color"]["curve_color"] = next((c["cls"] for c in e.curves if c["id"] == e.active), None)
        e.tiles.level(max(0, e.tiles.max_level - 6))
        return e

    def rotated_copy(scan: Path, deg: float) -> Path:
        """The scan turned by ``deg`` degrees (white fill), written to a temporary file."""
        import cv2
        from pylogdigitizer.scan_io import load_rgb
        img = load_rgb(scan)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -float(deg), 1.0)
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255))
        d = Path(tempfile.mkdtemp(prefix="rotated_", dir=run_dir)); q = d / (scan.stem + ".tif")
        from PIL import Image
        Image.fromarray(out).save(q)
        return q

    def detect(scan: Path, page: int = 0) -> dict:
        from pylogdigitizer.input_kind import detect_input_kind
        try:
            d = detect_input_kind(scan, page=page).as_dict()
        except Exception as ex:
            raise HTTPException(400, f"cannot inspect {scan.name}: {ex}")
        d["path"] = _display_path(scan, root)
        return d

    @app.get("/app/api/detect")
    def detect_get(scan: str, page: int = 0):
        """Input type of a scan or PDF before it is opened: {kind: scan|vector|mixed, method, method_label,
        evidence: {n_paths, n_curve_paths, text_chars, image_coverage, ...}, notes}."""
        q = resolve(scan)
        if q is None:
            raise HTTPException(400, "no scan given")
        return _clean(detect(q, max(0, int(page))))

    def build_vector(scan: Path, las, body: dict, j, page) -> _Entry:
        from pylogdigitizer.app.vector_entry import build_vector_entry
        j.progress(0, 3, "reading the curve paths, depth numbers and header from the PDF")
        base, curves, rgb, log = build_vector_entry(scan, las, body, page)
        j.progress(2, 3, "drawing the page")
        e = _Entry(base); e.sid = uuid.uuid4().hex[:10]; e.undo = UndoStack(); e.base = base
        e.guard = ExportGuard(guard_default / f"{safe_name(base.well, 'well')}_{safe_name(scan.stem, 'scan')}")
        e.app = dict(name="", scan_stem=scan.name, curve=curves[0]["name"], initial_scale=dict(base.scale), header_rows=None,
                     header_ocr=None, header_choice=None, settings={}, label_values=None, rotate=0.0, body=dict(body),
                     color=dict(is_color=False, confirmed=False, source=str(scan)),
                     vector=dict(page=int(log.info["page"]), notes=list(log.info.get("notes") or []),
                                 headers=log.info.get("headers") or []))
        e.tiles = TilePyramid(rgb)
        e.ident = dict(kind="vector"); e.curves = curves
        if las:
            from pylogdigitizer.las_export import read_las
            cols = list(read_las(las).df().columns)
            for c in curves:
                c["ref"] = c["s"].ref_curve = match_ref_curve(c["name"], cols) or match_ref_curve(c["header"], cols)
        select_curve(e, curves[0]["id"])
        e.tiles.level(max(0, e.tiles.max_level - 6))
        return e

    @app.post("/app/api/open")
    def open_scan(body: dict = Body(...)):
        """{scan, las?, method?: auto|trace|vector|mixed, page?, ...}: open and analyse. ``auto`` (default)
        uses the detected input type; vector PDF pages are read from their paths (no tracing)."""
        scan, las = resolve(body.get("scan")), resolve(body.get("las"))
        if scan is None:
            raise HTTPException(400, "no scan given")
        method = str(body.get("method") or "auto")
        if method not in ("auto", "trace", "vector", "mixed"):
            raise HTTPException(400, "method must be auto, trace, vector or mixed")
        try:
            page = int(body["page"]) if body.get("page") not in (None, "") else None
        except (TypeError, ValueError):
            raise HTTPException(400, "page must be a page number")
        is_pdf = scan.suffix.lower() == ".pdf"
        if not is_pdf and method in ("vector", "mixed"):
            raise HTTPException(400, f"{method} needs a PDF file; {scan.name} is an image (use trace)")
        found = detect(scan, page or 0)
        use = {"trace": "scan", "vector": "vector", "mixed": "mixed"}.get(method) or found["kind"]
        if use == "vector" and page is None and found.get("evidence", {}).get("curve_pages"):
            page = found["evidence"]["curve_pages"][0] - 1
        if use != "vector" and page:
            raise HTTPException(400, "scanned and mixed PDFs are read from their first page only; "
                                     "pick page 1 (vector PDFs can use any page)")

        def work(j):
            if use == "vector":
                e = build_vector(scan, las, body, j, page)
            else:
                e = build_entry(scan, las, body, j)
                if use == "mixed":
                    from pylogdigitizer.pdf_vector.mixed import mixed_header, apply_mixed_header
                    from pylogdigitizer.curve_identify import track_ranges
                    try:
                        H = mixed_header(scan, page or 0)
                        tr = track_ranges(e.base.L) if e.base.L is not None else {}
                        e.app["mixed"] = dict(applied=apply_mixed_header(e.curves, tr, H), n_headers=len(H["headers"]),
                                              depth_labels=len(H["depth_labels"]))
                        if e.active:
                            select_curve(e, e.active)
                    except Exception as ex:          # the traced curves stay; only the header source changes
                        e.base.errors["mixed"] = f"PDF text header not used: {type(ex).__name__}: {ex}"
            e.app["input"] = dict(found, used=use, method=method)
            sessions[e.sid] = e
            return dict(sid=e.sid, n_curves=len(e.curves), input=use)
        return job(None, "Load scan", work)

    @app.post("/app/api/upload")
    async def upload(request: Request):
        """Raw file body with a URL-encoded ``X-Filename`` header (the page's file picker): kept in this
        run's temporary folder (removed when the app stops) and opened from there. Streams to disk and
        refuses files over ``app.state.upload_max`` bytes (1 GiB). Returns the path to pass to /open."""
        name = Path(unquote(request.headers.get("x-filename") or "scan.tif")).name
        if Path(name).suffix.lower() not in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".pdf", ".las"):
            raise HTTPException(400, f"unsupported file type: {name}")
        cap = int(getattr(app.state, "upload_max", UPLOAD_MAX))
        d = run_dir / "uploads"; d.mkdir(exist_ok=True)
        q = d / name; tmp = d / (name + ".part"); n = 0
        try:
            with open(tmp, "wb") as fh:
                async for chunk in request.stream():
                    n += len(chunk)
                    if n > cap:
                        raise HTTPException(413, f"file too large (limit {cap // (1 << 20)} MB)")
                    fh.write(chunk)
            tmp.replace(q)
        finally:
            tmp.unlink(missing_ok=True)
        return dict(path=str(q), name=name, size=q.stat().st_size)

    @app.post("/app/api/{sid}/rotate")
    def rotate(sid: str, body: dict = Body(...)):
        """{deg}: turn the scan by a small extra angle (after the automatic deskew) and analyse it again."""
        e = get(sid); deg = float(body.get("deg", 0.0))
        if abs(deg) > 10:
            raise HTTPException(400, "rotate by at most 10 degrees")
        b = dict(e.app["body"]); scan = resolve(b.get("scan"))

        def work(j):
            n = build_entry(scan, e.base.las_path, b, j, sid=sid, rotate=deg)
            sessions[sid] = n
            return dict(sid=sid, rotate=deg)
        return job(e, "Rotate", work)

    # ---------------- curves: list, select, edit the side panel, layers
    @app.get("/app/api/{sid}/curves")
    def curves_get(sid: str):
        e = get(sid)
        with e.lock:
            kind = (getattr(e, "ident", None) or {}).get("kind", "single")
            return _clean(dict(kind=kind, active=e.active, curves=[curve_public(c) for c in e.curves]))

    @app.post("/app/api/{sid}/curves/select")
    def curves_select(sid: str, body: dict = Body(...)):
        e = get(sid)
        with e.lock:
            select_curve(e, str(body.get("id")))
            if e.app["color"].get("is_color"):
                e.app["color"]["curve_color"] = next(c["cls"] for c in e.curves if c["id"] == e.active)
            return state(e)

    @app.post("/app/api/{sid}/curves/{cid}")
    def curves_edit(sid: str, cid: str, body: dict = Body(...)):
        """{name?, unit?, left, right, type, backup: [l, r] | null, wrap_side?}: the side panel.
        Values are re-mapped at once (no re-trace)."""
        e = get(sid)
        c = next((c for c in e.curves if c["id"] == cid), None)
        if c is None:
            raise HTTPException(404, f"unknown curve {cid}")
        s = c["s"]
        try:
            l, r = float(body.get("left", s.scale["left"])), float(body.get("right", s.scale["right"]))
            bk = body.get("backup", s.scale.get("backup"))
            bk = [float(bk[0]), float(bk[1])] if bk not in (None, [], "") and None not in bk and "" not in bk else None
        except (TypeError, ValueError):
            raise HTTPException(400, "scale values must be numbers")
        typ = body.get("type", s.scale["type"])
        if typ not in ("linear", "log"):
            raise HTTPException(400, "type must be linear or log")
        with e.lock:
            push_undo(e, f"curve {c['name']}")          # snapshot under the lock: never a half-done trace
            s.set_scale(l, r, typ, backup=bk, wrap_side=body.get("wrap_side") or s.scale.get("wrap_side") or "right")
            if body.get("name"):
                c["name"] = s.curve = str(body["name"]).strip()
                if s.las_path:
                    from pylogdigitizer.las_export import read_las
                    c["ref"] = s.ref_curve = match_ref_curve(c["name"], read_las(s.las_path).df().columns)
            if "unit" in body:
                c["unit"] = str(body.get("unit") or "")
            c["review"] = [x for x in c.get("review", []) if not x.startswith(("name", "scale"))]
            if e.active == cid:
                select_curve(e, cid)
        return _clean(curve_public(c))

    @app.get("/app/api/{sid}/traces")
    def traces(sid: str):
        """Every curve's trace (sheet coords, decimated) for the traced-curves layer."""
        e = get(sid); out = []
        with e.lock:
            for c in e.curves:
                s = c["s"]
                if s.T is None:
                    continue
                ov = s.overlay(max_points=2500)
                out.append(dict(id=c["id"], name=c["name"], color=c["color"], trace=[[r, x] for r, x, *_ in ov["trace"]]))
            return _clean(dict(traces=out, active=e.active))

    @app.get("/app/api/{sid}/logs")
    def logs(sid: str):
        """All traced curves for the Check and save plot: per curve the sheet row and the position across
        its track of every 0.5 ft sample (null where off scale), plus 10 ft depth ticks (row, major)."""
        e = get(sid)
        with e.lock:
            T = logs_table(e)
        for c in T["curves"]:
            c.update(depth=c["depth"].tolist(), row=np.round(c["row"], 1).tolist(), frac=np.round(c["frac"], 4).tolist(),
                     value=c["value"].tolist())
        return _clean(T)

    @app.get("/app/api/{sid}/grid_lines")
    def grid_layer(sid: str):
        e = get(sid); s = e.base
        if s.L is None:
            return dict(available=False, v={}, h=[], tracks=[])
        if getattr(e, "grid_cache", None) is None:
            from pylogdigitizer.curve_identify import track_ranges
            tr = track_ranges(s.L)
            body = (getattr(e, "ident", None) or {}).get("body_rows") or (int(s.L.probe_row or 0), s.L.ink.shape[0])
            e.grid_cache = dict(available=True, tracks=[dict(name=k, left=v[0], right=v[1]) for k, v in tr.items()],
                                **grid_lines(s.L.ink, tr, body))
        return _clean(e.grid_cache)

    def highlight(e: _Entry):
        if getattr(e, "hl", None) is None or e.hl[0] != e.active:
            c = next(c for c in e.curves if c["id"] == e.active)
            ID = getattr(e, "ident", None)
            if ID is None or c.get("kind") == "single":
                s = c["s"]; m = np.zeros(e.tiles.h * e.tiles.w, bool).reshape(e.tiles.h, e.tiles.w)
                if s.T is not None:
                    ov = s.overlay(max_points=20000)
                    for r, x, *_ in ov["trace"]:
                        rr, xx = int(r), int(x)
                        m[max(0, rr - 2):rr + 3, max(0, xx - 2):xx + 3] = True
            else:
                from pylogdigitizer.curve_identify import curve_mask
                m = curve_mask(ID, c)
            e.hl = (e.active, MaskPyramid(m))
        return e.hl[1]

    @app.get("/app/api/{sid}/highlight/{cid}.dzi")
    def highlight_dzi(sid: str, cid: str):
        from fastapi.responses import Response
        e = get(sid)
        if cid != e.active:
            raise HTTPException(404, "highlight is served for the selected curve")
        return Response(highlight(e).dzi_xml(), media_type="application/xml")

    @app.get("/app/api/{sid}/highlight/{cid}_files/{level}/{name}")
    def highlight_tile(sid: str, cid: str, level: int, name: str):
        from fastapi.responses import Response
        e = get(sid)
        if cid != e.active:
            raise HTTPException(404, "highlight is served for the selected curve")
        c, r = name.split(".")[0].split("_")
        try:
            return Response(highlight(e).tile_png(level, int(c), int(r)), media_type="image/png",
                            headers={"Cache-Control": "max-age=3600"})
        except KeyError:
            raise HTTPException(404, "no tile")

    # ---------------- Check and save: every curve against the reference LAS
    @app.post("/app/api/{sid}/reference")
    def reference_set(sid: str, body: dict = Body(...)):
        """{las}: load a reference LAS (project-relative or an uploaded file) for the comparison."""
        e = get(sid); q = resolve(body.get("las"))
        from pylogdigitizer.las_export import read_las
        try:
            cols = list(read_las(q).df().columns)
        except Exception as ex:
            raise HTTPException(400, f"not a readable LAS file: {ex}")
        with e.lock:
            e.base.las_path = str(q)
            for c in e.curves:
                c["s"].las_path = str(q)
                c["ref"] = c["s"].ref_curve = match_ref_curve(c["name"], cols)
        return dict(las=_display_path(q, root), matched={c["name"]: c["ref"] for c in e.curves})

    @app.get("/app/api/{sid}/compare")
    def compare(sid: str):
        e = get(sid)
        with e.lock:
            return _compare(e)

    def _compare(e: _Entry) -> dict:
        las = e.base.las_path
        if not las:
            return dict(available=False, reason="no reference LAS loaded")
        from pylogdigitizer import qc_compare as qc
        from pylogdigitizer.las_export import read_las, resample
        df = read_las(las).df().sort_index()
        rows, plot = [], None
        for c in e.curves:
            s = c["s"]
            ref = c.get("ref") or match_ref_curve(c["name"], df.columns)
            row = dict(id=c["id"], name=c["name"], ref=ref, color=c["color"], unit=c.get("unit"))
            if ref is None or s.T is None or s.dm is None:
                rows.append(row); continue
            d, v = curve_series(c)
            rs = df[ref].dropna(); rs = rs[rs > -999]
            rd, rv, tv = rs.index.values.astype(float), rs.values.astype(float), v
            if s.scale["type"] == "log":
                rv = np.log10(np.clip(rv, 1e-3, None)); tv = np.log10(np.clip(v, 1e-3, None))
            m = qc.compare(rd, rv, d, tv)
            row.update(r=m["aligned"].get("r"), rmse=m["aligned"].get("rmse"), bias=m["aligned"].get("bias"),
                       n=m["aligned"].get("n", int(np.isfinite(m["test_aligned"]).sum()) if len(m["grid"]) else 0),
                       shift_ft=m["shift_ft"], log=s.scale["type"] == "log")
            rows.append(row)
            if c["id"] == e.active:
                g, vv = resample(d, v, 0.5)
                plot = dict(depth=g.tolist(), digitized=vv.tolist(),
                            ref=np.interp(g, rs.index.values, rs.values, left=np.nan, right=np.nan).tolist(), name=c["name"], ref_name=ref)
        return _clean(dict(available=True, las=_display_path(las, root), rows=rows, plot=plot, active=e.active))

    @app.get("/app/api/{sid}/state")
    def get_state(sid: str):
        return state(get(sid))

    # ---------------- colour palette (colour scans / raster PDFs only)
    @app.get("/app/api/{sid}/color")
    def color_get(sid: str):
        e = get(sid)
        st = state(e)
        return _clean(st.get("color") or dict(is_color=False, confirmed=False, palette={}, curve_color=None,
                                              chromatic=[], body_rows=None, reviews={}))

    @app.post("/app/api/{sid}/color/sample")
    def color_sample(sid: str, body: dict = Body(...)):
        """Sample the original colour scan at sheet (row, x). Used to add a swatch by clicking."""
        e = get(sid)
        src = (e.app.get("color") or {}).get("source") or e.s.scan_path
        from pylogdigitizer.scan_io import load_rgb
        rgb = load_rgb(src)
        r, x = int(round(float(body["row"]))), int(round(float(body["x"])))
        if not (0 <= r < rgb.shape[0] and 0 <= x < rgb.shape[1]):
            raise HTTPException(400, f"sample ({r}, {x}) outside the scan {rgb.shape[1]}×{rgb.shape[0]}")
        pix = rgb[r, x, :3]
        return dict(rgb=[int(pix[0]), int(pix[1]), int(pix[2])], row=r, x=x)

    @app.post("/app/api/{sid}/color")
    def color_set(sid: str, body: dict = Body(...)):
        """Edit or confirm the palette.

        Body: ``{palette?: {name: [r,g,b]}, curve_color?: str, confirm?: bool}``.
        ``white`` and ``black`` are required. Confirming identifies the curves again with this
        palette (one working image and session per curve) and selects a ``curve_color`` curve.
        """
        e = get(sid)
        col = e.app.setdefault("color", dict(is_color=True, confirmed=False, source=e.base.scan_path, palette={}, reviews={}))
        if body.get("palette") is not None:
            pal = {}
            for name, rgb in body["palette"].items():
                nm = str(name).strip()
                if not nm:
                    raise HTTPException(400, "palette names must be non-empty")
                try:
                    pal[nm] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                except Exception:
                    raise HTTPException(400, f"bad RGB for {nm!r}")
            if "white" not in pal or "black" not in pal:
                raise HTTPException(400, "palette needs 'white' and 'black' entries")
            col["palette"] = pal
            col["is_color"] = True
        if body.get("curve_color"):
            if body["curve_color"] not in (col.get("palette") or {}):
                raise HTTPException(400, f"curve_color {body['curve_color']!r} is not in the palette")
            col["curve_color"] = body["curve_color"]
        if not body.get("confirm"):
            return color_get(sid)

        def work(j):
            from pylogdigitizer.scan_color import require_palette
            require_palette(col.get("palette") or {})
            push_undo(e, "palette")
            if e.base.L is None or not e.base.L.track_names:     # no track layout: keep the loaded session
                col.update(confirmed=True)
                return dict(confirmed=True, curve_color=col.get("curve_color"), n_curves=len(e.curves))
            identify_into(e, j, 0, 14, palette=col["palette"])
            want = col.get("curve_color")
            pick = next((c for c in e.curves if c["cls"] == want and c["name"] == e.app["curve"]), None) or \
                next((c for c in e.curves if c["cls"] == want), None)
            if pick is not None:
                select_curve(e, pick["id"])
            col.update(confirmed=True, curve_color=next(c["cls"] for c in e.curves if c["id"] == e.active))
            e.grid_cache = None
            return dict(confirmed=True, curve_color=col["curve_color"], n_curves=len(e.curves))
        return job(e, "Confirm colour palette", work)

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

    @app.get("/app/api/{sid}/header/lines")
    def header_lines_get(sid: str):
        """Header OCR as printed lines (words grouped per track), each matched to its curve, with the
        header strip rows: what the Curves tab shows so the reading can be checked and corrected."""
        e = get(sid)
        words = header_words(sid)["words"]
        lines = header_lines(words, [curve_public(c) for c in e.curves])
        bottom = max((w["y1"] for w in words), default=None)
        if e.s.L is not None and e.s.L.probe_row:
            bottom = max(bottom or 0, float(e.s.L.probe_row))
        conf = [w["conf"] for w in words if w.get("conf") is not None]
        return _clean(dict(lines=lines, strip=[0.0, bottom], n_words=len(words),
                           mean_conf=float(np.mean(conf)) if conf else None, n_low=int(sum(q < 80 for q in conf))))

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
        # a colour scan's header was read line by line: keep those rows (as Re-run OCR does)
        rows = color_header_rows(e.header) if e.header.get("parse") == "lines" else header_rows(e.header)
        e.app["header_ocr"] = rows; e.app["header_rows"] = json.loads(json.dumps(rows))
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
            col = e.app.get("color") or {}
            src = col.get("source") or e.s.scan_path
            kw = {}
            if col.get("confirmed") and col.get("palette"):
                from pylogdigitizer.scan_io import load_rgb
                from pylogdigitizer.scan_color import classify_colors
                rgb = load_rgb(src)
                labels, names = classify_colors(rgb, col["palette"])
                kw = dict(parse="lines", labels=labels, label_names=names)
            L_hdr = getattr(e, "header_layout", None) if kw else None
            e.header = header_ocr.propose_header(src, layout=L_hdr or e.s.L, **kw)
            rows = color_header_rows(e.header) if kw.get("parse") == "lines" else header_rows(e.header)
            e.app["header_ocr"] = rows; e.app["header_rows"] = json.loads(json.dumps(rows))
            return dict(n=len(e.app["header_rows"]))
        return job(e, "Header OCR", work)

    def propagate_depth(e: _Entry, j, n0: int, N: int) -> None:
        """Give every curve session the depth anchors of the loaded scan, then re-cut and re-trace it."""
        b = e.base
        for k, c in enumerate(getattr(e, "curves", [])):
            cs = c["s"]
            if cs is b or not b.depth_anchors:
                continue
            j.progress(n0 + k, N, f"re-tracing {c['name']} with the new depth scale")
            cs.set_depth_anchors(b.depth_anchors)
            sec = cs.section
            if sec is not None and sec["kind"] == "box":
                cs.set_section(box=(sec["x0"], sec["x1"]), rows=tuple(sec["rows"]))
            elif sec is not None:
                cs.set_section(track=sec["track"])
            if cs.pan is not None:
                cs.auto_trace()

    # ---------------- step 5a: which log section (only when the sheet holds several and the header does not say)
    @app.post("/app/api/{sid}/depth_section")
    def depth_section(sid: str, body: dict = Body(...)):
        """{index: n} - use log section n for the depth scale, then re-cut and re-trace every curve."""
        e = get(sid); push_undo(e, "depth section")
        def work(j):
            s = e.base; N = 2 + len(e.curves); j.progress(0, N, "depth calibration of the chosen section")
            s.set_depth_section(int(body["index"]))
            e.app["label_values"] = list(s.L._ocr_values) if s.L._ocr_values is not None else None
            if s.track in s.L.track_names and (s.section is None or s.section["kind"] == "track"):
                j.progress(1, N, "tracing the chosen section")
                s.set_section(track=s.section["track"] if s.section else s.track); s.errors.pop("section", None)
                if s is e.s:
                    s.auto_trace()
            propagate_depth(e, j, 2, N)
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
            s = e.base; N = 2 + len(e.curves); j.progress(0, N, "depth calibration")
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
            if s is e.s:                                   # the loaded session is the curve (no track layout)
                if s.pan is not None and s.section and s.section["kind"] == "track" and s.L is not None:
                    j.progress(1, N, "re-tracing with the new depth scale")
                    s.set_section(track=s.section["track"]); s.auto_trace()
                elif s.pan is not None:
                    j.progress(1, N, "re-tracing"); s.auto_trace()
            propagate_depth(e, j, 1, N)
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
        """{left_dx, right_dx}: move the borders of the selected curve's track (sheet px), re-cut
        and re-trace every curve in that track."""
        e = get(sid); push_undo(e, "track edges")
        def work(j):
            tr = next((x.get("track") for x in e.curves if x["id"] == e.active), None)
            same = [c["s"] for c in e.curves if c["s"] is e.s or (tr and c.get("track") == tr and c.get("kind") != "single")]
            for k, cs in enumerate(same):
                j.progress(k, len(same), "re-cutting the track")
                cs.adjust_track_edges(float(body.get("left_dx", 0)), float(body.get("right_dx", 0)))
                cs.auto_trace()
            e.hl = None
            return dict(ok=True)
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
        if "mask_other" in body:       # trace on the separated curve layer (separation.curve_layer)
            e.s.trace_kw["mask_other"] = bool(body["mask_other"])
        if "n_curves" in body:         # several curves in the track: trace each, offer them as candidates
            n = int(body["n_curves"])
            if not 1 <= n <= 4:
                raise HTTPException(400, "n_curves must be 1-4")
            if n > 1:
                e.s.trace_kw["n_curves"] = n
            else:
                e.s.trace_kw.pop("n_curves", None)
            e.s.trace_kw.pop("pick", None)
        def work(j):
            s = e.s
            if mode == "conventional" and s.trace_kw.get("cost_mode", "conventional") == "conventional" and (
                    not body.get("force") or int(s.trace_kw.get("n_curves", 1)) > 1):   # set_cost_mode re-traces one curve only
                j.progress(0, 1, "conventional trace"); s.auto_trace(anchors_panel=s.anchors or None, excluded=s.excluded or None, strokes=s.strokes or None)
            else:
                j.progress(0, 1, f"{mode} cost map + trace" + (" (U-Net)" if mode != "conventional" else ""))
                s.set_cost_mode(mode, ml_model=f["ml"]["model"] if mode != "conventional" else None)
            return dict(ok=True)
        return job(e, "Trace", work)

    @app.get("/app/api/{sid}/trace/options")
    def trace_options(sid: str):
        e = get(sid)
        with e.lock:
            kw = dict(e.s.trace_kw); T = e.s.T or {}
            T = dict(candidates=len(T.get("candidates") or []), picked=T.get("picked", 0))
        return dict(cost_mode=kw.get("cost_mode", "conventional"), mask_other=bool(kw.get("mask_other", False)),
                    n_curves=int(kw.get("n_curves", 1)), pick=kw.get("pick", "auto"),
                    candidates=T["candidates"] or 1, picked=T["picked"])

    @app.post("/app/api/{sid}/trace/pick")
    def trace_pick(sid: str, body: dict = Body(...)):
        """{index}: use candidate ``index`` (from the overlay's candidates) as the curve; re-traces
        with the pins, guides and exclusions kept. Undo (Z) restores the previous pick."""
        e = get(sid); n = len((e.s.T or {}).get("candidates") or [])
        if n < 2:
            raise HTTPException(400, "no curve candidates: trace with n_curves > 1 first")
        i = int(body.get("index", -1))
        if not 0 <= i < n:
            raise HTTPException(400, f"index must be 0-{n - 1}")
        push_undo(e, f"pick curve {i + 1}")
        def work(j):
            s = e.s; s.trace_kw["pick"] = i
            j.progress(0, 1, f"tracing with curve {i + 1}")
            s.auto_trace(anchors_panel=s.anchors or None, excluded=s.excluded or None, strokes=s.strokes or None)
            return dict(ok=True, picked=s.T.get("picked"))
        return job(e, "Pick curve", work)

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

    @app.post("/app/api/{sid}/fix/node")
    def fix_node(sid: str, body: dict = Body(...)):
        """{row, x, radius}: manual node (drag or force-click); local ink snap inside the circle, instant."""
        e = get(sid)
        return guided(e, "node", lambda s: s.node_snap(float(body["row"]), float(body["x"]), radius_px=float(body.get("radius", 60))))

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
        e = get(sid)
        raw = body.get("path")
        if not raw:
            folder = guard_default / safe_name(e.app["name"], "export")
        else:
            q = Path(raw)
            folder = q if q.is_absolute() else (root / q)
        e.guard.set_folder(folder)
        return export_state(e)

    @app.post("/app/api/{sid}/export/{kind}")
    def export(sid: str, kind: str, body: dict = Body(default={})):
        """Write files (the only writer). ``las`` = every curve on one depth grid (``step`` ft, any
        step > 0, default 0.5); ``csv`` / ``qc_png`` / ``session`` = the selected curve; ``logs_csv`` = every curve on one grid
        and ``figure`` = the scan next to the plotted logs (both take ``ids`` to choose curves; ``figure``
        takes ``image: false`` to leave the scan out); ``all`` = LAS, CSV, QC plot and session."""
        e = get(sid); s = e.s; name = e.app["name"]; out = []
        try:                             # any depth step > 0 ft, typed in the app (scans and vector pages)
            step = float((body or {}).get("step", 0.5))
        except (TypeError, ValueError):
            raise HTTPException(400, "LAS step must be a number of feet greater than 0")
        check_las_step(step, e)
        kinds = ["las", "csv", "qc_png", "session"] if kind == "all" else [kind]
        ids = (body or {}).get("ids")
        if ids is not None:
            ids = {str(i) for i in ids}
            if not ids and kind in ("logs_csv", "figure"):
                raise HTTPException(400, "no curves chosen: tick at least one log to plot")
        stem = f"{safe_name(e.base.well, 'well')}_{safe_name(Path(e.app['scan_stem']).stem, 'scan')}"
        f = features(root)
        cur = next((c for c in e.curves if c["id"] == e.active), None)
        if cur is None:
            raise HTTPException(404, "no curve selected")
        for k in kinds:
            if k == "las":
                def wlas(p):
                    from pylogdigitizer.las_export import common_grid, write_las
                    keys = dict(zip((c["id"] for c in e.curves), unique_keys(c["name"] for c in e.curves)))
                    series = {keys[c["id"]]: curve_series(c) for c in e.curves
                              if c["s"].T is not None and c["s"].dm is not None}
                    if not series:
                        raise SessionError("nothing traced yet")
                    g, cv = common_grid(series, step=step)
                    units = {keys[c["id"]]: (c.get("unit") or UNITS.get(c["name"].split(":")[0], "")) for c in e.curves}
                    return write_las(p, g, cv, units=units, well={"WELL": e.base.well, "SRVC": "pyLogDigitizer app"})
                fname = name + ".las" if len(e.curves) == 1 else f"{stem}.las"
                out.append(locked(e, lambda s: e.guard.write(fname, wlas)))
            elif k == "csv":
                def wcsv(p):
                    from pylogdigitizer.las_export import resample
                    g, v = resample(*curve_series(cur), step); c = s.curve.split(":")[0]
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
                        elif d.get(kk):
                            d[kk] = Path(d[kk]).name
                    d["header_rows"] = e.app.get("header_rows")
                    d["curve_meta"] = {kk: cur.get(kk) for kk in ("track", "cls", "kind", "pattern", "unit")}
                    Path(p).write_text(json.dumps(_clean(d), indent=1)); return p
                out.append(locked(e, lambda s: e.guard.write(name + "_session.json", wses)))
            elif k == "logs_csv":      # every (or the chosen) curve on one depth grid, like the LAS
                def wlcsv(p):
                    from pylogdigitizer.las_export import common_grid
                    keys = dict(zip((c["id"] for c in e.curves), unique_keys(c["name"] for c in e.curves)))
                    series = {keys[c["id"]]: curve_series(c) for c in e.curves
                              if (ids is None or c["id"] in ids) and c["s"].T is not None and c["s"].dm is not None}
                    if not series:
                        raise SessionError("nothing traced yet")
                    g, cv = common_grid(series, step=step)
                    with open(p, "w", newline="") as fh:
                        w = csv.writer(fh); w.writerow(["DEPT", *cv])
                        for i, d in enumerate(g):
                            w.writerow([repr(float(d)), *(repr(float(v[i])) for v in cv.values())])
                    return p
                out.append(locked(e, lambda s: e.guard.write(f"{stem}_logs.csv", wlcsv)))
            elif k == "figure":        # the scan next to the plotted logs, height-aligned (PNG)
                if not f["plot"]["ok"]:
                    raise HTTPException(400, f["plot"]["hint"])
                def wfig(p):
                    from pylogdigitizer import qc_compare as qc
                    T = logs_table(e, ids, step)
                    if not T["curves"]:
                        raise SessionError("nothing traced yet")
                    img = e.tiles.level(e.tiles.max_level) if (body or {}).get("image", True) else None
                    return qc.logs_figure(str(p), T, image=img, title=f"{e.base.well} · {Path(e.app['scan_stem']).name}")
                out.append(locked(e, lambda s: e.guard.write(f"{stem}_logs.png", wfig)))
            elif k == "training":      # correction sample for ML training (project folder, not in "all")
                out.append(locked(e, lambda s: s.save_training(root / "training_data" / "corrections")))
            else:
                raise HTTPException(404, f"unknown export {k}")
        return dict(e.guard.state(), written=[str(p) for p in out], all_written=e.guard.state()["written"])
