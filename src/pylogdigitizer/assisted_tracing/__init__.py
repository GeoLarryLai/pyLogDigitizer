# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""assisted_tracing: interactive (human-in-the-loop) digitization of one curve.

The same API drives the web editor (``web_editor``) and notebook 01, so a session saved in the
editor can be replayed in Python.

    s = AssistedSession("example_data/.../4342.tif", curve="GR", track="track1", scale=(0, 150),
                        backup=(150, 300), use_hatch=True, las_path=".../1042553273.las")
    s.open()                      # auto layout + OCR depth calibration (errors kept in s.errors)
    s.auto_trace()                # conventional trace of the section
    s.add_anchor(row, x_sheet)    # click on the curve: re-trace only between the neighbouring anchors
    s.qc()                        # r / RMSE vs the paired LAS (if any)
    s.export_las("out.las"); s.save_training("training_data/corrections"); s.save_session("x.json")

Coordinates: ``row``/``x`` are pixels of the *working sheet* (the deskewed scan image the editor
shows). Anchors are stored as (sheet_row, x_panel) where x_panel is the column in the rectified
track panel (0 = left track edge). Depth anchors are (sheet_row, depth_ft).

Local re-trace: a new anchor re-runs the DP tracer (Rust/numba/Python backend) only on the rows
between its neighbouring anchors, at most ``window_ft`` above/below; the window ends are pinned
to the current path so the rest of the trace does not move.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from pylogdigitizer import workflows as wf, curve_tracing as ct, depth_calibration as dc, scale_calibration as sc
from pylogdigitizer.scan_io import load_ink
from pylogdigitizer.track_grid_detection import LayoutError
from pylogdigitizer.depth_calibration import CalibrationError

__all__ = ["AssistedSession", "SessionError", "UNITS"]
UNITS = {"GR": "API", "DT": "US/F", "ILD": "OHMM", "RHOB": "G/C3", "NPHI": "V/V", "SP": "MV", "CALI": "IN"}
SESSION_VERSION = 2


class SessionError(ValueError):
    """Wrong order of steps or bad input (message says what to do)."""


class AssistedSession:
    def __init__(self, scan_path, curve: str = "GR", well: str = "", las_path=None, track: str | None = "track1",
                 scale=(0.0, 150.0), scale_type: str = "linear", backup=None, wrap_side: str = "right",
                 use_hatch: bool = False, first_depth: float | None = None, label_interval: float | None = None,
                 use_ocr: bool = True, auto_layout: bool = True, x_range=None, rows=None, trace_kw: dict | None = None,
                 cost_mode: str | None = None, ml_model=None, ml_model_path=None, ml_weight: float | None = None):
        self.scan_path = str(scan_path); self.curve = curve; self.well = well or Path(scan_path).parent.name
        self.las_path = str(las_path) if las_path else None
        self.track = track; self.first_depth = first_depth; self.label_interval = label_interval
        self.use_ocr = use_ocr; self.auto_layout = auto_layout; self.x_range = x_range; self.rows_window = rows
        self.trace_kw = dict(trace_kw or {})
        if cost_mode is not None:
            self.trace_kw["cost_mode"] = cost_mode
        if ml_model is not None or ml_model_path is not None:
            self.trace_kw["ml_model"] = ml_model if ml_model is not None else ml_model_path
        if ml_weight is not None:
            self.trace_kw["ml_weight"] = ml_weight
        self.scale = dict(left=float(scale[0]), right=float(scale[1]), type=scale_type,
                          backup=list(backup) if backup is not None else None, wrap_side=wrap_side, use_hatch=bool(use_hatch))
        self.L = None; self.ink = None; self.errors: dict[str, str] = {}
        self.dm = None; self.depth_anchors: list[tuple[float, float]] = []; self.depth_checks: dict = {}; self.depth_source = ""
        self.depth_section: int | None = None; self.depth_sections: list[dict] = []   # log sections on the sheet
        self.section: dict | None = None; self.pan = None; self.width = 0; self._prob_map = None
        self.grid_lattice = None; self._pan_raw = None; self.grid_fit = None
        self.T: dict | None = None; self.anchors: list[tuple[float, float]] = []
        self.excluded: list[tuple[float, float]] = []    # "not this curve" clicks (sheet_row, x_panel)
        self.excluded_ids: list[int] = []; self.anchor_window_ft: dict = {}
        self.prefer_stroke = True; self.other_ink_cost = 0.35
        self.backup_intervals: list[tuple[float, float]] = []   # sheet-row intervals read on the backup scale
        self.values = None; self.wraps = []
        self.history: list[str] = []
        # guided fixing: guide strokes [[(sheet_row, x_panel), ..], ..], answered check spots, per-edit change log
        self.strokes: list[list[tuple[float, float]]] = []
        self.decisions: list[dict] = []
        self.change_log: list[dict] = []
        self.stroke_radius_px = 6.0; self.stroke_weight = 1.0
        self._undo: list[tuple[str, dict]] = []; self._spots = None

    # ------------------------------------------------------------------ open / layout
    def open(self, progress=None) -> "AssistedSession":
        """Load the sheet; auto layout + depth calibration + default section. Failures are kept in
        ``errors`` (layout / depth) so they can be fixed by hand (box section, depth clicks).
        ``progress(i, n, message)`` (optional) is called before each stage (used by the app)."""
        say = progress or (lambda *a: None)
        say(0, 3, "loading, deskewing and detecting tracks / depth labels")
        if self.auto_layout:
            try:
                self.L = wf.analyze_layout(self.scan_path, x_range=self.x_range, rows=self.rows_window)
                self.ink = self.L.ink
            except LayoutError as e:
                self.errors["layout"] = f"LayoutError: {e} - draw the track box by hand"
        if self.ink is None:
            x0, x1 = self.x_range or (None, None); r0, r1 = self.rows_window or (None, None)
            crop = None if self.x_range is None and self.rows_window is None else (x0 or 0, r0 or 0, x1, r1)
            if crop and (crop[2] is None or crop[3] is None):
                from pylogdigitizer.scan_io import read_info
                inf = read_info(self.scan_path); crop = (crop[0], crop[1], crop[2] or inf["width"], crop[3] or inf["height"])
            self.ink = load_ink(self.scan_path, crop)
        if self.L is not None:
            say(1, 3, "depth calibration" + (" (OCR of depth labels)" if self.use_ocr and dc.ocr_available(self.use_ocr if isinstance(self.use_ocr, str) else "auto") else ""))
            self._auto_depth()
            say(2, 3, "default log section")
            if self.track in (self.L.track_names if self.L else []):
                try:
                    self.set_section(track=self.track)
                except (SessionError, CalibrationError) as e:
                    self.errors["section"] = str(e)
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.ink.shape)

    def tracks(self) -> list[dict]:
        """Detected tracks as dict(name, left, right) at the median followed position (sheet px)."""
        if self.L is None:
            return []
        out = []
        for n in self.L.track_names + (["depth"] if "depth.left" in self.L.edge_names else []):
            li, ri = self.L.edge_names.index(n + ".left"), self.L.edge_names.index(n + ".right")
            out.append(dict(name=n, left=float(np.median(self.L.edges[:, li])), right=float(np.median(self.L.edges[:, ri]))))
        return out

    # ------------------------------------------------------------------ depth calibration
    def _auto_depth(self):
        try:
            # a sheet can hold several log sections (e.g. 2" and 5"): never guess - if the header does not
            # clearly say which one, it is picked by hand (set_depth_section)
            self.L, (dm, anc, res, checks) = wf._pick_section(self.L, self.track or "track1", self.first_depth,
                                                              self.label_interval, self.use_ocr, section=self.depth_section)
            self.dm, self.depth_anchors, self.depth_checks = dm, [(float(r), float(d)) for r, d in anc], checks
            self.depth_source = "auto (" + str(checks.get("source", "labels")) + ")"
            if checks.get("sections"):
                self.depth_sections = checks["sections"]
            self.errors.pop("depth", None); self.errors.pop("depth_section", None)
        except wf.SectionChoiceNeeded as e:
            self.depth_sections = e.sections
            self.errors["depth_section"] = str(e)
        except CalibrationError as e:
            self.errors["depth"] = f"CalibrationError: {e} - click >= 2 depth labels and type their depth"

    def set_depth_section(self, index: int):
        """Use log section ``index`` (see ``depth_sections``) for the depth scale."""
        if self.L is None or not 0 <= int(index) < len(self.L.sections):
            raise SessionError(f"no log section {index}")
        self.depth_section = int(index)
        self._auto_depth()
        if "depth" in self.errors:
            raise CalibrationError(self.errors["depth"])

    def set_depth_labels(self, first_depth: float | None = None, label_interval: float | None = None, use_ocr: bool | None = None):
        """Re-run the automatic label calibration with a typed first label value / interval."""
        if self.L is None:
            raise SessionError("no automatic layout: give depth anchors with set_depth_anchors")
        self.first_depth, self.label_interval = first_depth, label_interval
        if use_ocr is not None:
            self.use_ocr = use_ocr
        self._auto_depth()
        if "depth" in self.errors:
            raise CalibrationError(self.errors["depth"])
        self._after_depth_change()

    def set_depth_anchors(self, anchors: list[tuple[float, float]]):
        """Manual depth calibration: >= 2 (sheet_row, depth_ft) pairs (piecewise linear)."""
        a = sorted((float(r), float(d)) for r, d in anchors)
        if len(a) < 2:
            raise SessionError("need >= 2 depth anchors (row, depth)")
        if np.any(np.diff([d for _, d in a]) <= 0):
            raise SessionError("depth must increase with row (check typed depths)")
        self.dm = dc.fit_depth_map(a); self.depth_anchors = a; self.depth_source = "manual"
        self.depth_checks = dict(source="manual", n_anchors=len(a), px_per_ft=round(self.dm.px_per_ft, 3))
        self.errors.pop("depth", None); self._after_depth_change()

    def _after_depth_change(self):
        self.history.append("depth")

    # ------------------------------------------------------------------ scale
    def set_scale(self, left: float, right: float, scale_type: str = "linear", backup=None,
                  wrap_side: str = "right", use_hatch: bool | None = None):
        if scale_type not in ("linear", "log"):
            raise SessionError("scale_type must be 'linear' or 'log'")
        if scale_type == "log" and (left <= 0 or right <= 0):
            raise SessionError("log scale needs positive end values")
        self.scale = dict(left=float(left), right=float(right), type=scale_type,
                          backup=list(backup) if backup else None, wrap_side=wrap_side,
                          use_hatch=self.scale["use_hatch"] if use_hatch is None else bool(use_hatch))
        if self.T is not None:
            self._update_values()

    # ------------------------------------------------------------------ section
    def set_section(self, track: str | None = None, box: tuple[float, float] | None = None,
                    depth_range: tuple[float, float] | None = None, rows: tuple[float, float] | None = None):
        """Choose the log section: a detected ``track`` (boundaries followed down the sheet) or a
        hand-drawn ``box`` = (x_left, x_right) sheet px; vertical extent from ``rows`` (sheet px),
        ``depth_range`` (ft, needs depth calibration) or the default label extent. Clears the trace."""
        H = self.ink.shape[0]
        if rows is not None:
            r0, r1 = rows
        elif depth_range is not None:
            if self.dm is None:
                raise SessionError("depth_range needs a depth calibration first (or give rows)")
            r0, r1 = float(self.dm.inverse(depth_range[0])), float(self.dm.inverse(depth_range[1]))
        elif self.L is not None and len(self.L.labels) >= 3:
            r0, r1 = wf.section_rows(self.L, self.depth_anchors, self.depth_checks)
        else:
            r0, r1 = 0, H
        r0, r1 = int(max(0, min(r0, r1))), int(min(H, max(r0, r1)))
        if r1 - r0 < 50:
            raise SessionError(f"section rows {r0}-{r1} too short")
        if box is not None:
            x0, x1 = sorted(int(round(v)) for v in box)
            x0, x1 = max(0, x0), min(self.ink.shape[1], x1)
            if x1 - x0 < 20:
                raise SessionError("track box narrower than 20 px")
            self.section = dict(kind="box", x0=x0, x1=x1, rows=[r0, r1])
            self.pan = np.ascontiguousarray(self.ink[r0:r1, x0:x1]); self.width = x1 - x0
        else:
            if self.L is None or track not in self.L.track_names:
                raise SessionError(f"track {track!r} not detected; detected: {self.L.track_names if self.L else []} - draw a box")
            self.track = track; w = self.L.width_of(track)
            self.section = dict(kind="track", track=track, rows=[r0, r1], width=w)
            self.pan = self.L.panel(track, (r0, r1), w); self.width = w
        self.T = None; self._prob_map = None; self.anchors = []; self.excluded = []; self.backup_intervals = []; self.values = None; self.wraps = []
        self.grid_lattice = None; self._pan_raw = None; self.grid_fit = None
        self.strokes = []; self.decisions = []; self.change_log = []; self._undo = []; self._spots = None
        self.history.append("section")

    # ------------------------------------------------------------------ calibration editing (Stage 2)
    def adjust_track_edges(self, left_dx: float = 0.0, right_dx: float = 0.0):
        """Shift the followed left/right boundaries of the current track by whole-sheet offsets (px),
        or widen/narrow a hand-drawn box; re-cuts the panel (clears the trace)."""
        self._need_section()
        sec = dict(self.section)
        if sec["kind"] == "box":
            return self.set_section(box=(sec["x0"] + left_dx, sec["x1"] + right_dx), rows=tuple(sec["rows"]))
        n = sec["track"]
        li, ri = self.L.edge_names.index(n + ".left"), self.L.edge_names.index(n + ".right")
        E = np.array(self.L.edges, float); E[:, li] += float(left_dx); E[:, ri] += float(right_dx)
        if np.any(E[:, ri] - E[:, li] < 20):
            raise SessionError("track narrower than 20 px after the edit")
        self.L.edges = E
        self.edge_edits = getattr(self, "edge_edits", []) + [(n, float(left_dx), float(right_dx))]
        return self.set_section(track=n, rows=tuple(sec["rows"]))

    def grid_lines(self, min_frac: float = 0.5) -> list[float]:
        """Vertical grid-line centres in panel columns (for snapping scale clicks)."""
        from pylogdigitizer import track_grid_detection as tgd
        self._need_section()
        return [float(l["center"]) for l in tgd.vertical_lines(self.pan[: min(len(self.pan), 20000)], min_frac)]

    def scale_from_points(self, x1: float, v1: float, x2: float, v2: float, scale_type: str = "linear", **kw):
        """Manual scale: two panel columns with known values (e.g. two grid lines) -> values at the
        track edges (x = -0.5 and width - 0.5), linear or log10; then ``set_scale``."""
        if x1 == x2:
            raise SessionError("the two scale points need different x")
        f = (np.log10 if scale_type == "log" else (lambda v: v))
        if scale_type == "log" and (v1 <= 0 or v2 <= 0):
            raise SessionError("log scale needs positive values")
        a = (f(v2) - f(v1)) / (x2 - x1)
        g = lambda x: f(v1) + a * (x - x1)
        inv = (lambda y: float(10 ** y)) if scale_type == "log" else float
        left, right = inv(g(-0.5)), inv(g(self.width - 0.5))
        self.set_scale(round(left, 6), round(right, 6), scale_type, **kw)
        return left, right

    def depth_residuals(self) -> list[dict]:
        """Leave-one-out check of manual depth anchors: predicted depth of each anchor from the others."""
        from pylogdigitizer import depth_calibration as dcal
        a = list(self.depth_anchors); out = []
        for i, (r, d) in enumerate(a):
            rest = a[:i] + a[i + 1:]
            if len(rest) < 2:
                out.append(dict(row=r, depth=d, loo_ft=None)); continue
            pred = float(dcal.fit_depth_map(rest)(r))
            out.append(dict(row=r, depth=d, loo_ft=round(pred - d, 3)))
        return out

    def _edges(self, rows):
        rows = np.asarray(rows, float)
        if self.section["kind"] == "box":
            return np.full(rows.shape, float(self.section["x0"])), np.full(rows.shape, float(self.section["x1"]))
        n = self.section["track"]
        li, ri = self.L.edge_names.index(n + ".left"), self.L.edge_names.index(n + ".right")
        return np.interp(rows, self.L.centers, self.L.edges[:, li]), np.interp(rows, self.L.centers, self.L.edges[:, ri])

    def _sheet_to_raw(self, row, x):
        l, r = self._edges(row)
        if self.section["kind"] == "box":
            return np.asarray(x, float) - l
        return (np.asarray(x, float) - l) / (r - l) * self.width - 0.5

    def _raw_to_sheet(self, row, xp):
        l, r = self._edges(row)
        if self.section["kind"] == "box":
            return l + np.asarray(xp, float)
        return l + (np.asarray(xp, float) + 0.5) / self.width * (r - l)

    def sheet_to_panel(self, row, x):
        """Sheet (row, x) -> panel column (inverse of track_grid_detection.rectify, then of the
        grid lattice when the section is grid-rectified)."""
        xr = self._sheet_to_raw(row, x)
        lat = getattr(self, "grid_lattice", None)
        if lat is None:
            return xr
        rr = np.asarray(row, float) - self.section["rows"][0]
        return lat.to_rectified(np.broadcast_to(xr, np.broadcast(xr, rr).shape), np.broadcast_to(rr, np.broadcast(xr, rr).shape))[0]

    def panel_to_sheet(self, row, xp):
        lat = getattr(self, "grid_lattice", None)
        if lat is not None:
            rr = np.asarray(row, float) - self.section["rows"][0]
            b = np.broadcast(np.asarray(xp, float), rr).shape
            xp = lat.to_sheet(np.broadcast_to(np.asarray(xp, float), b), np.broadcast_to(rr, b))[0]
        return self._raw_to_sheet(row, xp)

    # ------------------------------------------------------------------ grid rectification + grid-line values (0.1.3)
    def rectify_grid(self, lattice=None, **kw) -> dict:
        """Detect the vertical grid lines of the current section, straighten them (x-warp per row,
        piecewise bilinear between lines; depth is left to the depth calibration) and re-cut the
        panel in rectified space. ``lattice`` (dict) applies an edited lattice instead. Clears the trace."""
        from pylogdigitizer import grid_rectify as gr
        self._need_section()
        raw = getattr(self, "_pan_raw", None)
        if raw is None:
            raw = self.pan
        if lattice is not None:
            lat = gr.Lattice.from_dict(lattice) if isinstance(lattice, dict) else lattice
        else:
            lat = gr.detect_lattice(raw, **kw)
            if lat is None or lat.X.shape[1] < 3:
                raise SessionError("fewer than 3 vertical grid lines found in this section")
        lat = gr.x_only(lat, raw.shape)   # depth stays with the depth calibration
        before = lat.residual()
        self._pan_raw = raw
        self.pan = np.ascontiguousarray(gr.rectify_image(raw, lat, raw.shape))
        self.grid_lattice = lat
        self.T = None; self._prob_map = None; self.anchors = []; self.excluded = []; self.values = None; self.wraps = []
        self.strokes = []; self.decisions = []; self.change_log = []; self._undo = []; self._spots = None
        self.history.append("rectify_grid")
        return dict(lines=lat.u.tolist(), n_rows=int(len(lat.v)), warp_px=before)

    def unrectify_grid(self):
        if getattr(self, "_pan_raw", None) is not None:
            self.pan = self._pan_raw; self._pan_raw = None; self.grid_lattice = None
            self.T = None; self.values = None; self.history.append("unrectify_grid")

    def grid_values(self, labels=None, kind: str | None = None, apply: bool = True) -> dict:
        """Value of every vertical grid line from labels [(panel_x, value)] (default: the current
        scale ends, so the detected lines get their implied values), with consistency warnings.
        ``apply`` sets the track scale from the fit (exact at the edges in rectified space)."""
        from pylogdigitizer import grid_values as gv
        self._need_section()
        lx = self.grid_lines() if getattr(self, "grid_lattice", None) is None else list(self.grid_lattice.u)
        sc_ = self.scale
        if not labels:
            labels = [(-0.5, sc_["left"]), (self.width - 0.5, sc_["right"])]
            kind = kind or sc_["type"]
        if len(lx) < 2:   # no clean straight lines (warped, unrectified): the labels are the knots
            lx = sorted(float(x) for x, _ in labels)
        fit = gv.fit_line_values(lx, labels, kind)
        if fit is None:
            raise SessionError("need at least 2 grid-line labels with values")
        ks = gv.knot_scale(fit) if len(lx) >= 2 else None
        # edges by the fitted model (not the knots), so the scale stays the global fit
        c = fit["coef"]; f = (lambda x: c[0] + c[1] * x) if fit["kind"] == "linear" else (lambda x: 10 ** (c[0] + c[1] * x))
        left, right = float(f(-0.5)), float(f(self.width - 0.5))
        fit.update(left=left, right=right, warnings=gv.check_divisions(fit))
        if apply:
            self.set_scale(round(left, 6), round(right, 6), fit["kind"], backup=sc_["backup"], wrap_side=sc_["wrap_side"])
        self.grid_fit = fit
        return fit

    # ------------------------------------------------------------------ tracing
    def _need_section(self):
        if self.pan is None:
            raise SessionError("select a section first (set_section: track or box)")

    def auto_trace(self, seed: tuple[float, float] | None = None, anchors_panel: list | None = None,
                   excluded: list | None = None, strokes: list | None = None):
        """Trace the whole section. ``seed`` = (sheet_row, sheet_x) click on the curve (becomes the
        first anchor). ``anchors_panel`` = saved anchors [(sheet_row, x_panel)] to replay."""
        self._need_section()
        r0 = self.section["rows"][0]
        anc = [tuple(map(float, a)) for a in (anchors_panel or [])]
        if seed is not None:
            anc.append((float(seed[0]), float(self.sheet_to_panel(seed[0], seed[1]))))
        ppf = self.dm.px_per_ft if self.dm is not None else None
        tkw = dict(self.trace_kw)
        if self._prob_map is not None and tkw.get("cost_mode", "conventional") != "conventional":
            tkw.setdefault("prob_map", self._prob_map)
        nc = int(tkw.pop("n_curves", 1) or 1); pick = tkw.pop("pick", "auto")
        if nc > 1:   # several curves in the track: trace each, keep ``pick`` (index or "auto")
            self.T = ct.trace_multi(self.pan, r0, n_curves=nc, pick=pick, px_per_ft=ppf, anchors_px=anc or None, **tkw)
        else:
            self.T = ct.trace_track(self.pan, r0, px_per_ft=ppf, anchors_px=anc or None, **tkw)
        if "prob_map" in self.T:
            self._prob_map = self.T["prob_map"]
        self.T["cost0"] = self.T["cost"].copy(); self.excluded_ids = []
        self.anchors = sorted(anc)
        self.excluded = []; self._spots = None
        self.strokes = [[(float(r), float(x)) for r, x in st] for st in (strokes or []) if len(st) >= 2]
        for e in excluded or []:
            self._exclude(e)
        if self.excluded or self.strokes or (self.prefer_stroke and self.anchors):
            self._rebuild_cost(); self._retrace_rows(0, len(self.T["path_padded"]) - 1)
        if seed is not None or anchors_panel:
            self.anchors = [self._snapped(a) if 0 <= a[1] < self.width else a for a in self.anchors]
        self._update_values(); self.history.append("auto_trace")
        return self

    def _dp(self, sheet_row) -> int:
        T = self.T
        return int(np.clip(round((sheet_row - T["row0"]) / T["fy"] - 0.5), 0, len(T["path_padded"]) - 1))

    def _dpx(self, x_panel) -> float:
        """Panel column -> padded DP column; beyond the track edges = the virtual off-scale columns
        (an anchor there tells the tracer the curve has left the track, i.e. wraps)."""
        T = self.T; ncol = T["cost"].shape[1]
        if T["pad"] and x_panel < 0:
            return 0.0
        if T["pad"] and x_panel >= self.width:
            return float(ncol - 1)
        return float(np.clip(x_panel / T["fx"] + T["pad"], 0, ncol - 1))

    def _snapped(self, a, radius_px: int = 8):
        """Move an anchor's x to the nearest curve-ink column within ``radius_px`` (same DP row)."""
        k = self._dp(a[0]); C = self.T["cost"]; x = self._dpx(a[1])
        L = self.T["labels"]
        c = ((C[max(0, k - 3):k + 4] == 0.0) | (L[max(0, k - 3):k + 4] > 0)).any(axis=0)    # +-3 DP rows: clicks on a grid-line row still snap
        rad = max(1, radius_px // self.T["fx"]); lo, hi = int(max(0, x - rad)), int(min(len(c) - 1, x + rad))
        z = np.flatnonzero(c[lo:hi + 1])
        if len(z):
            xb = lo + z[np.argmin(np.abs(lo + z - x))]
            return (a[0], float((xb - self.T["pad"]) * self.T["fx"] + self.T["fx"] / 2))
        return (float(a[0]), float(a[1]))

    def add_anchor(self, row: float, x: float, snap: bool = True, window_ft: float = 150.0, panel_x: bool = False):
        """Click on the curve at sheet (row, x): pin the trace there and re-trace locally.
        ``panel_x``: x is already a panel column. Returns the stored (sheet_row, x_panel)."""
        if self.T is None:
            self.auto_trace()
        xp = float(x) if panel_x else float(self.sheet_to_panel(row, x))
        lim = 0.15 * self.width
        if not (-lim <= xp <= self.width + lim):
            raise SessionError(f"anchor x {x:.0f} is outside the section (click within the track; just outside an "
                               "edge = curve off-scale/wrapped there)")
        off = xp < 0 or xp >= self.width
        a = self._snapped((float(row), xp)) if snap and not off else (float(row), xp)
        self.anchors = sorted([b for b in self.anchors if abs(b[0] - a[0]) > self.T["fy"]] + [a])
        self.anchor_window_ft[a] = float(window_ft)
        if self.prefer_stroke:
            self._rebuild_cost()
        self._retrace_around(a[0], window_ft)
        self.history.append("anchor")
        return a

    def _window_rows(self, sheet_row: float, window_ft: float) -> tuple[int, int]:
        T = self.T; h = len(T["path_padded"]); k = self._dp(sheet_row)
        ppf = self.dm.px_per_ft if self.dm is not None else 10.0
        win = max(20, int(window_ft * ppf / T["fy"]))
        ka = [self._dp(r) for r, _ in self.anchors]
        prev = [q for q in ka if q < k]; nxt = [q for q in ka if q > k]
        return max(prev[-1] if prev else 0, k - win, 0), min(nxt[0] if nxt else h - 1, k + win, h - 1)

    def _rebuild_cost(self):
        """cost = original cost, minus excluded strokes, with "follow the clicked stroke": within each
        anchor's re-trace window, curve ink that is NOT the anchored stroke costs ``other_ink_cost``
        (a click on a curve means "this stroke is the curve")."""
        T = self.T; c = T["cost0"].copy(); lab = T["labels"]
        for lid in self.excluded_ids:
            c[lab == lid] = 1.0
        if self.prefer_stroke:
            for r, x in self.anchors:
                k = self._dp(r); xi = int(round(self._dpx(x)))
                lo, hi = max(0, xi - 2), min(lab.shape[1], xi + 3)
                ids = lab[max(0, k - 4):k + 5, lo:hi].ravel(); ids = ids[ids > 0]
                if not len(ids):
                    continue
                lid = np.bincount(ids).argmax()
                a, b = self._window_rows(r, self.anchor_window_ft.get((r, x), 150.0))
                sl = c[a:b + 1]; ls = lab[a:b + 1]
                has_lid = (ls == lid).any(axis=1)[:, None]
                sl[(ls > 0) & (ls == lid)] = 0.0
                sl[has_lid & (ls > 0) & (ls != lid) & (sl < self.other_ink_cost)] = self.other_ink_cost
        for st in self.strokes:
            k0, k1, pen = self._stroke_penalty(st)
            c[k0:k1 + 1] += pen
        T["cost"] = c

    def _stroke_penalty(self, st):
        """Guide stroke -> (k0, k1, penalty[k0:k1+1, ncol]): 0 within ``stroke_radius_px`` of the stroke,
        rising to ``stroke_weight`` 3 radii further out (the tracer stays close to the stroke there)."""
        T = self.T; pts = sorted((self._dp(r), self._dpx(x)) for r, x in st)
        ks = np.array([p[0] for p in pts], float); xs = np.array([p[1] for p in pts], float)
        k0, k1 = int(ks[0]), int(ks[-1]); rows = np.arange(k0, k1 + 1)
        xi = np.interp(rows, ks, xs); cols = np.arange(T["cost0"].shape[1])
        d = np.abs(cols[None, :] - xi[:, None]) * T["fx"]
        r = self.stroke_radius_px
        return k0, k1, np.clip((d - r) / (3 * r), 0, 1) * self.stroke_weight

    def remove_anchor(self, index: int, window_ft: float = 150.0):
        a = self.anchors.pop(index)
        if self.prefer_stroke:
            self._rebuild_cost()
        self._retrace_around(a[0], window_ft); self.history.append("remove_anchor")
        return a

    def clear_anchors(self):
        """Remove all anchors and exclusions; fresh trace reusing cached section cost."""
        self.anchors = []; self.excluded = []; self.excluded_ids = []
        if self.T is None:
            return self.auto_trace()
        self._rebuild_cost()
        self._retrace_rows(0, len(self.T["path_padded"]) - 1)
        self.history.append("clear_anchors")
        return self

    def set_cost_mode(self, cost_mode: str = "conventional", ml_model=None,
                      ml_weight: float | None = None, caliper_weight: float | None = None):
        """Switch between 'conventional', 'ml', and 'blend' cost maps in-place and re-trace."""
        self.trace_kw["cost_mode"] = cost_mode
        if ml_model is not None:
            self.trace_kw["ml_model"] = ml_model
        if ml_weight is not None:
            self.trace_kw["ml_weight"] = float(ml_weight)
        if caliper_weight is not None:
            self.trace_kw["caliper_weight"] = float(caliper_weight)
        if self.T is None or "cost_conv" not in self.T:
            return self.auto_trace(anchors_panel=self.anchors or None, excluded=self.excluded or None)
        cost_conv = self.T["cost_conv"]
        fy, fx = self.T["fy"], self.T["fx"]
        if self._prob_map is None:
            cost_mode = ct.ml_fallback(cost_mode, self.trace_kw.get("ml_model"))
        if cost_mode == "conventional":
            cost_base = cost_conv.copy()
        else:
            import cv2
            from pylogdigitizer import ml_models
            H_dp, W_dp = cost_conv.shape
            if self._prob_map is None or self.T.get("_v61") is None:
                resolved_model = ct._resolve_default_ml_model(self.trace_kw.get("ml_model"))
                if fy < 3 and H_dp >= 24:
                    k_ml = max(1, 3 // fy)
                    h3 = H_dp // k_ml
                    cost3 = cost_conv[: h3 * k_ml].reshape(h3, k_ml, W_dp).min(axis=1)
                    feat3 = ml_models.extract_track_features(self.pan, cost3, fy=fy * k_ml, fx=fx)
                    prob3 = ml_models.predict_track_probabilities(
                        self.pan, cost3, model=resolved_model, fy=fy * k_ml, fx=fx, feat=feat3
                    )
                    self._prob_map = cv2.resize(prob3, (W_dp, H_dp), interpolation=cv2.INTER_LINEAR)
                    self.T["_v61"] = cv2.resize(feat3[2], (W_dp, H_dp), interpolation=cv2.INTER_LINEAR)
                    self.T["_rel_x"] = np.broadcast_to(np.linspace(0.0, 1.0, W_dp, dtype=np.float32)[None, :], (H_dp, W_dp))
                else:
                    feat = ml_models.extract_track_features(self.pan, cost_conv, fy=fy, fx=fx)
                    self._prob_map = ml_models.predict_track_probabilities(
                        self.pan, cost_conv, model=resolved_model, fy=fy, fx=fx, feat=feat
                    )
                    self.T["_v61"], self.T["_rel_x"] = feat[2], feat[6]
            self.T["prob_map"] = self._prob_map
            cost_base = ct.build_ml_cost(
                cost_conv,
                self._prob_map,
                v61=self.T["_v61"],
                rel_x=self.T["_rel_x"],
                mode=cost_mode,
                ml_weight=float(self.trace_kw.get("ml_weight", 0.15)),
                caliper_weight=float(self.trace_kw.get("caliper_weight", 0.55)),
            )
        pad = self.T["pad"]
        self.T["cost0"] = np.pad(cost_base, ((0, 0), (1, 1)), constant_values=0.8) if pad else cost_base
        self.T["cost_mode"] = cost_mode
        self.T["params"]["cost_mode"] = cost_mode
        self._rebuild_cost()
        self._retrace_rows(0, len(self.T["path_padded"]) - 1)
        if self._prob_map is not None:
            p_along = self._prob_map[np.arange(len(self.T["path"])), self.T["path"]]
            self.T["ml_conf"] = np.clip(np.convolve(p_along, np.ones(15) / 15.0, mode="same"), 0.0, 1.0)
        self.history.append(f"cost_mode:{cost_mode}")
        return self

    def retrace_all(self):
        """Global re-trace of the whole section with all current anchors and exclusions."""
        if self.T is None:
            return self.auto_trace()
        self._retrace_rows(0, len(self.T["path_padded"]) - 1); self.history.append("retrace_all")

    def exclude_stroke(self, row: float, x: float, panel_x: bool = False, radius_px: int = 10):
        """"Not this curve": click on a stroke the trace wrongly follows (another curve, grid,
        annotation). That whole connected stroke is made unattractive (paper cost) and the rows it
        spans are re-traced (anchors respected). Returns the stroke's sheet-row extent or None."""
        if self.T is None:
            self.auto_trace()
        xp = float(x) if panel_x else float(self.sheet_to_panel(row, x))
        ext = self._exclude((float(row), xp), radius_px)
        if ext is None:
            raise SessionError("no curve stroke under the click (click on the ink of the wrong curve)")
        k0, k1 = ext; h = len(self.T["path_padded"])
        self._retrace_rows(max(0, k0 - 5), min(h - 1, k1 + 5)); self.history.append("exclude")
        return float(self.T["rows"][k0]), float(self.T["rows"][k1])

    def _exclude(self, e, radius_px: int = 10):
        T = self.T; k = self._dp(e[0]); x = self._dpx(e[1]); lab = T["labels"][k]
        rad = max(1, radius_px // T["fx"]); lo, hi = int(max(0, x - rad)), int(min(len(lab) - 1, x + rad))
        z = np.flatnonzero(lab[lo:hi + 1] > 0)
        if not len(z):
            return None
        lid = int(lab[lo + z[np.argmin(np.abs(lo + z - x))]])
        m = T["labels"] == lid
        self.excluded.append((float(e[0]), float(e[1]))); self.excluded_ids.append(lid)
        self._rebuild_cost()
        rr = np.flatnonzero(m.any(axis=1))
        return int(rr[0]), int(rr[-1])

    def remove_exclusions(self):
        """Undo all "not this curve" clicks (restores the cost) and re-trace globally."""
        if self.T is not None:
            self.excluded = []; self.excluded_ids = []; self._rebuild_cost(); self.retrace_all()

    def _retrace_around(self, sheet_row: float, window_ft: float):
        self._retrace_rows(*self._window_rows(sheet_row, window_ft))

    def _retrace_rows(self, a: int, b: int):
        """Re-run the DP tracer on DP rows a..b with all anchors inside; the window ends are pinned to
        the current path (unless at the section ends) so nothing outside moves."""
        T = self.T; P = T["path_padded"].copy(); h = len(P)
        cons = [(self._dp(r) - a, self._dpx(x)) for r, x in self.anchors if a <= self._dp(r) <= b]
        if a > 0 and not any(c[0] == 0 for c in cons):
            cons.append((0, float(P[a])))
        if b < h - 1 and not any(c[0] == b - a for c in cons):
            cons.append((b - a, float(P[b])))
        p = T["params"]
        sub = ct.trace(T["cost"][a:b + 1], max_step=p["max_step"], smooth=p["smooth"], seg_weight=p["seg_weight"],
                       anchors=cons, anchor_tol=1, labels=T["labels"][a:b + 1], switch_cost=p["switch_cost"], use=p["use"])
        P[a:b + 1] = sub
        self._set_path(P)
        self.last_retrace = dict(dp_rows=(int(a), int(b)), sheet_rows=(float(T["rows"][a]), float(T["rows"][b])))

    def _set_path(self, P):
        T = self.T
        N = ct.finish_trace(T["cost"], P, T["row0"], T["fy"], T["fx"], T["pad"], T["params"]["edge_frac"])
        for k_extra in ("prob_map", "ml_conf", "cost_conv", "cost_mode", "_v61", "_rel_x", "_hatch_rows", "_hatch_rows_key"):
            if k_extra in T:
                N[k_extra] = T[k_extra]
        N.update(labels=T["labels"], params=T["params"], cost0=T.get("cost0", T["cost"])); self.T = N
        self._update_values()

    def _update_values(self):
        s = self.scale
        self.values, self.wraps = wf.trace_to_values(self.pan, self.T, self.width, (s["left"], s["right"]), s["type"],
                                                     s["backup"], s["wrap_side"], s["use_hatch"])
        if self.backup_intervals and s["backup"]:
            mk = sc.log_scale if s["type"] == "log" else sc.linear_scale
            bm = mk(0, self.width, *s["backup"]); rows = self.T["rows"]; v = np.array(self.values, float)
            for r0, r1 in self.backup_intervals:
                m = (rows >= r0) & (rows <= r1) & ~(self.T["off_left"] | self.T["off_right"])
                v[m] = bm(self.T["x"][m])
            self.values = v

    def set_backup_interval(self, row0: float, row1: float):
        """Mark sheet rows row0..row1 as "curve drawn on the backup scale here" (the curve wrapped and
        re-entered the track; automatic wrap detection missed it). Needs a backup scale."""
        if not self.scale["backup"]:
            raise SessionError("set a backup scale first (scale panel)")
        a, b = sorted((float(row0), float(row1)))
        self.backup_intervals = sorted(self.backup_intervals + [(a, b)])
        if self.T is not None:
            self._update_values()
        self.history.append("backup_interval")

    def clear_backup_intervals(self):
        self.backup_intervals = []
        if self.T is not None:
            self._update_values()

    # ------------------------------------------------------------------ guided fixing
    # Every user edit (pin, move, stroke, answer) goes through _edit: an undo snapshot is kept and the
    # change it made to the trace is logged (progress() turns that into "more clicks still help?").
    GUIDED_ACTIONS = ("pin", "unpin", "move", "stroke", "unstroke", "answer", "circle")

    def _state(self) -> dict:
        return dict(anchors=list(self.anchors), anchor_window_ft=dict(self.anchor_window_ft), excluded=list(self.excluded),
                    excluded_ids=list(self.excluded_ids), strokes=[list(s) for s in self.strokes],
                    decisions=[dict(d) for d in self.decisions], backup_intervals=list(self.backup_intervals),
                    path=self.T["path_padded"].copy(), n_log=len(self.change_log), circles=list(getattr(self, "circles", [])))

    def _edit(self, label: str, fn):
        if self.T is None:
            self.auto_trace()
        snap = self._state(); x0 = np.asarray(self.T["x"], float).copy()
        try:
            out = fn()
        except Exception:
            self._restore(snap)
            raise
        self._undo.append((label, snap)); self._spots = None
        self._log_change(label, x0)
        return out

    def _restore(self, snap: dict):
        for k in ("anchors", "anchor_window_ft", "excluded", "excluded_ids", "strokes", "decisions", "backup_intervals"):
            setattr(self, k, snap[k])
        self.circles = list(snap.get("circles", []))
        del self.change_log[snap["n_log"]:]
        self._rebuild_cost(); self._set_path(snap["path"]); self._spots = None

    def undo(self) -> str | None:
        """Undo the last guided edit (pin, move, stroke, answer). Returns its label or None."""
        if not self._undo:
            return None
        label, snap = self._undo.pop()
        self._restore(snap); self.history.append("undo")
        return label

    def _section_ft(self) -> float:
        ppf = self.dm.px_per_ft if self.dm is not None else 10.0
        return float(len(self.T["x"]) * self.T["fy"] / ppf)

    def _log_change(self, label: str, x0) -> dict:
        x1 = np.asarray(self.T["x"], float); d = np.abs(x1 - x0); d[~np.isfinite(d)] = 0.0
        moved = d > max(3.0, 2.0 * float(self.T.get("fx", 1)))   # ignore one-column jitter along the ink
        ppf = self.dm.px_per_ft if self.dm is not None else 10.0
        e = dict(action=label, changed_ft=round(float(moved.sum() * self.T["fy"] / ppf), 1),
                 changed_frac=round(float(moved.mean()), 4), max_px=round(float(d.max()), 1),
                 mean_px=round(float(d[moved].mean()) if moved.any() else 0.0, 1))
        self.change_log.append(e)
        return e

    def progress(self) -> dict:
        """How much the last click changed the trace, and whether further clicks still help
        (settled = the last two edits each moved <= max(5 ft, 0.5 % of the section))."""
        log = [e for e in self.change_log if e["action"] in ("pin", "move", "stroke", "answer")]
        if not log:
            return dict(last=None, recent=[], settled=False, small_ft=None,
                        message="Click on the real curve where the red trace is wrong.")
        small_ft = max(5.0, 0.005 * self._section_ft()); last = log[-1]
        small = lambda e: e["changed_ft"] <= small_ft
        settled = len(log) >= 2 and small(log[-1]) and small(log[-2])
        if settled:
            msg = "The last clicks changed very little - the trace has probably settled. Check it once more, then continue."
        elif small(last):
            msg = f"That click changed only {last['changed_ft']:.0f} ft of the trace."
        else:
            msg = f"That click moved {last['changed_ft']:.0f} ft of the trace (up to {last['max_px']:.0f} px)."
        return dict(last=last, recent=[e["changed_ft"] for e in log[-12:]], settled=settled, small_ft=round(small_ft, 1), message=msg)

    def pin(self, row: float, x: float, snap: bool = True, panel_x: bool = False, window_ft: float = 150.0):
        """Guided "Fix the trace" click: the trace must pass through (row, x) (hard constraint);
        only the stretch to the neighbouring pins is re-traced. Undoable (``undo``)."""
        return self._edit("pin", lambda: self.add_anchor(row, x, snap=snap, window_ft=window_ft, panel_x=panel_x))

    def circle_retrace(self, row: float, x: float, radius_px: float = 60.0, through: bool = False, panel_x: bool = False):
        """Cursor-circle fix: re-trace only the rows inside a circle (centre (row, x), sheet px), with the
        search limited to the circle's columns; the path enters and leaves at the current trace, so
        nothing outside the circle moves. ``through=True`` also pins the centre. Undoable."""
        def f():
            T = self.T; fy, fx = T["fy"], T["fx"]
            xp = float(x) if panel_x else float(self.sheet_to_panel(row, x))
            c = self._dp(row); rr = max(2, int(round(radius_px / fy)))
            a, b = max(0, c - rr), min(len(T["path_padded"]) - 1, c + rr)
            if b - a < 2:
                raise SessionError("circle too small")
            cx, rc = self._dpx(xp), radius_px / fx
            cost = T["cost"][a:b + 1].copy(); cols = np.arange(cost.shape[1], dtype=float)
            for k in range(cost.shape[0]):
                dy = (a + k - c) * fy
                half = np.sqrt(max(radius_px ** 2 - dy ** 2, 0.0)) / fx
                cost[k, np.abs(cols - cx) > max(half, 1.0)] += 50.0       # outside the circle: effectively forbidden
            P = T["path_padded"].copy()
            cons = [(0, float(P[a])), (b - a, float(P[b]))] if 0 < a or b < len(P) - 1 else []
            cons = [cc for cc in cons if not (cc[0] == 0 and a == 0) and not (cc[0] == b - a and b == len(P) - 1)]
            if through:
                cons.append((c - a, cx))
            p = T["params"]
            sub = ct.trace(cost, max_step=p["max_step"], smooth=p["smooth"], seg_weight=p["seg_weight"], anchors=cons,
                           anchor_tol=1, labels=T["labels"][a:b + 1], switch_cost=p["switch_cost"], use=p["use"])
            P[a:b + 1] = sub; self._set_path(P); self._update_values()
            self.circles = getattr(self, "circles", []) + [(float(row), float(xp), float(radius_px), bool(through))]
            self.last_retrace = dict(dp_rows=(int(a), int(b)), sheet_rows=(float(T["rows"][a]), float(T["rows"][b])))
            return self.last_retrace
        return self._edit("circle", f)

    def unpin(self, index: int):
        if not 0 <= index < len(self.anchors):
            raise SessionError(f"no pin {index}")
        return self._edit("unpin", lambda: self.remove_anchor(index, self.anchor_window_ft.get(self.anchors[index], 150.0)))

    def move_pin(self, index: int, row: float, x: float, snap: bool = True, panel_x: bool = False):
        """Drag pin ``index`` to (row, x): both the old and the new stretch are re-traced."""
        if not 0 <= index < len(self.anchors):
            raise SessionError(f"no pin {index}")
        def f():
            a = self.anchors.pop(index); w = self.anchor_window_ft.pop(a, 150.0)
            self._rebuild_cost(); self._retrace_around(a[0], w)
            return self.add_anchor(row, x, snap=snap, window_ft=w, panel_x=panel_x)
        return self._edit("move", f)

    def add_stroke(self, points, panel_x: bool = False):
        """Guide stroke: [(row, x), ..] dragged along a hard stretch of the curve; the tracer stays
        within a few px of it there (soft: pins still win). Re-traces the rows it spans."""
        if self.T is None:
            self.auto_trace()
        pts = [(float(r), float(x) if panel_x else float(self.sheet_to_panel(r, x))) for r, x in points]
        if len(pts) < 2 or len({self._dp(r) for r, _ in pts}) < 2:
            raise SessionError("drag the stroke along the curve (at least a few pixels up or down)")
        pts = sorted(pts)
        def f():
            self.strokes.append(pts); self._rebuild_cost()
            ks = [self._dp(r) for r, _ in pts]; h = len(self.T["path_padded"]); m = 10
            self._retrace_rows(max(0, min(ks) - m), min(h - 1, max(ks) + m))
            return pts
        return self._edit("stroke", f)

    def remove_stroke(self, index: int):
        if not 0 <= index < len(self.strokes):
            raise SessionError(f"no stroke {index}")
        def f():
            st = self.strokes.pop(index); self._rebuild_cost()
            ks = [self._dp(r) for r, _ in st]; h = len(self.T["path_padded"])
            self._retrace_rows(max(0, min(ks) - 10), min(h - 1, max(ks) + 10))
        return self._edit("unstroke", f)

    # ---- spots to check: where an alternative path disagrees with the trace (not the confidence)
    def _alt_path(self):
        """Second-best path: same cost, pins and parameters, but the current path's corridor costs +0.6."""
        T = self.T; P = T["path_padded"]; C = T["cost"]; w = C.shape[1]; p = T["params"]
        band = max(1, int(round(6 / T["fx"])))
        C2 = C + np.where(np.abs(np.arange(w)[None, :] - P[:, None]) <= band, 0.6, 0.0)
        cons = [(self._dp(r), self._dpx(x)) for r, x in self.anchors]
        return ct.trace(C2, max_step=p["max_step"], smooth=p["smooth"], seg_weight=p["seg_weight"], anchors=cons or None,
                        anchor_tol=1, labels=T["labels"], switch_cost=p["switch_cost"], use=p["use"])

    def _col_to_panel(self, col: float) -> float:
        T = self.T; ncol = T["cost"].shape[1]
        if T["pad"] and col < T["pad"]:
            return -0.05 * self.width
        if T["pad"] and col >= ncol - T["pad"]:
            return 1.05 * self.width
        return float((col - T["pad"]) * T["fx"] + T["fx"] / 2)

    def check_spots(self, max_gap: float = 0.25, max_spots: int = 30) -> list[dict]:
        """Places worth a look, with a plain-language question and the candidates:
        'crossover' = an almost equally good other line runs elsewhere here (crossing / touching curves);
        'wrap' = the trace leaves the track (off-scale). Answered spots are not asked again.
        Also sets ``disagreement`` (px per DP row, 0 = no doubt) used to highlight the trace."""
        if self._spots is not None:
            return self._spots
        T = self.T; P = T["path_padded"]; C = T["cost"]; fy = T["fy"]
        ppf = self.dm.px_per_ft if self.dm is not None else 10.0
        alt = self._alt_path(); dx = np.abs(alt - P).astype(float) * T["fx"]
        far = dx > max(8.0, 0.03 * self.width)
        from pylogdigitizer import scale_wraparound as sw
        dis = np.zeros(len(P)); spots = []
        done = [(d["k0"], d["k1"]) for d in self.decisions]
        seen = lambda a, b: any(a <= d1 and b >= d0 for d0, d1 in done)
        for a, b in sw.detect_wraps(far, max(3, int(round(3 * ppf / fy)))):
            ii = np.arange(a, b); gap = float(C[ii, alt[ii]].mean() - C[ii, P[ii]].mean())
            if gap > max_gap:
                continue
            dis[a:b] = dx[a:b]
            if seen(a, b):
                continue
            k = a + int(np.argmax(dx[a:b])); xs = sorted([("trace", float(P[k])), ("other", float(alt[k]))], key=lambda t: t[1])
            spots.append(dict(kind="crossover", k0=int(a), k1=int(b), k=int(k), score=(b - a) * fy / ppf * (1 - max(gap, 0) / max_gap),
                              question="Does the curve go left or right here?",
                              choices=[dict(id=side, label=f"{side.capitalize()} line" + (" (the current trace)" if src == "trace" else ""),
                                            x_panel=self._col_to_panel(c), current=src == "trace") for side, (src, c) in zip(("left", "right"), xs)]))
        self.disagreement = dis
        off = T["off_left"] | T["off_right"]
        for a, b in sw.detect_wraps(off, max(3, int(round(3 * ppf / fy)))):
            if seen(a, b):
                continue
            k = (a + b) // 2; right = bool(T["off_right"][k]); ncol = C.shape[1]; pad = T["pad"]
            lo, hi = pad, ncol - pad
            ink = np.flatnonzero(C[k, lo:hi] <= 0.05) + lo
            inside = None
            if len(ink):
                c = ink[-1] if right else ink[0]
                if abs(c - (hi - 1 if right else lo)) * T["fx"] < 0.5 * self.width:
                    inside = float(c)
            ch = [dict(id="off", label=f"Yes - it goes off the {'right' if right else 'left'} edge (wraps around)",
                       x_panel=1.05 * self.width if right else -0.05 * self.width, current=True)]
            if inside is not None:
                ch.append(dict(id="inside", label="No - it stays inside the track", x_panel=self._col_to_panel(inside), current=False))
            spots.append(dict(kind="wrap", k0=int(a), k1=int(b), k=int(k), score=(b - a) * fy / ppf,
                              question=f"The trace runs off the {'right' if right else 'left'} edge here. Does the real curve leave the track?",
                              choices=ch))
        spots = sorted(sorted(spots, key=lambda s: -s["score"])[:max_spots], key=lambda s: s["k"])
        for i, s in enumerate(spots):
            r = float(T["rows"][s["k"]])
            s.update(id=i, row=r, row0=float(T["rows"][s["k0"]]), row1=float(T["rows"][min(s["k1"], len(P) - 1)]), score=round(s["score"], 1))
            for c in s["choices"]:
                c["x"] = float(self.panel_to_sheet(r, c["x_panel"]))
        self._spots = spots
        return spots

    def answer_spot(self, spot_id: int, choice: str):
        """Answer a check spot: pins the chosen candidate (or 'skip' = leave it as is) and records the
        decision (saved with the session and in the training data)."""
        sp = next((s for s in self.check_spots() if s["id"] == spot_id), None)
        if sp is None:
            raise SessionError(f"no spot {spot_id} (the list changed - reload it)")
        dec = dict(kind=sp["kind"], k0=sp["k0"], k1=sp["k1"], row=sp["row"], choice=choice)
        if choice == "skip":
            return self._edit("answer", lambda: self.decisions.append(dec))
        c = next((c for c in sp["choices"] if c["id"] == choice), None)
        if c is None:
            raise SessionError(f"choice must be one of {[c['id'] for c in sp['choices']] + ['skip']}")
        off = c["x_panel"] < 0 or c["x_panel"] >= self.width
        def f():
            a = self.add_anchor(sp["row"], c["x_panel"], snap=not off, panel_x=True)
            dec["x_panel"] = a[1]; self.decisions.append(dec); return a
        return self._edit("answer", f)

    # ------------------------------------------------------------------ results
    def result(self) -> dict:
        if self.T is None:
            raise SessionError("no trace yet (auto_trace)")
        if self.dm is None:
            raise SessionError("no depth calibration (set_depth_anchors)")
        return dict(depth=self.dm(self.T["rows"]), value=np.asarray(self.values, float), conf=self.T["conf"],
                    rows=self.T["rows"], x_panel=self.T["x"], off_scale=self.T["off_left"] | self.T["off_right"])

    def overlay(self, row_range=None, max_points: int = 6000) -> dict:
        """Trace polyline in sheet coords (decimated), anchors and depth anchors - for drawing."""
        out = dict(anchors=[], depth_anchors=[list(a) for a in self.depth_anchors], trace=[], section=self.section)
        if self.T is None:
            return out
        rows, xp = self.T["rows"], self.T["x"]
        i0, i1 = 0, len(rows)
        if row_range:
            i0, i1 = np.searchsorted(rows, row_range[0]), np.searchsorted(rows, row_range[1])
        st = max(1, (i1 - i0) // max_points)
        rr = rows[i0:i1:st]; xs = self.panel_to_sheet(rr, xp[i0:i1:st])
        off = (self.T["off_left"] | self.T["off_right"])[i0:i1:st]; cf = self.T["conf"][i0:i1:st]
        out["trace"] = [[round(float(a), 1), round(float(b), 1), int(o), round(float(c), 2)] for a, b, o, c in zip(rr, xs, off, cf)]
        cands = self.T.get("candidates") or []
        if len(cands) > 1:   # other curves in the track (n_curves > 1), drawn for the candidate picker
            out["candidates"] = []
            for k, t in enumerate(cands):
                cr = t["rows"][i0:i1:st]; cx = self.panel_to_sheet(cr, t["x"][i0:i1:st])
                sty = t.get("style") or {}
                out["candidates"].append(dict(index=k, picked=k == self.T.get("picked"), score=round(float(t.get("score", 0)), 3),
                                              width=round(float(sty.get("width", 0)), 1), ink=round(float(sty.get("ink", 0)), 2),
                                              dashed=bool(sty.get("dash_period", 0) > 0 or 0 < sty.get("ink", 1) < 0.7),
                                              trace=[[round(float(a), 1), round(float(b), 1)] for a, b in zip(cr, cx)]))
        out["anchors"] = [[r, float(self.panel_to_sheet(r, x)), x] for r, x in self.anchors]
        out["excluded"] = [[r, float(self.panel_to_sheet(r, x)), x] for r, x in self.excluded]
        out["backup_intervals"] = [list(a) for a in self.backup_intervals]
        out["strokes"] = [[[r, float(self.panel_to_sheet(r, x))] for r, x in st] for st in self.strokes]
        return out

    def reference(self):
        """(depth, values) of the curve in the paired LAS, or None."""
        if not self.las_path or not Path(self.las_path).exists():
            return None
        from pylogdigitizer.las_export import read_las
        df = read_las(self.las_path).df()
        if self.curve not in df.columns:
            return None
        ref = df[self.curve].dropna(); ref = ref[ref > -999]
        return ref.index.values.astype(float), ref.values.astype(float)

    def qc(self, plot_path: str | None = None) -> dict | None:
        """r / RMSE / bias / MAE vs the paired LAS (log10 for log scales), after +-20 ft alignment."""
        from pylogdigitizer import qc_compare as qc
        ref = self.reference()
        if ref is None or self.T is None or self.dm is None:
            return None
        R = self.result(); rv, tv = ref[1], R["value"]
        if self.scale["type"] == "log":
            rv = np.log10(np.clip(rv, 1e-3, None)); tv = np.log10(np.clip(tv, 1e-3, None))
        c = qc.compare(ref[0], rv, R["depth"], tv)
        if plot_path and len(c["grid"]):
            qc.qc_plot(c["grid"], c["ref"], c["test_aligned"], plot_path, f"{self.well} {Path(self.scan_path).name} {self.curve}"
                       f" ({len(self.anchors)} anchors)")
        m = dict(c["aligned"]); m.update(shift_ft=c["shift_ft"], n_anchors=len(self.anchors))
        return dict(metrics=m, compare=c)

    def resampled(self, step: float = 0.5):
        """(depth grid, values) on a regular ``step`` grid - what export_las writes."""
        from pylogdigitizer.las_export import resample
        R = self.result()
        return resample(R["depth"], R["value"], step)

    def export_las(self, path, step: float = 0.5, srvc: str = "pyLogDigitizer assisted") -> Path:
        from pylogdigitizer.las_export import write_las
        name = self.curve.split(":")[0]
        g, v = self.resampled(step)
        return write_las(path, g, {name: v}, units={name: UNITS.get(name, "")},
                         well={"WELL": self.well, "SRVC": srvc})

    # ------------------------------------------------------------------ persistence
    def calibration(self) -> dict:
        return dict(depth_anchors=[list(a) for a in self.depth_anchors], depth_source=self.depth_source,
                    px_per_ft=self.dm.px_per_ft if self.dm is not None else None, scale=dict(self.scale))

    def session_dict(self) -> dict:
        return dict(version=SESSION_VERSION, scan=self.scan_path, well=self.well, curve=self.curve, las=self.las_path,
                    auto_layout=self.auto_layout, x_range=self.x_range, rows_window=self.rows_window,
                    section=self.section, calibration=self.calibration(), anchors=[list(a) for a in self.anchors],
                    excluded=[list(a) for a in self.excluded], backup_intervals=[list(a) for a in self.backup_intervals],
                    strokes=[[list(p) for p in st] for st in self.strokes], decisions=self.decisions,
                    trace_kw=self.trace_kw, edge_edits=[list(e) for e in getattr(self, "edge_edits", [])],
                    circles=[list(c) for c in getattr(self, "circles", [])])

    def save_session(self, path) -> Path:
        """Save everything needed to replay this digitization (no image data)."""
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.session_dict(), indent=1, default=float)); return p

    @classmethod
    def from_session_file(cls, path, root=None, layout=None, cost_mode: str | None = None,
                          ml_model_path: str | None = None, ml_weight: float | None = None) -> "AssistedSession":
        """Replay a saved session: same section, calibration, scale and anchors (global trace with
        all anchors). ``root`` resolves relative scan/LAS paths; ``layout`` reuses an analyzed Layout."""
        d = json.loads(Path(path).read_text()); root = Path(root) if root else Path.cwd()
        fix = lambda p: (str(p) if (p is None or Path(p).is_absolute()) else str(root / p))
        sc_ = d["calibration"]["scale"]
        s = cls(fix(d["scan"]), curve=d["curve"], well=d["well"], las_path=fix(d.get("las")),
                track=(d["section"] or {}).get("track"), scale=(sc_["left"], sc_["right"]), scale_type=sc_["type"],
                backup=sc_["backup"], wrap_side=sc_["wrap_side"], use_hatch=sc_["use_hatch"],
                auto_layout=d.get("auto_layout", True), x_range=d.get("x_range"), rows=d.get("rows_window"),
                trace_kw=d.get("trace_kw"),
                cost_mode=cost_mode or d.get("cost_mode", "conventional"),
                ml_model_path=ml_model_path if ml_model_path is not None else d.get("ml_model_path"),
                ml_weight=ml_weight if ml_weight is not None else d.get("ml_weight"))
        if layout is not None:
            s.L, s.ink = layout, layout.ink
        else:
            if s.auto_layout:
                try:
                    s.L = wf.analyze_layout(s.scan_path, x_range=s.x_range, rows=s.rows_window); s.ink = s.L.ink
                except LayoutError:
                    s.L = None
            if s.ink is None:
                s.ink = load_ink(s.scan_path)
        s.set_depth_anchors(d["calibration"]["depth_anchors"]); s.depth_source = d["calibration"].get("depth_source", "saved")
        sec = d["section"]
        if sec["kind"] == "box":
            s.set_section(box=(sec["x0"], sec["x1"]), rows=sec["rows"])
        else:
            for n, dl, dr in d.get("edge_edits") or []:      # hand-moved track boundaries
                li, ri = s.L.edge_names.index(n + ".left"), s.L.edge_names.index(n + ".right")
                E = np.array(s.L.edges, float); E[:, li] += dl; E[:, ri] += dr; s.L.edges = E
            s.edge_edits = [tuple(e) for e in d.get("edge_edits") or []]
            s.set_section(track=sec["track"], rows=sec["rows"])
        s.backup_intervals = [tuple(x) for x in d.get("backup_intervals", [])]
        s.auto_trace(anchors_panel=d.get("anchors") or None, excluded=d.get("excluded") or None, strokes=d.get("strokes") or None)
        for r, x, rad, thr in d.get("circles") or []:            # cursor-circle re-traces, in order
            s.circle_retrace(r, x, radius_px=rad, through=thr, panel_x=True)
        s.decisions = [dict(x) for x in d.get("decisions", [])]
        return s

    def save_training(self, root) -> Path:
        """Save image crop + curve mask + trace + calibration + metadata (ml_training_data format)."""
        from pylogdigitizer import ml_training_data as mtd
        R = self.result(); r0 = self.section["rows"][0]
        trace = dict(sheet_row=R["rows"], panel_row=R["rows"] - r0, x_panel=R["x_panel"], depth_ft=R["depth"],
                     value=R["value"], conf=R["conf"], off_scale=R["off_scale"])
        meta = dict(well=self.well, scan=Path(self.scan_path).name, scan_path=self.scan_path, curve=self.curve,
                    track=self.section.get("track", "box"), section=self.section, panel_rows=self.section["rows"],
                    width=self.width, source="assisted_tracing", backend=ct.backend(), n_anchors=len(self.anchors),
                    excluded=[list(a) for a in self.excluded], picks=[list(a) for a in self.anchors],
                    strokes=[[list(p) for p in st] for st in self.strokes], decisions=self.decisions,
                    n_strokes=len(self.strokes))
        qcr = self.qc()
        if qcr:
            meta["qc"] = {k: v for k, v in qcr["metrics"].items() if k != "reason"}
        return mtd.save_correction(root, self.pan, trace, self.anchors, self.calibration(), meta)
