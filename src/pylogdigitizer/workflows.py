# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""workflows: end-to-end conventional digitization of one curve.

Orchestration only - the logic lives in the modules:
scan_io.load_ink -> scan_cleanup.deskew_blocks -> track_grid_detection.find_probe / follow_lines /
rectify -> depth_calibration.choose_labels / ocr_depth_labels / calibrate_labels /
validate_depth_map -> curve_tracing.trace_track -> scale_calibration -> scale_wraparound.

Every stage fails loudly (LayoutError / CalibrationError with a hint) instead of returning
garbage. Overrides (x_range, rows, width, first_depth, label_interval) are optional.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np

from pylogdigitizer.scan_io import load_ink, read_info
from pylogdigitizer import scan_cleanup as scl, track_grid_detection as tgd, depth_calibration as dc
from pylogdigitizer import scale_calibration as sc, curve_tracing as ct, scale_wraparound as sw
from pylogdigitizer import header_ocr as hdo
from pylogdigitizer.track_grid_detection import LayoutError
from pylogdigitizer.depth_calibration import CalibrationError

__all__ = ["Layout", "analyze_layout", "calibrate_depth", "digitize_curve", "trace_to_values", "section_rows", "LayoutError", "CalibrationError", "SectionChoiceNeeded", "prepare_color_scan", "open_color_curve", "color_curve_values", "apply_auto_backup", "trace_wraps"]


@dataclass
class Layout:
    ink: np.ndarray                 # deskewed ink of the working strip
    x_offset: int
    y_offset: int
    skew_deg: np.ndarray            # per-block skew angles
    tracks: list[dict]              # panels detected in the probe block
    probe_row: int
    centers: np.ndarray             # block centre rows for boundary following
    edges: np.ndarray               # followed boundary x per block [n_blocks, n_lines]
    edge_names: list[str]
    labels: list[dict] = field(default_factory=list)
    label_mode: str = "none"
    label_scores: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    _depth_panel: np.ndarray | None = field(default=None, repr=False)
    _ocr_values: list | None = field(default=None, repr=False)
    sections: list = field(default_factory=list, repr=False)   # [(labels, mode, scores)] per depth-label lattice
    layout_mode: str = "default"    # "default", "thin_borders" or "grey_thin_borders" (grey-grid retry)

    def with_section(self, i: int) -> "Layout":
        """The same layout using the i-th depth-label lattice (a sheet can hold e.g. a 2" and a 5" log)."""
        import dataclasses
        lab, mode, sc_ = self.sections[i]
        out = dataclasses.replace(self, labels=lab, label_mode=mode, label_scores=sc_, _ocr_values=None)
        out._ocr_engine = None
        return out

    def width_of(self, name: str) -> int:
        li, ri = self.edge_names.index(name + ".left"), self.edge_names.index(name + ".right")
        return int(round(np.median(self.edges[:, ri] - self.edges[:, li])))

    def panel(self, name: str, rows: tuple[int, int], width: int | None = None) -> np.ndarray:
        """Rectified panel between two followed boundaries, e.g. name='track1' or 'depth'."""
        li, ri = self.edge_names.index(name + ".left"), self.edge_names.index(name + ".right")
        return tgd.rectify(self.ink, rows, self.centers, self.edges[:, li], self.edges[:, ri],
                           width or self.width_of(name))

    @property
    def track_names(self) -> list[str]:
        return [n[:-5] for n in self.edge_names if n.endswith(".left") and n.startswith("track")]

    @property
    def label_rows(self) -> np.ndarray:
        return np.array([l["row"] for l in self.labels], float)


GREY_INK_THRESHOLD = 200   # grey-grid retry: luminance below this counts as ink


def analyze_layout(path, x_range: tuple[int, int] | None = None, rows: tuple[int, int] | None = None,
                   block: int = 800, deskew_block: int = 2000) -> Layout:
    """Load the sheet (or the optional ``x_range``/``rows`` window), deskew, find the first stable
    track layout below the header, follow the boundaries and pick the depth labels.

    If no layout is found, the sheet is retried with mid-grey counted as ink and thin track
    borders accepted (grey-grid digital plots). When the default pass only succeeded through a
    fallback detector, a thin-border layout (continuous borders) is preferred when it only adds
    track splits to it (same depth column, every edge kept).
    ``L.layout_mode`` records which pass worked."""
    t0 = time.time()
    info = read_info(path)
    x0, x1 = x_range or (0, info["width"])
    r0, r1 = rows or (0, info["height"])
    ink = load_ink(path, (x0, r0, x1, r1))
    ink, ang = scl.deskew_blocks(ink, deskew_block)
    mode = "default"
    try:
        probe, tracks = tgd.find_probe(ink, block)
    except LayoutError:
        # Digital plots (e.g. Eli / Gemini triple combos) draw the grid and the thin track
        # borders in mid-grey (128,128,128), which the default ink threshold drops. Retry
        # with grey counted as ink and thin (>= 3 px) borders accepted.
        ink_g = load_ink(path, (x0, r0, x1, r1), threshold=GREY_INK_THRESHOLD)
        ink_g, ang_g = scl.deskew_blocks(ink_g, deskew_block)
        probe, tracks = tgd.find_probe(ink_g, block, thin_borders=True)
        ink, ang, mode = ink_g, ang_g, "grey_thin_borders"
    else:
        # A fallback detector (grid spacing / horizontal grid) can merge neighbouring tracks
        # whose shared border is thin (e.g. the 1-bit working images of a colour digital
        # plot). Prefer a thin-border layout when it is valid, keeps the same depth column
        # and resolves more tracks.
        if tracks and tracks[0].get("detector") != "detect_tracks":
            try:
                probe_t, tracks_t = tgd.find_probe(ink, block, thin_borders=True)
            except LayoutError:
                tracks_t = []
            # accepted only when it ADDS splits to the layout that already worked (every edge of
            # the fallback layout is kept), so a scan read correctly before is never re-split
            if (tracks_t and tracks_t[0].get("detector") == "detect_tracks_thin"
                    and len(tracks_t) > len(tracks) and tgd.same_depth_column(tracks, tracks_t)
                    and tgd.only_adds_splits(tracks, tracks_t)):
                probe, tracks, mode = probe_t, tracks_t, "thin_borders"
    names, xs = [], []
    k = 0
    for t in tracks:
        nm = "depth" if t["kind"] == "depth" else f"track{(k := k + 1)}"
        names += [nm + ".left", nm + ".right"]; xs += [t["grid_x"][0], t["grid_x"][-1]]
    t2 = time.time()
    cen_dn, e_dn = tgd.follow_lines(ink, xs, block=block, rows=(probe, ink.shape[0]))
    cen_up, e_up = tgd.follow_lines(ink[::-1], xs, block=block, rows=(ink.shape[0] - probe, ink.shape[0]))
    cen_up = ink.shape[0] - np.asarray(cen_up, float)
    e_up = np.asarray(e_up, float).reshape(-1, len(xs)); e_dn = np.asarray(e_dn, float).reshape(-1, len(xs))
    centers = np.r_[cen_up[::-1], cen_dn]; edges = np.r_[e_up[::-1], e_dn]
    if len(centers) == 0:
        raise LayoutError("track boundaries could not be followed (sheet too short?)")
    if mode != "default" and len(edges):
        # digital plots are perfectly straight: 1 px jitter between followed blocks would only
        # make rectify() stretch the panels row by row, which garbles the depth labels
        med = np.median(edges, axis=0)
        flat = np.ptp(edges, axis=0) <= 2
        edges[:, flat] = med[flat]
    L = Layout(ink, x0, r0, ang, tracks, probe, centers, edges, names)
    L.layout_mode = mode
    t3 = time.time()
    dcol = L.panel("depth", (0, ink.shape[0]))
    if mode != "default":
        # thin-border digital plots: drop the border lines themselves from the depth column,
        # so the full-height rules do not hide the label rows
        dt = next(t for t in tracks if t["kind"] == "depth")
        tl = max(0, int(dt["left"] - dt["grid_x"][0]) + 1)
        tr_ = max(0, int(dt["grid_x"][-1] - dt["right"]) + 1)
        li, ri = L.edge_names.index("depth.left"), L.edge_names.index("depth.right")
        lft, rgt = edges[:, li] + tl, edges[:, ri] - tr_
        # one output column per source column (the -0.5 keeps rectify's pixel-centre sampling
        # on integer columns instead of rounding .5 values alternately up and down)
        dcol = tgd.rectify(ink, (0, ink.shape[0]), centers, lft - 0.5, rgt - 0.5,
                           int(round(float(np.median(rgt - lft)))))
    L._depth_panel = dcol
    L.labels, L.label_mode, L.label_scores = dc.choose_labels(dcol, row_min=probe)
    L.sections = [(L.labels, L.label_mode, L.label_scores)]
    while len(L.sections[-1][0]) >= 3 and len(L.sections) < 4:      # further log sections lower on the sheet
        lr = [l["row"] for l in L.sections[-1][0]]
        start = lr[-1] + 1.5 * float(np.median(np.diff(lr)))
        if start >= ink.shape[0] - 1000:
            break
        lab, mode, sc_ = dc.choose_labels(dcol, row_min=start)
        if len(lab) < 5:
            break
        L.sections.append((lab, mode, sc_))
    L.timings = dict(load_deskew_s=t2 - t0, follow_s=t3 - t2, labels_s=time.time() - t3)
    return L


def calibrate_depth(L: Layout, track: str = "track1", first_depth: float | None = None,
                    label_interval: float | None = None, use_ocr: bool = True, validate: bool = True):
    """Depth map for the label lattice of ``L``. OCR (``use_ocr``: True = auto engine - RapidOCR, else
    RapidOCR -, False, or 'rapidocr'; see ocr_engines)
    reads the labels; ``first_depth`` is the fallback / cross-check. Returns (DepthMap, anchors,
    residuals, checks). Raises CalibrationError with a hint."""
    rows = L.label_rows
    if len(rows) < 3:
        raise CalibrationError(f"{len(rows)} depth labels found (scores {L.label_scores}); need >= 3 on a regular "
                               "lattice - check the depth column in the detection figure or pass rows=")
    values = None
    engine = use_ocr if isinstance(use_ocr, str) else "auto"     # use_ocr: True/False or an engine name
    if use_ocr and dc.ocr_available(engine):
        from pylogdigitizer import ocr_engines
        engine = ocr_engines.resolve(engine)
        if L._ocr_values is None or getattr(L, "_ocr_engine", engine) != engine:
            dcol = L._depth_panel if L._depth_panel is not None else L.panel("depth", (0, L.ink.shape[0]))
            L._ocr_values = [v for _, v in dc.ocr_depth_labels(dcol, L.labels, engine=engine)]
            L._ocr_engine = engine
        values = L._ocr_values
    a0, a1 = int(max(0, rows[0] - 300)), int(min(L.ink.shape[0], rows[-1] + 300))
    panel = L.panel(track, (a0, a1))
    c, major = tgd.grid_rows(panel, 0.6)
    heavy = c[major] + a0
    dm, anchors, res, info = dc.calibrate_labels(rows, heavy, values, first_depth, label_interval)
    checks = dc.validate_depth_map(dm, anchors, res, L.ink.shape[0] - L.probe_row, n_labels=len(rows)) if validate else {}
    checks.update(info)
    return dm, anchors, res, checks


LINE_HINTS = ("thin", "thick", "dashed", "dotted", "solid", "left", "right")


def check_line_hint(h: str) -> str:
    """Validate a per-well line hint (appearance or position of this curve's line in its track)."""
    if h not in LINE_HINTS:
        raise ValueError(f"line_hint must be one of {LINE_HINTS}, got {h!r}")
    return h


def digitize_curve(path, track: str, scale: tuple[float, float], first_depth: float | None = None,
                   depth_range: tuple[float, float] | None = None, width: int | None = None,
                   backup: tuple[float, float] | None = None, wrap_side: str = "right",
                   use_hatch: bool = True, label_interval: float | None = None, layout: Layout | None = None,
                   x_range=None, rows=None, scale_type: str = "linear", use_ocr: bool = True,
                   section: int | None = None, grid_rectify: bool = False, **trace_kw) -> dict:
    """Conventional end-to-end digitization of one curve.

    ``scale`` = (value at left edge, value at right edge) of ``track`` (reversed allowed, e.g. DT
    (100, 40)); ``scale_type`` 'linear' or 'log' (e.g. ILD (0.2, 2000)). ``backup`` = backup-scale
    (left, right) values; ``wrap_side`` is the edge the curve leaves through ('right' = GR 0-150
    re-entering from the left on 150-300; 'left' = DT 100->40 re-entering from the right on
    160->100). ``depth_range`` defaults to the label lattice extent (+-1 label spacing), which
    excludes repeat sections. ``section``: which log section to use when the sheet holds more than
    one (e.g. a 2" and a 5" print); without it the one overlapping ``depth_range`` or clearly matching
    the header depths is used, otherwise :class:`SectionChoiceNeeded` lists the sections. Returns dict(depth, value, conf, layout, depth_map, anchors,
    residuals, checks, trace, wraps, panel_rows, panel)."""
    L = layout or analyze_layout(path, x_range=x_range, rows=rows)
    if track not in L.track_names:
        raise LayoutError(f"{track} not found; detected tracks: {L.track_names}")
    L, (dm, anc, res, checks) = _pick_section(L, track, first_depth, label_interval, use_ocr, depth_range, section)
    lo_row, hi_row = section_rows(L, anc, checks)
    if depth_range is None:
        depth_range = (float(dm(lo_row)), float(dm(hi_row)))
    r0 = int(max(lo_row, dm.inverse(depth_range[0]))); r1 = int(min(hi_row, dm.inverse(depth_range[1])))
    if r1 - r0 < 50:
        raise CalibrationError(f"depth_range {depth_range} is outside the calibrated section "
                               f"({dm(lo_row):.0f}-{dm(hi_row):.0f} ft)")
    width = width or L.width_of(track)
    pan = L.panel(track, (r0, r1), width)
    if grid_rectify:   # straighten bent grid lines inside the track before tracing (0.1.3)
        from pylogdigitizer import grid_rectify as gr
        pan, lat = gr.rectify_panel(pan)
        checks["grid_lines"] = 0 if lat is None else int(lat.X.shape[1])
    nc = int(trace_kw.pop("n_curves", 1)); pick = trace_kw.pop("pick", "auto")
    vec = None
    vmode = trace_kw.pop("vector_trace", False)
    # "model": re-trace on the ONNX segmenter's curve ink (bundled default model unless seg_model=...)
    amode = trace_kw.pop("appearance_trace", False)
    if amode not in (False, None, "model"):
        raise ValueError("appearance_trace must be 'model' (or off)")
    seg_model = trace_kw.pop("seg_model", None)
    seed_curves = trace_kw.pop("seed_curves", False)   # independent seeds for multi-curve tracks
    # per-well/track line hint for scans without a readable legend: which seeded line is this curve
    # ("thin"/"thick"/"dashed"/"solid"/"left"/"right"). Appearance only, never LAS values.
    line_hint = trace_kw.pop("line_hint", None)
    if line_hint:
        check_line_hint(line_hint); seed_curves = True
    spikes = trace_kw.pop("spikes", None)              # model mode with a backup scale: read fast off-scale spikes
    clip_null = trace_kw.pop("clip_null", None)        # null where the curve is clipped at the track edge (default: on with "model")
    if vmode:   # vectorize first, drop grid vectors, trace curve vectors only
        from pylogdigitizer import vectorize as vz
        if vmode == "v1":
            vec = vz.vector_separate(pan, n_curves=nc)
        else:     # linked chains; text is decided on vectors, so the raster text mask is off
            vec = vz.vector_separate_v2(pan, n_curves=nc)
            if vmode != "v2_mask":
                trace_kw["mask_other"] = False
            checks["vector_rescued"] = vec["n_rescued"]; checks["vector_text_fixed"] = vec["n_text_fixed"]
        checks["vector_segments"] = int(sum(s is not None for s in vec["V"]["segs"]))
        checks["vector_grid_segments"] = int((vec["cls"] == vz.GRID).sum())
    tpan = pan if vec is None else vec["curve_ink"]
    if nc <= 1:
        trace_kw.pop("pick_margin", None)   # multi-curve only
    if nc > 1:      # several curves in the track (e.g. GR + caliper): trace each, keep ``pick``
        T = ct.trace_multi(tpan, r0, n_curves=nc, pick=pick, px_per_ft=dm.px_per_ft, **trace_kw)
        checks["curves_traced"] = len(T["candidates"]); checks["curve_picked"] = T["picked"]
        checks["curve_scores"] = [round(float(t["score"]), 3) for t in T["candidates"]]
    else:
        T = ct.trace_track(tpan, r0, px_per_ft=dm.px_per_ft, **trace_kw)
    if vec is not None and vmode not in ("v1", "v2_nochain"):
        T = vz.apply_chains(T, vec, pan.shape, n_curves=nc)
        checks["vector_fallback_rows"] = int(T.get("vector_fallback", 0))
    if amode == "model":   # ONNX segmenter: curve ink only; candidates re-traced on it
        import time as _t
        from pylogdigitizer import appearance_trace as at
        from pylogdigitizer.scan_segmenter import curve_probability, default_model
        t0 = _t.time()
        cp = curve_probability(pan, seg_model or default_model())
        cink = np.asarray(pan, bool) & (cp > 0.5)
        cands = T.get("candidates") or [T]
        seeded = at.seeded_multi_trace(cink, int(T["fy"]), len(cands)) if (len(cands) > 1 and seed_curves) else []
        if len(seeded) == len(cands):     # independent seeds; roles from the pick rule or the line hint
            order = at.match_to(cands, seeded)
            if line_hint:
                k = at.role_by_style(seeded, line_hint); pi = cands.index(T) if T in cands else 0
                rest = [sd for j, sd in enumerate(seeded) if j != k]
                order = rest[:pi] + [seeded[k]] + rest[pi:]
                checks["seed_styles"] = [dict(width=round(float(np.median(sd["width"])), 1), gap=round(sd["gap_frac"], 3)) for sd in seeded]
            for t, sd in zip(cands, order):
                nx = sd["x"][:len(t["x"])]
                t["off_left"] = nx <= 1; t["off_right"] = nx >= width - 2
                at._set_x(t, nx, np.ones(len(nx), bool))
            checks["seeded"] = [round(float(sd["seed_x"]), 1) for sd in seeded]
            checks["reseeds"] = [sd["reseeds"] for sd in seeded]
        else:
            T = at.apply_identity(T, cink)
        checks["seg_curve_frac"] = round(float((cp > 0.5).mean()), 4)
        checks["appearance_s"] = round(_t.time() - t0, 1)
    v, wr = trace_to_values(pan, T, width, scale, scale_type, backup, wrap_side, use_hatch)
    iw = []
    if amode == "model" and backup is not None:   # explicit wrap transition onto the backup curve
        mk = sc.log_scale if scale_type == "log" else sc.linear_scale
        wb = at.wrap_backup(cink, np.asarray(T["x"], float), int(T["fy"]), width, wrap_side)
        for a, b, xb in wb:
            v[a:b] = mk(0, width, *backup)(xb)
        checks["model_wraps"] = [(int(a), int(b)) for a, b, _ in wb]
        # overflow wraps the tracer bridged with a connector: re-read on the inked samples only
        fx = np.asarray(T["x"], float); nfx = len(fx)
        Bk = at.band_ink(cink, int(T["fy"]), nfx, dilate=1)
        xi = np.clip(np.nan_to_num(fx).astype(int), 0, Bk.shape[1] - 1)
        Cs = np.c_[np.zeros(len(Bk), np.int32), np.cumsum(Bk, axis=1, dtype=np.int32)]
        kk = np.arange(min(nfx, len(Bk)))
        on = np.zeros(nfx, bool)
        on[kk] = (Cs[kk, np.minimum(xi[kk] + 4, Bk.shape[1])] - Cs[kk, np.maximum(xi[kk] - 3, 0)]) > 0
        iw, cm = sw.ink_jump_wraps(fx, on, width, wrap_side)
        inwb = np.zeros(nfx, bool)
        for a, b, _ in wb:
            inwb[a:b] = True
        bsv = mk(0, width, *backup)(fx)
        for a, b in iw:
            seg = np.arange(a, b); seg = seg[~inwb[seg]]
            v[seg] = bsv[seg]
        v[cm[:len(v)]] = np.nan
        checks["ink_wraps"] = [(int(a), int(b)) for a, b in iw]; checks["connector_null"] = int(cm.sum())
        if spikes if spikes is not None else True:     # fast spikes off the scale: backup peak, else null
            sp = at.spike_excursions(cink, np.asarray(T["x"], float), int(T["fy"]), width, wrap_side)
            to_v = mk(0, width, *backup)
            for i, f in sp.items():
                if i < len(v):
                    v[i] = to_v(np.array([f * (width - 1)]))[0] if np.isfinite(f) else np.nan
            checks["spikes"] = dict(backup=int(sum(np.isfinite(f) for f in sp.values())), clipped=int(sum(not np.isfinite(f) for f in sp.values())))
    if (clip_null if clip_null is not None else amode == "model"):
        v = clip_to_null(np.asarray(v, float), np.asarray(T["x"], float), width, wr,
                         [(a, b) for a, b, _ in (wb if amode == "model" and backup is not None else [])] + list(iw))
        checks["clipped_null"] = int(np.isnan(v).sum())
    checks.update(width=width, rows=(r0, r1), depth_top=round(float(dm(r0)), 1), depth_base=round(float(dm(r1)), 1),
                  label_mode=L.label_mode, n_wraps=len(wr), probe_row=L.probe_row)
    if use_ocr:
        try:
            hd = header_depth_info(L)
            checks["header_depths"] = {k: hd[k] for k in ("top", "bottom")}
            wmsg = hdo.depth_range_warning(checks["depth_top"], checks["depth_base"], hd)
            if wmsg:
                checks["depth_mismatch"] = wmsg
        except Exception as e:                       # header check is advisory only
            checks["header_depths_error"] = str(e)
    return dict(depth=dm(T["rows"]), value=v, conf=T["conf"], layout=L, depth_map=dm, anchors=anc,
                residuals=res, checks=checks, trace=T, wraps=wr, panel_rows=(r0, r1), panel=pan, vector=vec,
                grid_lattice=lat if grid_rectify else None)


class SectionChoiceNeeded(CalibrationError):
    """The sheet holds more than one log section (e.g. a 2" and a 5" print) and nothing says which
    one is wanted. ``sections``: list of dict(index, top, base, row0, row1, px_per_ft)."""
    def __init__(self, sections: list[dict]):
        self.sections = sections
        lst = "; ".join(f"section={c['index']}: {c['top']:.0f}-{c['base']:.0f} ft" for c in sections)
        super().__init__(f"This sheet has {len(sections)} log sections ({lst}). Choose one: pass section=<number> "
                         "or depth_range=(top, base). In the app, click the section on the scan.")


def _section_list(cands) -> list[dict]:
    out = []
    for i, Li, cal in cands:
        lo, hi = section_rows(Li, cal[1], cal[3])
        out.append(dict(index=i, top=round(float(cal[0](lo)), 1), base=round(float(cal[0](hi)), 1),
                        row0=float(lo), row1=float(hi), px_per_ft=round(float(cal[0].px_per_ft), 2)))
    return out


def _header_match(sections: list[dict], hd: dict) -> list[int]:
    """Sections whose top and base both agree with the header's logged interval (within 5 % of the
    section length, at least 50 ft)."""
    top, bot = hd.get("top"), hd.get("bottom")
    if top is None or bot is None:
        return []
    out = []
    for c in sections:
        tol = max(50.0, 0.05 * (c["base"] - c["top"]))
        if abs(c["top"] - top) <= tol and abs(c["base"] - bot) <= tol:
            out.append(c["index"])
    return out


def _pick_section(L: Layout, track, first_depth, label_interval, use_ocr, depth_range=None, section=None):
    """(layout, calibrate_depth result) for one log section. One section on the sheet: use it.
    Several: use ``section`` if given, else the one overlapping ``depth_range``, else the one the
    header's top/bottom depths clearly match; otherwise raise :class:`SectionChoiceNeeded` - no guessing."""
    if section == "auto":
        section = None
    if section is not None:
        i = int(section)
        L2 = L if L.labels is L.sections[i][0] else L.with_section(i)
        cal = calibrate_depth(L2, track, first_depth, label_interval, use_ocr)
        cal[3]["section_used"] = int(section)
        return L2, cal
    if len(L.sections) <= 1:
        return L, calibrate_depth(L, track, first_depth, label_interval, use_ocr)
    cands, errors = [], []
    for i in range(len(L.sections)):
        Li = L if i == 0 else L.with_section(i)
        try:
            cands.append((i, Li, calibrate_depth(Li, track, first_depth if i == 0 else None, label_interval, use_ocr)))
        except CalibrationError as e:
            errors.append(e)
    if not cands:
        raise errors[0]
    if len(cands) == 1:
        return cands[0][1], cands[0][2]
    secs = _section_list(cands)
    pick, why = [], ""
    if depth_range is not None:
        pick = [c["index"] for c in secs if min(c["base"], depth_range[1]) - max(c["top"], depth_range[0]) > 0]
        why = "depth_range"
    elif use_ocr:
        try:
            pick = _header_match(secs, header_depth_info(L)); why = "header depths"
        except Exception:
            pick = []
    if len(pick) != 1:
        raise SectionChoiceNeeded(secs)
    i, Li, cal = next(c for c in cands if c[0] == pick[0])
    cal[3].update(sections=secs, section_used=i, section_reason=why)
    return Li, cal


def header_depth_info(L: Layout) -> dict:
    """Logged depth interval read from the sheet header (cached on the layout)."""
    hd = getattr(L, "_header_depths", None)
    if hd is None:
        hd = hdo.header_depths(L.ink, L.probe_row)
        L._header_depths = hd
    return hd


def section_rows(L: Layout, anchors, checks: dict) -> tuple[float, float]:
    """Default log-section rows: the depth-label lattice extent +- one label spacing (excludes
    repeat sections; junk labels dropped by OCR move the top to the first kept label)."""
    lr = L.label_rows
    sp = float(np.median(np.diff(lr)))
    if checks.get("n_ocr_dropped") and anchors:
        lo_a, hi_a = min(a[0] for a in anchors), max(a[0] for a in anchors)
        lr = [r for r in lr if lo_a - sp / 2 <= r <= hi_a] or lr
    return max(L.probe_row, lr[0] - sp), min(L.ink.shape[0], lr[-1] + sp)


def clip_to_null(v: np.ndarray, x: np.ndarray, width: float, wraps=(), spans=(), edge: float | None = None,
                 min_len: int = 3) -> np.ndarray:
    """Clipping rule: where the trace runs flat along a track edge (the curve is clipped by the fixed
    scale and no backup curve carries it), the value is unknown, so write NaN for those samples.
    Samples inside ``wraps`` (sample indices or (a, b) spans) and backup ``spans`` keep their values;
    edge runs shorter than ``min_len`` samples (a curve touching the edge in passing) are kept."""
    v = np.array(v, float); x = np.asarray(x, float)
    e = float(edge if edge is not None else max(3.0, 0.01 * width))
    at_edge = (x <= e) | (x >= width - 1 - e)
    keep = np.zeros(len(v), bool)
    for w in list(wraps) + list(spans):
        if isinstance(w, (tuple, list)) and len(w) == 2:
            keep[int(w[0]):int(w[1])] = True
        else:
            keep[int(w)] = True
    m = at_edge & ~keep
    p = np.r_[0, m.astype(np.int8), 0]; d = np.diff(p)
    for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        if b - a >= min_len:
            v[a:b] = np.nan
    return v


def trace_to_values(pan: np.ndarray, T: dict, width: int, scale, scale_type: str = "linear", backup=None,
                    wrap_side: str = "right", use_hatch: bool = True):
    """Panel trace -> curve values on the main scale, unwrapped onto the ``backup`` scale where the
    curve leaves the track. Returns (values, wraps). Used by :func:`digitize_curve` and the
    assisted editor after each re-trace."""
    mk = sc.log_scale if scale_type == "log" else sc.linear_scale
    v = mk(0, width, *scale)(T["x"])
    wr = []
    if backup is not None:
        if not use_hatch and T.get("pad", 0) > 0 and "path_padded" in T:
            ncol = T["cost"].shape[1]; pad = T["pad"]
            off = (T["path_padded"] >= ncol - pad) if wrap_side == "right" else (T["path_padded"] < pad)
        else:
            off = T["off_right"] if wrap_side == "right" else T["off_left"]
        from_left = wrap_side == "right"     # backup curve re-enters from the opposite edge
        if use_hatch:
            hr_key = ("_hatch_rows", T["fy"], bool(from_left))
            if T.get("_hatch_rows_key") != hr_key or "_hatch_rows" not in T:
                T["_hatch_rows"] = sw.hatch_rows(pan, T["fy"], from_left=from_left)
                T["_hatch_rows_key"] = hr_key
            off = off | T["_hatch_rows"]
        wr = sw.detect_wraps(off, 3)
        bk = sw.backup_values_from_panel(pan, np.arange(len(v)), T["fy"], wr, mk(0, width, *backup),
                                         from_left=from_left, use_hatch=use_hatch,
                                         edge_pad=3.0 if use_hatch else 0.0)
        v = sw.unwrap(v, wr, bk)
        # stretches where the tracer itself followed the backup trace (edge-to-edge jumps): value it
        # on the backup scale directly
        oo = oi = None
        if T.get("pad", 0) > 0 and "path_padded" in T:
            ncol = T["cost"].shape[1]; pad = T["pad"]
            oo, oi = T["path_padded"] >= ncol - pad, T["path_padded"] < pad
        fw = sw.edge_jump_wraps(T["x"], width, wrap_side, off_out=oo, off_in=oi)
        if fw:
            bs = mk(0, width, *backup)(T["x"])
            for a, b in fw:
                v[a:b] = bs[a:b]
            wr = sorted(set(wr) | set(fw))
    return v, wr


def propose_header(path, layout: Layout | None = None, **kw) -> dict:
    """OCR the track headers and return curve/unit/scale PROPOSALS (header_ocr.propose_header).
    Nothing is applied: review the proposals, then pass a confirmed one through
    :func:`digitize_kwargs_from_proposal` into :func:`digitize_curve`."""
    from pylogdigitizer import header_ocr
    return header_ocr.propose_header(path, layout=layout, **kw)


def digitize_kwargs_from_proposal(proposal: dict, confirmed: bool = False, **overrides) -> dict:
    """Turn ONE header_ocr curve proposal into ``digitize_curve`` keyword arguments
    (track, scale, scale_type, backup, wrap_side). Proposals are never applied implicitly: this
    raises unless ``confirmed=True`` (the caller asserts a human checked the values; the error
    message lists the fields flagged ``needs_review``). ``overrides`` (e.g. scale=(0, 150)) replace
    proposed values."""
    f = {k: proposal[k] for k in ("scale_left", "scale_right", "scale_type", "backup")}
    out = dict(track=proposal["track"], scale_type=f["scale_type"]["value"])
    if f["scale_left"]["value"] is not None:
        out["scale"] = (f["scale_left"]["value"], f["scale_right"]["value"])
    if f["backup"]["value"] is not None:
        out["backup"] = tuple(f["backup"]["value"])
    out.update(overrides)
    review = [k for k in ("mnemonic", "scale_left", "scale_right", "scale_type", "backup")
              if proposal[k]["needs_review"] and not ({"scale_left": "scale", "scale_right": "scale"}.get(k, k)
                                                      in overrides)]
    if "scale" not in out:
        raise ValueError("proposal has no scale; pass scale=(left, right)")
    if not confirmed:
        raise ValueError(f"header proposal not confirmed (fields flagged for review: {review or 'none'}); "
                         "check the values and pass confirmed=True")
    if out.get("backup") is not None and "wrap_side" not in out:
        l, r = out["scale"]; bl, br = out["backup"]
        out["wrap_side"] = "right" if abs(bl - r) <= abs(br - l) else "left"
    return out


# --------------------------------------------------------------------------- colour-scan helpers
def prepare_color_scan(path, palette: dict | None = None, **suggest_kw) -> dict:
    """Load a colour scan, suggest or apply a palette, classify, and build the black-ink base.

    Returns dict(rgb, palette, labels, names, body_rows, base). ``palette`` if given is used
    as-is (manual palette via the API); otherwise :func:`scan_color.suggest_palette` proposes one.
    """
    from pylogdigitizer.scan_io import load_rgb
    from pylogdigitizer.scan_color import suggest_palette, classify_colors, find_log_body
    from pylogdigitizer.curve_ink import make_base_ink
    rgb = load_rgb(path)
    pal = dict(palette) if palette else suggest_palette(rgb, **suggest_kw)
    labels, names = classify_colors(rgb, pal)
    body = find_log_body(labels, names)
    base = make_base_ink(labels, names, body)
    return dict(rgb=rgb, palette=pal, labels=labels, names=names, body_rows=body, base=base)


def trace_wraps(s, jump: float = 0.35, edge: float = 0.08, near_ft: float = 1.5) -> tuple:
    """Wrap-arounds the trace of session ``s`` followed: it left the main curve towards the wrap
    edge, jumped across to the backup curve near the opposite edge, followed it, then jumped back.

    Returns ``(wraps [(a, b)] in trace samples, connector mask)``.
    """
    T = s.T
    if T is None or "cost" not in T or "path_padded" not in T:
        return [], np.zeros(0, bool)
    n = len(T["x"])
    cost, pp = T["cost"], np.asarray(T["path_padded"])
    on = (cost[np.arange(n), pp] < 0.5) & ~(T["off_left"] | T["off_right"])
    xs = np.asarray(T["x"], float) / float(s.width)
    right_side = (s.scale.get("wrap_side") or "right") == "right"
    if not right_side:
        xs = 1.0 - xs
    ncol, pad = cost.shape[1], int(T["pad"])
    inner = cost[:, pad:ncol - pad] < 0.5
    w = inner.shape[1]
    e0, e1 = max(1, int(round(0.015 * w))), max(2, int(round(edge * w)))
    ink_l, ink_r = inner[:, e0:e1].any(axis=1), inner[:, w - e1:w - e0].any(axis=1)
    if not right_side:
        ink_l, ink_r = ink_r, ink_l
    k = max(2, int(round(near_ft * (s.dm.px_per_ft if s.dm is not None else 20) / T["fy"])))

    def near(m, p, q):
        return bool(m[max(0, p - k):min(n, q + k + 1)].any())

    idx = np.flatnonzero(on)
    conn = np.zeros(n, bool)

    def across_paper(p, q):
        if q - p > 1:
            return True
        lo, hi = sorted((int(pp[p]), int(pp[q])))
        return float((np.minimum(cost[p, lo:hi + 1], cost[q, lo:hi + 1]) >= 0.5).mean()) > 0.5

    wraps, a = [], None
    for p, q in zip(idx[:-1], idx[1:]):
        d = xs[q] - xs[p]
        if abs(d) < jump or not (near(ink_l, p, q) and near(ink_r, p, q)) or not across_paper(p, q):
            continue
        if a is None and d < 0 and xs[p] > 0.5 > xs[q]:
            a = int(q); conn[p + 1:q] = True
        elif a is not None and d > 0 and xs[p] < 0.5 < xs[q]:
            wraps.append((a, int(q))); conn[p + 1:q] = True; a = None
    return [(x, y) for x, y in wraps if y > x], conn


def apply_auto_backup(s, cv: dict, auto_backup: bool = True):
    """If the trace wraps and no backup scale is set, assume the next full track width.

    Sets ``s.errors['wrap_note']``. Does nothing when ``auto_backup`` is False or a backup is
    already set (other than recording the depths where the wrap was seen).
    """
    from pylogdigitizer.scale_wraparound import default_backup
    if s.T is None or s.dm is None:
        return s
    wr, _ = trace_wraps(s)
    if not wr:
        return s
    ft = ", ".join(f"{s.dm(s.T['rows'][a]):.0f}-{s.dm(s.T['rows'][b - 1]):.0f}" for a, b in wr[:8])
    if s.scale.get("backup"):
        s.errors["wrap_note"] = f"wraps onto the backup scale {s.scale['backup']} at {ft} ft"
    elif auto_backup:
        bk = default_backup(cv)
        s.set_scale(s.scale["left"], s.scale["right"], s.scale["type"], backup=bk,
                    wrap_side=s.scale.get("wrap_side") or "right", use_hatch=False)
        s.errors["wrap_note"] = (f"wraps around at {ft} ft - backup scale {bk[0]:g}-{bk[1]:g} assumed "
                                 "(give 'backup' if the print uses another)")
    return s


def open_color_curve(tif, cv: dict, well: str, trace_kw: dict, edge_px: int = 6, depth_range=None,
                     first_depth=None, label_interval=None, x_range=None, auto_backup: bool = True,
                     las_path=None):
    """AssistedSession for one colour-isolated working image: layout + depth + auto trace.

    ``x_range`` is the track's (left, right) x on the original sheet. If the session finds a
    different width on this working image (a thick curve can look like a track border), the
    given position is used as a box section and a note goes to ``s.errors``.
    ``auto_backup``: when the trace wraps and ``cv`` has no backup, assume the next full
    track width (:func:`scale_wraparound.default_backup`).
    ``las_path`` is the paired reference LAS used by :meth:`AssistedSession.qc`.
    ``cv['use_hatch']`` (default False) reads hatched off-scale fills as on the black-ink path.
    Track-edge columns are excluded (``edge_px``) so the tracer cannot hug a thick
    printed border — without that the path collapses to a constant edge value.
    """
    from pylogdigitizer.assisted_tracing import AssistedSession
    s = AssistedSession(str(tif), curve=cv["name"], well=well, track=cv.get("track", "track1"),
                        scale=(float(cv["left"]), float(cv["right"])), scale_type=cv.get("scale_type", "linear"),
                        backup=cv.get("backup"), wrap_side=cv.get("wrap_side", "right"),
                        use_hatch=bool(cv.get("use_hatch", False)),
                        first_depth=first_depth, label_interval=label_interval, trace_kw=dict(trace_kw),
                        las_path=las_path)
    s.open()
    if s.pan is None and s.dm is None:
        return s
    box = None
    if x_range is not None:
        want = float(x_range[1]) - float(x_range[0])
        if s.pan is None or abs(s.width - want) > max(6.0, 0.03 * want):
            box = (float(x_range[0]), float(x_range[1]))
            s.errors["layout_note"] = (f"{cv.get('track', 'track')} found {s.width} px wide on this curve's image, "
                                       f"header says {want:.0f} px - used that position x = {box[0]:.0f}-{box[1]:.0f}")
    if box is not None or depth_range:
        rows = None if depth_range or s.section is None else tuple(s.section["rows"])
        try:
            s.set_section(track=cv.get("track", "track1"), box=box,
                          depth_range=tuple(depth_range) if depth_range else None, rows=rows)
            s.errors.pop("section", None)
        except Exception as e:
            s.errors["section"] = str(e)
            return s
    if s.pan is None:
        return s
    if edge_px:
        s.trace_kw["exclude_cols"] = [(0, edge_px), (s.width - edge_px, s.width + 2)]
    s.auto_trace()
    return apply_auto_backup(s, cv, auto_backup=auto_backup)


def _clip_edge(s) -> float:
    """Distance from a track edge (px) that counts as 'at the edge' for the clipping rule: the
    columns kept free of the trace at each edge (``exclude_cols``) plus 2 px, at least 1 % of
    the width (the trace cannot get closer than the excluded columns)."""
    e = max(3.0, 0.01 * s.width)
    for a, b in (s.trace_kw or {}).get("exclude_cols") or []:
        if a <= 0:
            e = max(e, float(b) + 2.0)
        elif b >= s.width:
            e = max(e, float(s.width - a) + 2.0)
    return e


def color_curve_values(s, null_offscale: bool = True, fill_backup=None):
    """(depth, value) of a traced colour-scan session.

    ``null_offscale``: where the trace runs flat along a track edge, write null instead of
    the edge value. With a backup scale, stretches where the trace followed the wrapped
    curve (:func:`trace_wraps`) and the session's ``backup_intervals`` are read on the
    backup scale. ``fill_backup`` (from :func:`scale_wraparound.fill_backups`): where a
    wrap is shaded, the backup value is read at the end of the shading (most reliable).
    Using a fill attribution always sets ``s.errors['backup_review']``.
    With ``use_hatch`` set on the session, its own values (hatch-aware backup reading) are used.
    """
    from pylogdigitizer import scale_calibration as sc
    R = s.result()
    x = np.asarray(R["x_panel"], float)
    if not s.scale.get("backup") or s.scale.get("use_hatch"):
        # no backup, or hatched off-scale fills: the session values already carry the backup reading
        v = np.asarray(R["value"], float)
        if null_offscale:
            v = clip_to_null(v, x, s.width, s.wraps, edge=_clip_edge(s))
        return np.asarray(R["depth"], float), v
    mk = sc.log_scale if s.scale["type"] == "log" else sc.linear_scale
    v = mk(0, s.width, s.scale["left"], s.scale["right"])(x)
    bk = mk(0, s.width, *s.scale["backup"])(x)
    wr, conn = trace_wraps(s)
    rows = np.asarray(R["rows"], float)
    for r0, r1 in getattr(s, "backup_intervals", []) or []:
        idx = np.flatnonzero((rows >= r0) & (rows <= r1))
        if len(idx):
            wr = wr + [(int(idx[0]), int(idx[-1]) + 1)]
    for a, b in wr:
        v[a:b] = bk[a:b]
    if null_offscale:
        v = clip_to_null(v, x, s.width, wr, edge=_clip_edge(s))
    if len(conn):
        v[conn[:len(v)]] = np.nan
    if fill_backup is not None:
        fr, fxs = (np.asarray(a_, float) for a_ in fill_backup[:2])
        k = np.clip(np.round(rows - fr[0]).astype(int), 0, len(fr) - 1)
        xf = np.where((rows >= fr[0]) & (rows <= fr[-1]), fxs[k], np.nan)
        m = np.isfinite(xf)
        if m.any():
            v[m] = mk(0, s.width, *s.scale["backup"])(np.asarray(s.sheet_to_panel(rows[m], xf[m] + 1.0), float))
            kk = max(2, int(round(1.5 * (s.dm.px_per_ft if s.dm is not None else 20) / s.T["fy"])))
            near = np.convolve(m.astype(float), np.ones(2 * kk + 1), "same") > 0
            edge_side = x / s.width < 0.25 if (s.scale.get("wrap_side") or "right") == "right" else x / s.width > 0.75
            tail = near & ~m & edge_side
            v[tail] = bk[tail]
        s.errors["backup_review"] = ("shaded fill used to read the backup scale — review this "
                                     "attribution before trusting the off-scale values")
    return np.asarray(R["depth"], float), v

