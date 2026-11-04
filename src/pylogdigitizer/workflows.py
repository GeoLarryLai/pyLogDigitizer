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

__all__ = ["Layout", "analyze_layout", "calibrate_depth", "digitize_curve", "trace_to_values", "section_rows", "LayoutError", "CalibrationError", "SectionChoiceNeeded"]


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


def analyze_layout(path, x_range: tuple[int, int] | None = None, rows: tuple[int, int] | None = None,
                   block: int = 800, deskew_block: int = 2000) -> Layout:
    """Load the sheet (or the optional ``x_range``/``rows`` window), deskew, find the first stable
    track layout below the header, follow the boundaries and pick the depth labels."""
    t0 = time.time()
    info = read_info(path)
    x0, x1 = x_range or (0, info["width"])
    r0, r1 = rows or (0, info["height"])
    ink = load_ink(path, (x0, r0, x1, r1))
    ink, ang = scl.deskew_blocks(ink, deskew_block)
    probe, tracks = tgd.find_probe(ink, block)
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
    L = Layout(ink, x0, r0, ang, tracks, probe, centers, edges, names)
    t3 = time.time()
    dcol = L.panel("depth", (0, ink.shape[0]))
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
    T = ct.trace_track(pan, r0, px_per_ft=dm.px_per_ft, **trace_kw)
    v, wr = trace_to_values(pan, T, width, scale, scale_type, backup, wrap_side, use_hatch)
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
                residuals=res, checks=checks, trace=T, wraps=wr, panel_rows=(r0, r1), panel=pan)


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
