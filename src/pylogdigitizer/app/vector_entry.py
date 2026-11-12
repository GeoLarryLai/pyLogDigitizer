# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Vector PDF pages in the app: one session per curve read from its exact path (no tracing).

The page is drawn as a picture (:func:`pylogdigitizer.pdf_vector.render.render_page`) so the
viewer, the Check and save plot, the comparison with a reference LAS and every export work
exactly as for a scan. Each curve is a :class:`VectorCurveSession`: it answers the same calls as a
traced curve's session (values, depth, overlay, QC), with the path points as its "trace".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pylogdigitizer.assisted_tracing import AssistedSession, SessionError

TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
VECTOR_NOTE = "vector file: tracing not needed"


class _PageDepth:
    """Sheet row (rendered page pixels) -> depth, from the page's depth-number fit."""

    def __init__(self, fit, height_pt: float, scale: float):
        self.a, self.b = fit; self.h = height_pt; self.s = scale
        self.px_per_ft = abs(1.0 / (self.b * scale))

    def __call__(self, rows):
        y = self.h - np.asarray(rows, float) / self.s
        return self.a + self.b * y


def _hex(c) -> str:
    r, g, b = (int(round(255 * float(np.clip(v, 0, 1)))) for v in c[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


class VectorCurveSession(AssistedSession):
    """One vector curve, shaped like a traced curve's session (read-only: nothing to trace)."""

    def __init__(self, pdf_path, page: int, rgb: np.ndarray, scale: float, height_pt: float, trace, depth_fit,
                 depth_labels, well: str, las_path=None):
        info = trace.info
        super().__init__(pdf_path, curve=trace.mnemonic, well=well, las_path=las_path, track=None,
                         scale=(info["left"], info["right"]), scale_type=trace.kind, auto_layout=False)
        self.page = page; self.vector = True
        self.ink = rgb
        x, y = np.asarray(info["x"], float), np.asarray(info["y"], float)
        ok = np.isfinite(x) & np.isfinite(y)
        rows = (height_pt - y) * scale; cols = x * scale
        o = np.argsort(np.where(ok, rows, np.inf), kind="stable")
        o = o[ok[o]]
        self._x_pt = x[o]; self._x0, self._x1 = float(info["x0"]), float(info["x1"])
        self.dm = _PageDepth(depth_fit, height_pt, scale)
        self.depth_anchors = [(float((height_pt - yy) * scale), float(d)) for yy, d in depth_labels]
        self.depth_source = "PDF depth numbers"
        n = len(o)
        self.T = dict(rows=rows[o], x=cols[o], conf=np.ones(n), off_left=np.zeros(n, bool), off_right=np.zeros(n, bool),
                      labels=np.zeros(n, int), params={}, cost=np.zeros(n))
        self.width = int(round((self._x1 - self._x0) * scale))
        r0, r1 = (float(self.T["rows"][0]), float(self.T["rows"][-1])) if n else (0.0, 0.0)
        self.section = dict(track=None, x=[self._x0 * scale, self._x1 * scale], rows=[r0, r1])
        self.disagreement = np.zeros(n, int)
        self._update_values()

    # values straight from the path's x on the printed scale (no tracing, no wrap guessing)
    def _update_values(self):
        s = self.scale
        f = (self._x_pt - self._x0) / max(self._x1 - self._x0, 1e-9)
        if s["type"] == "log":
            a, b = np.log10(s["left"]), np.log10(s["right"])
            self.values = 10 ** (a + f * (b - a))
        else:
            self.values = s["left"] + f * (s["right"] - s["left"])
        self.wraps = []

    def panel_to_sheet(self, row, xp):
        return np.asarray(xp, float)

    def sheet_to_panel(self, row, x):
        return np.asarray(x, float)

    def check_spots(self, max_gap: float = 0.25, max_spots: int = 30) -> list[dict]:
        return []

    def _refuse(self, *a, **k):
        raise SessionError(VECTOR_NOTE + ": the curve is read from the file's own path")

    auto_trace = retrace_all = pin = add_anchor = circle_retrace = node_snap = add_stroke = answer_spot = _refuse
    set_section = rectify_grid = exclude_stroke = set_depth_anchors = set_depth_labels = _refuse

    def save_training(self, root):
        raise SessionError("training samples are made from traced scans; this curve was read from a vector PDF")

    def session_dict(self) -> dict:
        return dict(kind="vector", scan=self.scan_path, page=self.page, well=self.well, curve=self.curve,
                    las=self.las_path, calibration=self.calibration(), section=self.section)


def build_vector_entry(pdf: Path, las, body: dict, page: int | None, scale: float = 3.0):
    """Read the page with :func:`digitize_vector`, draw it, and make one session per curve.
    Returns ``(base_session, curves, rgb, log)``; ``curves`` are the app's curve dicts."""
    from pylogdigitizer.pdf_vector.vector_log import digitize_vector
    from pylogdigitizer.pdf_vector.render import render_page
    log = digitize_vector(pdf, page=page, step=0.5, fallback=False)
    vp = log.vector_page
    rgb = render_page(pdf, vp.page_index, scale, rotation=vp.rotation)
    well = body.get("well") or log.well.get("WELL") or pdf.stem
    curves = []
    for k, t in enumerate(log.traces):
        s = VectorCurveSession(pdf, vp.page_index, rgb, scale, vp.height, t, vp.depth_fit, vp.depth_labels, well, las)
        c = t.info
        curves.append(dict(id=f"c{k + 1}", track=f"x {c['x0']:.0f}-{c['x1']:.0f} pt", cls=_hex(t.color), kind="single",
                           pattern="vector path", color=_hex(t.color) if _hex(t.color) not in ("#000000",) else TAB10[k % 10],
                           coverage=None, name=t.mnemonic, unit=t.unit, header=c["name"], review=[], ref=None, s=s,
                           track_px=(float(c["x0"]) * scale, float(c["x1"]) * scale)))
    if not curves:
        raise ValueError(f"{pdf.name}: no vector curves found")
    return curves[0]["s"], curves, rgb, log
