# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""grid_values: a value for every vertical grid line of a track from OCR'd scale labels.

Labels (x, value) from the header scale line are matched to the nearest grid line, a robust
linear (value ~ x) or logarithmic (log10 value ~ x) model is fitted over all label pairs
(RANSAC-style: every pair proposes a model, the one with most agreeing labels wins), inconsistent
labels are flagged with their suggested value, and every grid line gets a value. ``KnotScale``
maps pixel column -> value piecewise between labelled lines (rectified space), so mild residual
non-uniformity between lines is honoured; it behaves like ``ScaleMap`` for tracing/export.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _nice(v: float) -> float:
    if v == 0 or not np.isfinite(v):
        return v
    e = 10 ** np.floor(np.log10(abs(v)))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8, 10):
        if abs(abs(v) / e - m) < 0.02 * m:
            return float(np.sign(v) * m * e)
    return float(v)


def _fit(xs, ys):
    A = np.c_[np.ones(len(xs)), xs]
    return np.linalg.lstsq(A, ys, rcond=None)[0]


def fit_line_values(line_x, labels, kind: str | None = None, tol_lines: float = 0.25) -> dict | None:
    """``line_x``: x of each vertical grid line (sorted). ``labels``: [(x, value), ...].
    ``kind``: 'linear', 'log' or None (choose). Returns dict(kind, values[n], coef, inliers,
    outliers=[{x, read, suggested}], residual) or None with < 2 usable labels.
    A label agrees when its model position is within ``tol_lines`` of a line spacing."""
    lx = np.sort(np.asarray(line_x, float))
    lab = [(float(x), float(v)) for x, v in labels if v is not None and np.isfinite(v)]
    if len(lab) < 2 or len(lx) < 2:
        return None
    sp = float(np.median(np.diff(lx)))
    kinds = [kind] if kind else (["linear", "log"] if all(v > 0 for _, v in lab) else ["linear"])
    best = None
    for k in kinds:
        xs = np.array([x for x, _ in lab]); vs = np.array([v for _, v in lab])
        if k == "log":
            if np.any(vs <= 0):
                continue
            ys = np.log10(vs)
        else:
            ys = vs
        for a in range(len(xs)):
            for b in range(a + 1, len(xs)):
                if abs(xs[b] - xs[a]) < 0.5 * sp or ys[b] == ys[a]:
                    continue
                c = _fit(xs[[a, b]], ys[[a, b]])
                # position error in pixels of each label under this model
                xp = (ys - c[0]) / c[1]
                inl = np.abs(xp - xs) < tol_lines * sp
                if inl.sum() < 2:
                    continue
                c2 = _fit(xs[inl], ys[inl])
                res = float(np.sqrt(np.mean(((ys[inl] - c2[0]) / c2[1] - xs[inl]) ** 2)))
                score = (int(inl.sum()), -res)
                if best is None or score > best[0]:
                    best = (score, k, c2, inl, res, xs, vs)
    if best is None:
        return None
    (_, _), k, c, inl, res, xs, vs = best
    model = (lambda x: c[0] + c[1] * np.asarray(x, float)) if k == "linear" else \
            (lambda x: 10 ** (c[0] + c[1] * np.asarray(x, float)))
    raw = model(lx)
    span = float(np.max(np.abs(raw))) or 1.0
    vals = np.array([0.0 if abs(v) < 1e-9 * span else _nice(float(f"{v:.10g}")) for v in raw])
    outl = [dict(x=float(x), read=float(v), suggested=_nice(float(model(x))))
            for x, v, ok in zip(xs, vs, inl) if not ok]
    return dict(kind=k, coef=c.tolist(), line_x=lx.tolist(), values=vals.tolist(), inliers=int(inl.sum()),
                n_labels=len(xs), outliers=outl, residual_px=res)


def check_divisions(fit: dict) -> list[str]:
    """Consistency warnings: linear lines must step evenly; log lines must repeat per decade."""
    w = []
    v = np.asarray(fit["values"], float)
    if fit["kind"] == "linear":
        d = np.diff(v)
        if len(d) and np.any(np.abs(d - np.median(d)) > 0.02 * abs(np.median(d)) + 1e-12):
            w.append("uneven value steps between grid lines")
    else:
        ld = np.log10(v)
        if ld.max() - ld.min() < 0.9:
            w.append("log track spans less than one decade")
    if fit["outliers"]:
        w.append(f"{len(fit['outliers'])} label(s) disagree with the grid: "
                 + ", ".join(f"{o['read']:g}->{o['suggested']:g}" for o in fit["outliers"]))
    return w


def set_line_value(fit: dict, j: int, value: float) -> dict:
    """User correction: grid line ``j`` has ``value``. Refit with it as a fixed label plus the
    current inlier lines' values (so one click moves the whole track consistently)."""
    lx = np.asarray(fit["line_x"]); vals = list(fit["values"])
    # keep the far end fixed: the edited line + opposite end define the model
    far = len(lx) - 1 if j < len(lx) / 2 else 0
    out = fit_line_values(lx, [(lx[j], value), (lx[far], vals[far])], fit["kind"], tol_lines=0.25)
    return out or fit


@dataclass
class KnotScale:
    """Piecewise map x -> value through grid-line knots (linear in value or in log10 value)."""
    xs: np.ndarray
    values: np.ndarray
    kind: str = "linear"

    def __post_init__(self):
        o = np.argsort(self.xs)
        self.xs = np.asarray(self.xs, float)[o]; self.values = np.asarray(self.values, float)[o]

    @property
    def x_left(self): return float(self.xs[0])
    @property
    def x_right(self): return float(self.xs[-1])
    @property
    def v_left(self): return float(self.values[0])
    @property
    def v_right(self): return float(self.values[-1])

    def _y(self):
        return np.log10(self.values) if self.kind == "log" else self.values

    def __call__(self, x):
        x = np.asarray(x, float); y = self._y(); xs = self.xs
        # interpolate, extrapolate linearly with the end segments (wrap/off-scale handled elsewhere)
        out = np.interp(x, xs, y)
        lo, hi = x < xs[0], x > xs[-1]
        out = np.where(lo, y[0] + (x - xs[0]) * (y[1] - y[0]) / (xs[1] - xs[0]), out)
        out = np.where(hi, y[-1] + (x - xs[-1]) * (y[-1] - y[-2]) / (xs[-1] - xs[-2]), out)
        return 10 ** out if self.kind == "log" else out

    def inverse(self, v):
        y = np.log10(np.asarray(v, float)) if self.kind == "log" else np.asarray(v, float)
        yy = self._y()
        if yy[-1] < yy[0]:
            return np.interp(y, yy[::-1], self.xs[::-1])
        return np.interp(y, yy, self.xs)


def knot_scale(fit: dict) -> KnotScale:
    return KnotScale(np.asarray(fit["line_x"]), np.asarray(fit["values"]), fit["kind"])
