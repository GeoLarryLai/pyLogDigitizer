# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""grid_rectify: grid-line lattice detection and 2-D rectification of warped (non-flat) log scans.

A ``Lattice`` holds sheet positions ``X[i, j], Y[i, j]`` of the intersection of horizontal grid
line ``i`` with vertical grid line ``j`` and their ideal (rectified) positions ``u[j]`` (columns) and
``v[i]`` (rows). The forward map rectified (u, v) -> sheet (x, y) is piecewise bilinear over the
lattice cells (clamped extrapolation outside); ``to_rectified`` inverts it by Newton iteration.
``tps_fit`` / ``tps_eval`` give a thin-plate spline for sparse control points (editor drags).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ----------------------------------------------------------------------------- line filters
def line_response(ink: np.ndarray, axis: str = "v", length: int = 61, tilt: int = 3) -> np.ndarray:
    """Pixels on straight-ish runs >= ``length`` px along ``axis`` ('v' vertical, 'h' horizontal)
    (morphological opening with a 1 x length line). Curves crossing at an angle are removed."""
    import cv2
    a = ink.astype(np.uint8)
    if tilt > 0:   # widen across the line first so slightly tilted / bent lines survive the opening
        d = np.ones((1, 2 * tilt + 1) if axis == "v" else (2 * tilt + 1, 1), np.uint8)
        a = cv2.dilate(a, d)
    k = np.ones((length, 1) if axis == "v" else (1, length), np.uint8)
    return cv2.morphologyEx(a, cv2.MORPH_OPEN, k).astype(bool) & np.asarray(ink, bool)


def _peaks(profile: np.ndarray, min_frac: float, min_sep: int) -> np.ndarray:
    """Centres of runs where ``profile >= min_frac``; runs closer than ``min_sep`` keep the stronger."""
    m = profile >= min_frac
    e = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
    runs = list(zip(e[::2], e[1::2]))
    out: list[tuple[float, float]] = []
    for a, b in runs:
        c = a + (b - 1 - a) / 2
        s = float(profile[a:b].sum())
        if out and c - out[-1][0] < min_sep:
            if s > out[-1][1]:
                out[-1] = (c, s)
            continue
        out.append((c, s))
    return np.array([c for c, _ in out], float)


def _track(resp: np.ndarray, starts: np.ndarray, block: int, search: int, min_frac: float):
    """Follow lines (columns of ``resp``) down the rows in blocks; returns (centres, pos[nb, n])."""
    prev = starts.astype(float).copy()
    cen, pos, hit = [], [], []
    for s in range(0, resp.shape[0], block):
        b = resp[s:s + block]
        if b.shape[0] < max(8, block // 3) and cen:
            break
        cf = b.mean(axis=0)
        cur = prev.copy(); h = np.zeros(len(prev), bool)
        for k, x in enumerate(prev):
            a, e = max(0, int(round(x)) - search), min(b.shape[1], int(round(x)) + search + 1)
            seg = cf[a:e]
            if seg.size and seg.max() >= min_frac:
                w = np.where(seg >= 0.6 * seg.max(), seg, 0.0)
                # nearest strong run to the previous position
                idx = np.flatnonzero(w)
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                best = min(runs, key=lambda r: abs(a + r.mean() - x) - 1e-3 * w[r].sum())
                cur[k] = a + float((best * w[best]).sum() / w[best].sum()); h[k] = True
        cen.append(s + b.shape[0] / 2)
        pos.append(cur); hit.append(h)
        prev = cur
    return np.array(cen), np.array(pos), np.array(hit)


def _seed_track(resp, min_frac, min_sep, block, search, seed, min_cover=0.7):
    """Seed lines in the ``seed``-row band with the most line peaks, then follow them both ways."""
    best = None
    for s0 in range(0, max(1, resp.shape[0] - seed + 1), seed):
        pk = _peaks(resp[s0:s0 + seed].mean(0), min_frac, min_sep)
        if best is None or len(pk) > len(best[1]):
            best = (s0, pk)
    s0, pk = best
    if len(pk) < 2:
        return None, None
    a = s0 + seed // 2
    cd, pd, hd = _track(resp[a:], pk, block, search, min_frac * 0.5)
    cen, pos, hit = a + cd, pd, hd
    if a > block // 3:
        cu, pu, hu = _track(resp[:a][::-1], pk, block, search, min_frac * 0.5)
        cen, pos, hit = np.r_[a - cu[::-1], cen], np.r_[pu[::-1], pos], np.r_[hu[::-1], hit]
    keep = hit.mean(0) >= min_cover   # grid lines run (almost) the whole window; curve pieces don't
    if keep.sum() < 2:
        return None, None
    pos, hit = pos[:, keep], hit[:, keep]
    o = np.argsort(np.median(pos, 0)); pos, hit = pos[:, o], hit[:, o]
    out = [0]
    for k in range(1, pos.shape[1]):   # a seed on a curve that drifted onto a grid line: keep one
        if np.median(np.abs(pos[:, k] - pos[:, out[-1]])) < min_sep:
            if hit[:, k].sum() > hit[:, out[-1]].sum():
                out[-1] = k
            continue
        out.append(k)
    return cen, pos[:, out]


def _smooth(p: np.ndarray, k: int = 3) -> np.ndarray:
    if p.shape[0] < 3 or k < 2:
        return p
    pad = np.pad(p, ((k // 2, k // 2), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(p.shape[1])], 1)


# ----------------------------------------------------------------------------- lattice
@dataclass
class Lattice:
    X: np.ndarray            # (nh, nv) sheet x of each intersection
    Y: np.ndarray            # (nh, nv) sheet y
    u: np.ndarray            # (nv,) rectified column of each vertical line
    v: np.ndarray            # (nh,) rectified row of each horizontal line
    shape: tuple[int, int] = (0, 0)   # rectified output (rows, cols)
    locked: np.ndarray | None = field(default=None)

    # -- serialisation (app state / undo)
    def to_dict(self) -> dict:
        return dict(X=self.X.tolist(), Y=self.Y.tolist(), u=self.u.tolist(), v=self.v.tolist(),
                    shape=list(self.shape))

    @classmethod
    def from_dict(cls, d: dict) -> "Lattice":
        return cls(np.asarray(d["X"], float), np.asarray(d["Y"], float), np.asarray(d["u"], float),
                   np.asarray(d["v"], float), tuple(d["shape"]))

    # -- forward map rectified -> sheet
    def to_sheet(self, uu, vv):
        uu = np.asarray(uu, float); vv = np.asarray(vv, float)
        j = np.clip(np.searchsorted(self.u, uu, side="right") - 1, 0, len(self.u) - 2)
        i = np.clip(np.searchsorted(self.v, vv, side="right") - 1, 0, len(self.v) - 2)
        s = (uu - self.u[j]) / (self.u[j + 1] - self.u[j])
        t = (vv - self.v[i]) / (self.v[i + 1] - self.v[i])
        def bil(A):
            return ((1 - s) * (1 - t) * A[i, j] + s * (1 - t) * A[i, j + 1]
                    + (1 - s) * t * A[i + 1, j] + s * t * A[i + 1, j + 1])
        return bil(self.X), bil(self.Y)

    # -- inverse map sheet -> rectified (Newton on the bilinear map, numeric Jacobian)
    def to_rectified(self, xx, yy, iters: int = 20, tol: float = 1e-3):
        xx = np.asarray(xx, float); yy = np.asarray(yy, float)
        # initial guess from the global affine fit
        A = np.c_[self.X.ravel(), self.Y.ravel(), np.ones(self.X.size)]
        U = np.broadcast_to(self.u[None, :], self.X.shape).ravel()
        V = np.broadcast_to(self.v[:, None], self.X.shape).ravel()
        cu = np.linalg.lstsq(A, U, rcond=None)[0]; cv_ = np.linalg.lstsq(A, V, rcond=None)[0]
        uu = cu[0] * xx + cu[1] * yy + cu[2]; vv = cv_[0] * xx + cv_[1] * yy + cv_[2]
        h = 0.5
        for _ in range(iters):
            fx, fy = self.to_sheet(uu, vv)
            ex, ey = xx - fx, yy - fy
            if np.all(np.abs(ex) < tol) and np.all(np.abs(ey) < tol):
                break
            ax, ay = self.to_sheet(uu + h, vv); bx, by = self.to_sheet(uu, vv + h)
            j11, j21 = (ax - fx) / h, (ay - fy) / h
            j12, j22 = (bx - fx) / h, (by - fy) / h
            det = j11 * j22 - j12 * j21
            det = np.where(np.abs(det) < 1e-9, 1e-9, det)
            uu = uu + (j22 * ex - j12 * ey) / det
            vv = vv + (-j21 * ex + j11 * ey) / det
        return uu, vv

    def move_node(self, i: int, j: int, x: float, y: float, radius: int = 0) -> "Lattice":
        """Move intersection (i, j) to sheet (x, y). With ``radius`` > 0 neighbouring nodes follow
        with a smooth (cosine) fall-off so a drag reshapes the grid locally."""
        X, Y = self.X.copy(), self.Y.copy()
        dx, dy = x - X[i, j], y - Y[i, j]
        if radius <= 0:
            X[i, j] += dx; Y[i, j] += dy
        else:
            ii, jj = np.mgrid[0:X.shape[0], 0:X.shape[1]]
            d = np.hypot(ii - i, jj - j) / (radius + 1)
            w = np.where(d < 1, 0.5 * (1 + np.cos(np.pi * d)), 0.0)
            X += w * dx; Y += w * dy
        return Lattice(X, Y, self.u.copy(), self.v.copy(), self.shape)

    def residual(self) -> float:
        """RMS deviation (px) of the lattice from a pure affine grid: a measure of scan warp."""
        A = np.c_[np.broadcast_to(self.u[None], self.X.shape).ravel(),
                  np.broadcast_to(self.v[:, None], self.X.shape).ravel(), np.ones(self.X.size)]
        r = []
        for B in (self.X.ravel(), self.Y.ravel()):
            c = np.linalg.lstsq(A, B, rcond=None)[0]
            r.append(B - A @ c)
        return float(np.sqrt(np.mean(np.square(r))))


def _intersections(cv, pv, ch, ph, n_iter: int = 6):
    """Intersect vertical lines x_j(y) (centres ``cv``, positions ``pv[nb, nv]``) with horizontal
    lines y_i(x) (``ch``, ``ph[nb, nh]``)."""
    nv, nh = pv.shape[1], ph.shape[1]
    X = np.zeros((nh, nv)); Y = np.zeros((nh, nv))
    for i in range(nh):
        for j in range(nv):
            y = float(np.median(ph[:, i])); x = float(np.median(pv[:, j]))
            for _ in range(n_iter):
                x = float(np.interp(y, cv, pv[:, j]))
                y = float(np.interp(x, ch, ph[:, i]))
            X[i, j], Y[i, j] = x, y
    return X, Y


def regularize(lat: Lattice) -> Lattice:
    """Ideal positions: vertical lines at their mean x, horizontal lines at their mean y, then
    snapped onto a regular spacing when the spacing is near-uniform (within 15 %)."""
    def reg(p):
        d = np.diff(p)
        if len(d) >= 2 and np.all(np.abs(d - np.median(d)) < 0.15 * np.median(d)):
            k = np.arange(len(p)); c = np.polyfit(k, p, 1)
            return np.polyval(c, k)
        return p
    return Lattice(lat.X, lat.Y, reg(lat.X.mean(0)), reg(lat.Y.mean(1)), lat.shape)


def detect_lattice(ink: np.ndarray, length: int = 61, min_frac_v: float = 0.35, min_frac_h: float = 0.35,
                   min_sep: int = 8, block: int = 100, search: int = 10, smooth: int = 3,
                   seed: int = 120) -> Lattice | None:
    """Detect the grid lattice of one track (or a whole sheet window) from a boolean ink image.
    Vertical lines are found in the top block and followed down; horizontal lines in the left
    block and followed right. Returns None when fewer than 2 x 2 lines are found."""
    ink = np.asarray(ink, bool)
    rv, rh = line_response(ink, "v", length), line_response(ink, "h", length)
    cv, pv = _seed_track(rv, min_frac_v, min_sep, block, search, seed)
    ch, ph = _seed_track(rh.T, min_frac_h, min_sep, block, search, seed)
    if pv is None or ph is None:
        return None
    pv, ph = _smooth(pv, smooth), _smooth(ph, smooth)
    X, Y = _intersections(cv, pv, ch, ph)
    lat = regularize(Lattice(X, Y, np.zeros(X.shape[1]), np.zeros(X.shape[0]), ink.shape))
    return lat


def lattice_from_lines(xs_rows: np.ndarray, centers: np.ndarray, ys: np.ndarray) -> Lattice:
    """Lattice from tracked vertical lines only (``xs_rows[nb, nv]`` at block ``centers``) and a
    list of horizontal grid rows ``ys``: the depth direction is taken as unwarped."""
    X = np.stack([np.array([np.interp(y, centers, xs_rows[:, j]) for j in range(xs_rows.shape[1])]) for y in ys])
    Y = np.broadcast_to(np.asarray(ys, float)[:, None], X.shape).copy()
    return regularize(Lattice(X, Y, np.zeros(X.shape[1]), np.zeros(X.shape[0])))


def x_only(lat: Lattice, shape: tuple[int, int]) -> Lattice:
    """Keep only the across-track (x) warp: horizontal lines at their mean row, rectified rows =
    sheet rows (depth stays with the depth calibration). Bands with no horizontal lines span the panel."""
    Y = np.broadcast_to(lat.Y.mean(1, keepdims=True), lat.X.shape).copy()
    o = np.argsort(Y[:, 0]); keep = np.r_[True, np.diff(Y[o, 0]) > 0.5]
    X, Y = lat.X[o][keep], Y[o][keep]
    if len(Y) < 2:
        X = np.r_[X[:1], X[:1]]; Y = np.c_[[0.0] * X.shape[1], [shape[0] - 1.0] * X.shape[1]].T
    return Lattice(X, Y, lat.u.copy(), Y[:, 0].copy(), tuple(shape))


def rectify_panel(pan: np.ndarray, min_lines: int = 3, **kw):
    """Detect + straighten (x-only) a track panel. Returns (rectified panel, lattice) or (pan, None)."""
    lat = detect_lattice(pan, **kw)
    if lat is None or lat.X.shape[1] < min_lines:
        return pan, None
    lat = x_only(lat, pan.shape)
    return rectify_image(pan, lat, pan.shape), lat


# ----------------------------------------------------------------------------- resampling
def rectify_image(img: np.ndarray, lat: Lattice, out_shape: tuple[int, int] | None = None,
                  window: tuple[int, int, int, int] | None = None, order: int = 1,
                  chunk: int = 8000) -> np.ndarray:
    """Resample ``img`` into rectified space. ``window`` = (r0, r1, c0, c1) in rectified
    coordinates (for previews / tiles). Bilinear (order 1) or nearest (order 0)."""
    import cv2
    H, W = out_shape or (lat.shape if lat.shape[0] else img.shape[:2])
    r0, r1, c0, c1 = window or (0, H, 0, W)
    src = img.astype(np.uint8) * (255 if img.dtype == bool else 1)
    bv = 0 if img.dtype == bool else 255
    out = np.empty((r1 - r0, c1 - c0), np.uint8)
    for a in range(r0, r1, chunk):        # cv2.remap needs < 32767 rows: chunk with a source slab
        b = min(r1, a + chunk)
        vv, uu = np.mgrid[a:b, c0:c1].astype(np.float32)
        mx, my = lat.to_sheet(uu, vv)
        s0 = int(max(0, np.floor(my.min()) - 2)); s1 = int(min(src.shape[0], np.ceil(my.max()) + 3))
        if s1 <= s0:
            out[a - r0:b - r0] = bv; continue
        out[a - r0:b - r0] = cv2.remap(src[s0:s1], mx.astype(np.float32), (my - s0).astype(np.float32),
                                       cv2.INTER_LINEAR if order else cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    return out > 127 if img.dtype == bool else out


def straightness(ink_rect: np.ndarray, lat: Lattice, length: int = 61, band: int = 6) -> dict:
    """Grid straightness in a rectified image: for every vertical / horizontal line, the
    per-block centroid deviation (px) from its ideal position. Returns p95 and max."""
    rv, rh = line_response(ink_rect, "v", length), line_response(ink_rect, "h", length)
    dev = []
    for u in lat.u:
        a, b = int(max(0, u - band)), int(min(ink_rect.shape[1], u + band + 1))
        for s in range(0, ink_rect.shape[0] - 100, 100):
            w = rv[s:s + 100, a:b].sum(0)
            if w.sum():
                dev.append(abs(a + (np.arange(b - a) * w).sum() / w.sum() - u))
    for v in lat.v:
        a, b = int(max(0, v - band)), int(min(ink_rect.shape[0], v + band + 1))
        for s in range(0, ink_rect.shape[1] - 100, 100):
            w = rh[a:b, s:s + 100].sum(1)
            if w.sum():
                dev.append(abs(a + (np.arange(b - a) * w).sum() / w.sum() - v))
    dev = np.array(dev) if dev else np.zeros(1)
    return dict(p95=float(np.percentile(dev, 95)), max=float(dev.max()), n=int(dev.size))


# ----------------------------------------------------------------------------- thin-plate spline
def _U(r2):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 0, r2 * np.log(r2), 0.0) * 0.5


def tps_fit(src: np.ndarray, dst: np.ndarray, lam: float = 0.0) -> dict:
    """Thin-plate spline mapping ``src`` (n, 2) -> ``dst`` (n, 2), regularized by ``lam``
    (Bookstein 1989). Written from the published formulation."""
    src = np.asarray(src, float); dst = np.asarray(dst, float); n = len(src)
    K = _U(((src[:, None] - src[None]) ** 2).sum(-1)) + lam * np.eye(n)
    P = np.c_[np.ones(n), src]
    L = np.zeros((n + 3, n + 3)); L[:n, :n] = K; L[:n, n:] = P; L[n:, :n] = P.T
    b = np.zeros((n + 3, 2)); b[:n] = dst
    return dict(src=src, coef=np.linalg.solve(L, b))


def tps_eval(t: dict, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float); s = t["src"]; c = t["coef"]; n = len(s)
    out = np.empty_like(pts)
    for a in range(0, len(pts), 20000):
        p = pts[a:a + 20000]
        U = _U(((p[:, None] - s[None]) ** 2).sum(-1))
        out[a:a + 20000] = U @ c[:n] + c[n] + p @ c[n + 1:]
    return out


def lattice_from_tps(lat: Lattice, src_pts, dst_pts, lam: float = 0.0) -> Lattice:
    """Deform all lattice nodes with a TPS fitted to sparse control moves (src -> dst sheet)."""
    t = tps_fit(src_pts, dst_pts, lam)
    P = tps_eval(t, np.c_[lat.X.ravel(), lat.Y.ravel()])
    return Lattice(P[:, 0].reshape(lat.X.shape), P[:, 1].reshape(lat.X.shape), lat.u.copy(), lat.v.copy(), lat.shape)


# ----------------------------------------------------------------------------- synthetic warped grid (tests / demos)
def synthetic_grid(H: int = 1200, W: int = 600, nv: int = 11, nh: int = 13, margin: int = 40, amp: float = 12.0,
                   shear: float = 0.02, line: int = 2, seed: int = 0):
    """Bool ink image of a regular grid warped by a smooth bulge + shear, and the true lattice.
    The true forward map is analytic so tests can check lattice recovery and round trips."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 2)
    def warp(u, v):
        x = u + amp * np.sin(np.pi * v / H + ph[0]) * np.sin(np.pi * u / W) + shear * (v - H / 2)
        y = v + 0.5 * amp * np.sin(np.pi * u / W + ph[1]) * np.sin(np.pi * v / H)
        return x, y
    u = np.linspace(margin, W - margin, nv); v = np.linspace(margin, H - margin, nh)
    ink = np.zeros((H, W), bool)
    t = np.linspace(0, 1, 4 * max(H, W))
    for uj in u:
        x, y = warp(np.full_like(t, uj), margin + t * (H - 2 * margin))
        _stamp(ink, x, y, line)
    for vi in v:
        x, y = warp(margin + t * (W - 2 * margin), np.full_like(t, vi))
        _stamp(ink, x, y, line)
    U, V = np.meshgrid(u, v)
    X, Y = warp(U, V)
    return ink, Lattice(X, Y, u, v, (H, W)), warp


def _stamp(ink, x, y, w):
    for dx in range(-(w // 2), w - w // 2):
        for dy in range(-(w // 2), w - w // 2):
            xi = np.clip(np.round(x + dx).astype(int), 0, ink.shape[1] - 1)
            yi = np.clip(np.round(y + dy).astype(int), 0, ink.shape[0] - 1)
            ink[yi, xi] = True
