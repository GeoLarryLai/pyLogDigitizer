# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Overlay QA for a raster digitization, drawn on the ORIGINAL scan (full sheet, every track):

* grid lines of every track, each snapped to the printed line locally (per row block for vertical
  lines, per x segment for horizontal ones, sub-pixel centroid), so skew / paper warp is followed;
* followed track edges;
* the traced curve(s) at the pixel where their ink is printed (wrapped / overflow segments sit on
  their wrapped ink, not on the unwrapped value);
* optional side panel with the unwrapped values on the same rows.

Coordinates go panel (grid-rectified) -> deskewed strip -> sheet, inverting each step exactly.
OpenCV only (no extra runtime dependency)."""
from __future__ import annotations

import numpy as np

# BGR
CURVE_C = [(0, 0, 230), (200, 60, 0), (0, 150, 0), (180, 0, 180)]   # picked curve first (red)
EDGE_C, GRID_C, VALUE_C = (40, 160, 40), (0, 150, 255), (0, 0, 140)


# ----------------------------------------------------------------------------- coordinate maps
def strip_to_sheet(L, x, y, deskew_block: int = 2000):
    """Deskewed-strip (x, y) -> original sheet pixel (x, y). Inverts ``deskew_blocks`` (vertical
    shear about the strip centre, one angle per ``deskew_block`` rows)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ang = np.asarray(L.skew_deg, float).ravel()
    b = np.clip((y // deskew_block).astype(int), 0, max(len(ang) - 1, 0))
    t = np.tan(np.deg2rad(ang[b])) if len(ang) else 0.0
    xc = L.ink.shape[1] / 2.0
    return x + L.x_offset, y - np.round((x - xc) * t) + L.y_offset   # same rounding as the shear


def _edge_x(L, name: str, rows):
    k = L.edge_names.index(name)
    return np.interp(rows, L.centers, L.edges[:, k])


def panel_to_strip(L, track: str, rows, x, width: int, lattice=None, r0: int = 0):
    """Panel column ``x`` (grid-rectified when ``lattice`` is given) at strip ``rows`` -> strip x."""
    rows = np.asarray(rows, float); x = np.asarray(x, float)
    if lattice is not None:
        x, _ = lattice.to_sheet(x, rows - r0)
    xl, xr = _edge_x(L, track + ".left", rows), _edge_x(L, track + ".right", rows)
    return xl + (x + 0.5) / width * (xr - xl)


# ----------------------------------------------------------------------------- grid refinement
def _runs(idx: np.ndarray) -> list[float]:
    if len(idx) == 0:
        return []
    g = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [float(np.mean(s)) for s in g]


def _centroid(prof: np.ndarray, min_val: float):
    """Sub-pixel centre of the run around the profile maximum (None when too weak)."""
    k = int(np.argmax(prof))
    if prof[k] < min_val:
        return None
    thr = 0.5 * prof[k]; a = k; b = k
    while a > 0 and prof[a - 1] >= thr:
        a -= 1
    while b < len(prof) - 1 and prof[b + 1] >= thr:
        b += 1
    w = prof[a:b + 1].astype(float)
    return float((np.arange(a, b + 1) * w).sum() / w.sum())


def _line_cols(P: np.ndarray, frac: float, blk: int = 100, cover: float = 0.5, contrast: float = 0.25) -> np.ndarray:
    """Columns of ``P`` that are printed lines: inked in >= ``frac`` of the rows of at least
    ``cover`` of the ``blk``-row blocks, and clearly darker than the columns 4 px either side
    (a curve that runs vertically for a while is wide and does not pass the contrast test)."""
    n = P.shape[0] // blk
    if n < 1:
        m = P.mean(0)[None]
    else:
        m = P[:n * blk].reshape(n, blk, -1).mean(1)
    sh = lambda k: np.roll(m, k, 1)
    loc = m - np.maximum(sh(4), sh(-4))
    good = (m >= frac) & (loc >= contrast)
    return good.mean(0) >= cover


def refine_grid(L, track: str, rows: tuple[int, int], v_frac: float = 0.35, h_frac: float = 0.45,
                block: int = 120, seg: int = 60, search: int = 5) -> dict:
    """Detect a track's grid lines and snap each to the printed ink.

    Vertical lines: per ``block`` rows, the ink-column profile within +-``search`` px of the line's
    current position gives a sub-pixel centre (median-smoothed over 3 blocks, interpolated per row).
    Horizontal lines: per ``seg`` px x segment, the ink-row profile within +-``search`` rows.
    Returns {'v': [(ys, xs)], 'h': [(xs, ys)], 'residual': {...}} in strip coordinates, with the
    residual (|line - ink centre| measured on independent half-size windows) before and after."""
    a, b = int(rows[0]), int(rows[1])
    W = L.width_of(track); P = L.panel(track, (a, b), W)
    ink = L.ink; H, Wi = ink.shape
    vcols = _runs(np.flatnonzero(_line_cols(P, v_frac)))
    from pylogdigitizer import grid_rectify as gr          # bent lines: follow them block by block
    cv_, pv_ = gr._seed_track(gr.line_response(P, "v", 61), 0.35, 8, 100, 6, 120)
    tracked = [] if pv_ is None else [pv_[:, k] for k in range(pv_.shape[1])]
    vcols = [c for c in vcols if all(np.min(np.abs(t - c)) > 6 for t in tracked)]
    hrows = _runs(np.flatnonzero(_line_cols(P.T, h_frac, blk=max(8, P.shape[1] // 8), cover=0.5)))
    yy = np.arange(a, b)
    xl, xr = _edge_x(L, track + ".left", yy), _edge_x(L, track + ".right", yy)
    d = np.arange(-search, search + 1)

    def vprof(xn, y0, y1):        # ink count per offset d over rows y0..y1 along nominal x
        ys = np.arange(y0, y1); xi = np.clip(np.round(xn[ys - a]).astype(int)[:, None] + d, 0, Wi - 1)
        return ink[ys[:, None], xi].sum(0), np.round(xn[ys - a]) - xn[ys - a]

    def vfit(xn, blk):
        cs, offs = [], []
        for y0 in range(a, b, blk):
            y1 = min(b, y0 + blk); p, fr = vprof(xn, y0, y1)
            c = _centroid(p, 0.6 * (y1 - y0))
            cs.append((y0 + y1) / 2); offs.append(np.nan if c is None else c - search + float(np.mean(fr)))
        return np.array(cs), np.array(offs)

    out = {"v": [], "h": [], "v0": [], "h0": [], "residual": {}}
    rv_b, rv_a = [], []
    paths = [np.interp(yy - a, cv_, t) for t in tracked] + [np.full(len(yy), c) for c in vcols]
    for c in paths:
        xn = xl + (c + 0.5) / W * (xr - xl)
        cy, off = vfit(xn, block)
        ok = np.isfinite(off)
        if ok.sum() >= 2:
            o = off.copy(); o[ok] = [np.median(off[ok][max(0, i - 1):i + 2]) for i in range(ok.sum())]
            xs = xn + np.interp(yy, cy[ok], o[ok])
        else:
            xs = xn
        out["v"].append((yy.astype(float), xs)); out["v0"].append((yy.astype(float), xn))
        _, e0 = vfit(xn, block // 2); _, e1 = vfit(xs, block // 2)
        rv_b += list(np.abs(e0[np.isfinite(e0)])); rv_a += list(np.abs(e1[np.isfinite(e1)]))

    def hfit(yn_of_x, x0, x1, segw):
        cs, offs = [], []
        for s in np.arange(x0, x1, segw):
            xs_ = np.arange(int(s), int(min(x1, s + segw)))
            if len(xs_) < 4:
                continue
            yi = np.round(yn_of_x(xs_)).astype(int)
            prof = ink[np.clip(yi[:, None] + d, 0, H - 1).T, xs_[None, :]].sum(1)
            c = _centroid(prof, 0.6 * len(xs_))
            cs.append(xs_.mean()); offs.append(np.nan if c is None else c - search + float(np.mean(yi - yn_of_x(xs_))))
        return np.array(cs), np.array(offs)

    rh_b, rh_a = [], []
    for r in hrows:
        y = a + r; yi = int(round(y))
        x0, x1 = float(xl[min(len(yy) - 1, int(r))]), float(xr[min(len(yy) - 1, int(r))])
        cx, off = hfit(lambda xs_: np.full(len(xs_), y), x0 + 2, x1 - 2, seg)
        ok = np.isfinite(off)
        xs = np.linspace(x0, x1, 64)
        ys = y + (np.interp(xs, cx[ok], off[ok]) if ok.sum() >= 1 else 0.0)
        out["h"].append((xs, ys)); out["h0"].append((xs, np.full(len(xs), y)))
        f = lambda q, xs=xs, ys=ys: np.interp(q, xs, ys)
        _, e0 = hfit(lambda xs_: np.full(len(xs_), y), x0 + 2, x1 - 2, seg // 2); _, e1 = hfit(f, x0 + 2, x1 - 2, seg // 2)
        rh_b += list(np.abs(e0[np.isfinite(e0)])); rh_a += list(np.abs(e1[np.isfinite(e1)]))

    def st(v):
        v = np.asarray(v, float)
        return None if not len(v) else dict(median=round(float(np.median(v)), 2), p95=round(float(np.percentile(v, 95)), 2),
                                             max=round(float(v.max()), 2), n=int(len(v)))
    out["residual"] = dict(n_v=len(paths), n_h=len(hrows), v_before=st(rv_b), v_after=st(rv_a),
                           h_before=st(rh_b), h_after=st(rh_a))
    return out


def _profile_offsets(dark, P, axis, win, search, min_cov=0.6):
    """For a polyline P (N,2 sheet px, dense) measure the ink centre across it per ``win`` samples.
    axis 'v': line runs down the page (offset in x); 'h': line runs across (offset in y).
    Returns (centres along the line, offsets px, NaN where no clear printed line)."""
    H, W = dark.shape; d = np.arange(-search, search + 1)
    cs, offs = [], []
    for s0 in range(0, len(P) - 1, win):
        Q = P[s0:s0 + win]
        if len(Q) < max(3, win // 2):
            continue
        if axis == "v":
            yi = np.clip(np.round(Q[:, 1]).astype(int), 0, H - 1); xn = Q[:, 0]
            xi = np.clip(np.round(xn).astype(int)[:, None] + d, 0, W - 1); prof = dark[yi[:, None], xi].sum(0)
            fr = np.round(xn) - xn; c_along = Q[:, 1].mean()
        else:
            xi = np.clip(np.round(Q[:, 0]).astype(int), 0, W - 1); yn = Q[:, 1]
            yi = np.clip(np.round(yn).astype(int)[:, None] + d, 0, H - 1); prof = dark[yi, xi[:, None]].sum(0)
            fr = np.round(yn) - yn; c_along = Q[:, 0].mean()
        c = _centroid(prof, min_cov * len(Q))
        cs.append(c_along); offs.append(np.nan if c is None else c - search + float(fr.mean()))
    return np.array(cs), np.array(offs)


def _densify(P, step=1.0):
    P = np.asarray(P, float); seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    t = np.arange(0, seg[-1] + 1e-9, step)
    return np.c_[np.interp(t, seg, P[:, 0]), np.interp(t, seg, P[:, 1])]


def snap_line(dark, P, axis, win=None, search=5):
    """Snap a sheet polyline to the printed line under it (sub-pixel, piecewise per window, median
    smoothed). Returns (snapped dense polyline, residual before |px| array, residual after)."""
    win = win or (60 if axis == "v" else 24)
    D = _densify(P); k = 0 if axis == "v" else 1; j = 1 - k
    cs, off = _profile_offsets(dark, D, axis, win, search)
    ok = np.isfinite(off)
    S = D.copy()
    if ok.sum():
        o = off[ok]; o = np.array([np.median(o[max(0, i - 1):i + 2]) for i in range(len(o))])
        S[:, k] = D[:, k] + np.interp(D[:, j], cs[ok], o)
    _, e0 = _profile_offsets(dark, D, axis, win // 2, search)
    _, e1 = _profile_offsets(dark, S, axis, win // 2, search)
    return S, np.abs(e0[np.isfinite(e0)]), np.abs(e1[np.isfinite(e1)])


def _measure(dark, P, axis, search=5):
    """|offset| of a drawn sheet polyline from the printed line centre (same windows as snap_line's check)."""
    win = (60 if axis == "v" else 24) // 2
    _, e = _profile_offsets(dark, _densify(P), axis, win, search)
    return np.abs(e[np.isfinite(e)])


def _stats(v):
    v = np.asarray(v, float)
    return None if not len(v) else dict(median=round(float(np.median(v)), 2), p95=round(float(np.percentile(v, 95)), 2),
                                         max=round(float(v.max()), 2), n=int(len(v)))


# ----------------------------------------------------------------------------- overlay
def overlay_scan(scan_path, R: dict, track: str, out_path=None, rows: tuple[int, int] | None = None,
                 tracks: list[str] | None = None, side_panel: bool = True, scale: tuple[float, float] | None = None,
                 max_height: int = 8000, title: str | None = None):
    """Overlay ``R`` (from :func:`workflows.digitize_curve` on ``track``) on the original scan.

    Full sheet width, every track's snapped grid (orange), track edges (green), traced curve (red,
    other candidates blue/green/purple) at its printed ink position; optional side panel with the
    unwrapped value (dark red) on the same rows. ``rows`` (strip rows) crops a zoom; default is the
    traced section. Returns (BGR image, {track: grid residual})."""
    import cv2
    from pylogdigitizer.scan_io import load_scan
    L, T = R["layout"], R["trace"]
    pr0, pr1 = (int(v) for v in R["panel_rows"])
    a, b = (pr0, pr1) if rows is None else (max(0, int(rows[0])), min(L.ink.shape[0], int(rows[1])))
    # sheet window covering strip rows a..b (+ the skew shift)
    _, ys_top = strip_to_sheet(L, np.array([0, L.ink.shape[1]]), np.array([a, a]))
    _, ys_bot = strip_to_sheet(L, np.array([0, L.ink.shape[1]]), np.array([b, b]))
    s0 = int(max(0, np.floor(min(ys_top.min(), ys_bot.min())) - 5)); s1 = int(np.ceil(max(ys_top.max(), ys_bot.max())) + 5)
    x0 = int(L.x_offset); x1 = x0 + L.ink.shape[1]
    g = load_scan(scan_path, crop=(x0, s0, x1, s1))
    g = g.image if hasattr(g, "image") else np.asarray(g)
    img = cv2.cvtColor(np.asarray(g, np.uint8), cv2.COLOR_GRAY2BGR)
    img = (img.astype(np.float32) * 0.6 + 255 * 0.4).astype(np.uint8)       # fade the print a little
    toimg = lambda xs, ys: (lambda X, Y: np.c_[X - x0, Y - s0])(*strip_to_sheet(L, xs, ys))
    pl = lambda pts, col, w: cv2.polylines(img, [np.round(np.asarray(pts) * 16).astype(np.int32)], False, col, w, cv2.LINE_AA, 4)   # sub-pixel
    res = {}
    dark = np.asarray(g, np.uint8) < 160
    chunk = 3000                       # grid detection per chunk: faint / bent lines and local warp
    for nm in (tracks or L.track_names):
        acc = {k: [] for k in ("v0", "v", "h0", "h")}; nv = nh = 0
        for c0 in range(a, b, chunk):
            c1 = min(b, c0 + chunk)
            if c1 - c0 < 200:
                continue
            G = refine_grid(L, nm, (c0, c1)); nv = max(nv, len(G["v"])); nh += len(G["h"])
            for ax_, key in (("v", "v"), ("h", "h")):
                for (p_, q_), (p0, q0) in zip(G[key], G[key + "0"]):
                    st = 20 if ax_ == "v" else 1
                    xs, ys = (q_, p_) if ax_ == "v" else (p_, q_)
                    xs0, ys0 = (q0, p0) if ax_ == "v" else (p0, q0)
                    S, _, e1 = snap_line(dark, toimg(xs[::st], ys[::st]), ax_)
                    acc[key + "0"].extend(_measure(dark, toimg(xs0[::st], ys0[::st]), ax_))
                    acc[key].extend(e1)
                    pl(S[::4] if ax_ == "v" else S, GRID_C, 1)
        res[nm] = dict(n_v=nv, n_h=nh, v_before=_stats(acc["v0"]), v_after=_stats(acc["v"]),
                       h_before=_stats(acc["h0"]), h_after=_stats(acc["h"]))
        yy = np.arange(a, b, 10)
        for e in ("left", "right"):
            pl(toimg(_edge_x(L, f"{nm}.{e}", yy), yy), EDGE_C, 2)
    W = R["panel"].shape[1]; lat = R.get("grid_lattice")
    cands = T.get("candidates") or [T]
    pick = T.get("picked", 0) if T.get("candidates") else 0
    order = [pick] + [i for i in range(len(cands)) if i != pick]
    val = np.asarray(R["value"], float)
    for k, i in enumerate(order[::-1]):
        kk = len(order) - 1 - k
        t = cands[i]; rr = np.asarray(t["rows"], float); x = np.asarray(t["x"], float)
        ok = np.isfinite(x) & (rr >= a) & (rr < b)
        if i == pick and len(val) == len(x):
            ok &= np.isfinite(val)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        xs = panel_to_strip(L, track, rr[idx], x[idx], W, lat, pr0)
        P = toimg(xs, rr[idx])
        for s in np.split(np.arange(len(idx)), np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(s) > 1:
                pl(P[s], CURVE_C[kk % len(CURVE_C)], 2 if kk == 0 else 1)
    if side_panel:
        pw = 260; side = np.full((img.shape[0], pw, 3), 255, np.uint8)
        rr = np.asarray(T["rows"], float); ok = np.isfinite(val) & (rr >= a) & (rr < b)
        if ok.any():
            lo, hi = scale if scale else (np.nanmin(val), np.nanmax(val))
            _, Y = strip_to_sheet(L, np.full(ok.sum(), L.ink.shape[1] / 2), rr[ok])
            X = 10 + (val[ok] - lo) / max(hi - lo, 1e-9) * (pw - 20)
            for s in np.split(np.arange(ok.sum()), np.flatnonzero(np.diff(np.flatnonzero(ok)) > 1) + 1):
                if len(s) > 1:
                    cv2.polylines(side, [np.round(np.c_[X[s], Y[s] - s0]).astype(np.int32)], False, VALUE_C, 2, cv2.LINE_AA)
            for f in (0, 0.5, 1):
                xx = int(10 + f * (pw - 20)); side[:, xx] = (200, 200, 200)
            cv2.putText(side, f"{lo:.4g}", (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(side, f"{hi:.4g}", (pw - 70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        img = np.concatenate([img, np.full((img.shape[0], 6, 3), 90, np.uint8), side], 1)
    if img.shape[0] > max_height:
        f = max_height / img.shape[0]
        img = cv2.resize(img, (max(1, int(img.shape[1] * f)), max_height), interpolation=cv2.INTER_AREA)
    img = _legend(img, title or f"{track}: traced curve on the original scan")
    if out_path:
        cv2.imwrite(str(out_path), img)
    return img, res


def _legend(img, title: str):
    import cv2
    s = max(0.6, img.shape[1] / 2600)
    hh = int(110 * s); band = np.full((hh, img.shape[1], 3), 255, np.uint8)
    cv2.putText(band, title, (10, int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (0, 0, 0), max(1, int(2 * s)), cv2.LINE_AA)
    items = [("traced curve (on its printed ink)", CURVE_C[0]), ("other traced candidates", CURVE_C[1]),
             ("grid, snapped to print", GRID_C), ("track edges", EDGE_C), ("side panel: unwrapped value", VALUE_C)]
    x = 10
    for txt, col in items:
        cv2.line(band, (x, int(80 * s)), (x + int(40 * s), int(80 * s)), col, max(2, int(4 * s)))
        cv2.putText(band, txt, (x + int(48 * s), int(88 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0), max(1, int(1.5 * s)), cv2.LINE_AA)
        x += int((60 + 11 * len(txt)) * s)
    return np.concatenate([band, img], 0)
