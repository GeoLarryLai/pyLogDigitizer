"""Direction-aware, appearance-matching tracer (idea C).

Viterbi over samples (one every ``fy`` panel rows) with state (x, heading); heading is
L / V / R (the last step went left, straight, right). A step is short (<= ``max_step`` px, may
cross a gap) or long and sideways along connected ink of the current band (slopes and the out-
and back-strokes of near-flat spikes, cheap per pixel). Turning costs a curvature penalty; a
reversal L<->R (spike tip, zig-zag) is the dearest. The emission does not reward the darkest ink: it
rewards ink whose stroke width matches the target curve's width, learnt from the prior trace and
then per block from the first pass ("trace so far"); a dashed target gets a cheaper miss. Input is
normally the residual of :mod:`pylogdigitizer.grid_layer` (grid explained away). KGS scans are 1-bit,
so there is no grey level to match; width and dash pattern carry the appearance.
"""
from __future__ import annotations

import numpy as np

__all__ = ["band_ink", "width_map", "trace_appearance", "apply_appearance"]

L_, V_, R_ = 0, 1, 2
# curvature penalty between headings (L: moving left down the log, V: straight, R: right). A reversal
# L<->R is the dearest (a spike tip or a zig-zag); straight <-> slanted is cheap.
KINK = np.array([[0.0, 0.05, 0.3],
                 [0.05, 0.0, 0.05],
                 [0.3, 0.05, 0.0]])


def band_ink(ink: np.ndarray, fy: int, n: int | None = None, dilate: int = 1) -> np.ndarray:
    """OR of each sample's ``fy`` rows (plus ``dilate`` neighbouring samples each side)."""
    H, W = ink.shape
    n = n or -(-H // fy)
    p = np.zeros((n * fy, W), bool); p[:min(H, n * fy)] = ink[:n * fy]
    b = p.reshape(n, fy, W).any(1)
    out = b.copy()
    for k in range(1, dilate + 1):
        out[k:] |= b[:-k]; out[:-k] |= b[k:]
    return out


def width_map(ink: np.ndarray, fy: int, n: int | None = None, dilate: int = 1) -> np.ndarray:
    """Stroke width (2 x distance to paper) per sample band, max over the band (and ``dilate``
    neighbouring samples, matching :func:`band_ink`)."""
    import cv2
    H, W = ink.shape
    n = n or -(-H // fy)
    out = np.zeros((n, W), np.float32)
    ch = 6000 - 6000 % fy
    for s in range(0, n * fy, ch):
        e = min(H, s + ch)
        if e <= s:
            break
        d = cv2.distanceTransform(np.pad(ink[s:e], 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1] * 2
        m = (e - s) // fy * fy
        if m:
            out[s // fy:s // fy + m // fy] = d[:m].reshape(-1, fy, W).max(1)
        if e - s > m:
            out[s // fy + m // fy] = d[m:].max(0)
    b = out.copy()
    for k in range(1, dilate + 1):
        out[k:] = np.maximum(out[k:], b[:-k]); out[:-k] = np.maximum(out[:-k], b[k:])
    return out


def _runs(row: np.ndarray):
    p = np.r_[0, row.astype(np.int8), 0]
    d = np.diff(p)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _sweep(M: np.ndarray, runs, c: float, right: bool):
    """For x in an ink run: min over x' in the same run, x' < x (``right``) or x' > x, of M[x'] + c|x - x'|.
    Returns (value, argmin x')."""
    W = M.size
    val = np.full(W, np.inf); arg = np.full(W, -1)
    for a, b in runs:
        if b - a < 2:
            continue
        idx = np.arange(a, b)
        if not right:
            idx = idx[::-1]
        v = M[idx] - c * (idx if right else -idx)
        cm = np.minimum.accumulate(v)
        k = np.maximum.accumulate(np.where(v <= cm, np.arange(len(v)), 0))
        # strictly before: shift by one
        val[idx[1:]] = cm[:-1] + c * (idx[1:] if right else -idx[1:])
        arg[idx[1:]] = idx[k[:-1]]
    return val, arg


def _viterbi(B, Wm, prior, w_t, miss, max_step=3, step_c=0.05, side_c=0.01, w_wt=0.6,
             p_wt=0.5, p_scale=40.0, p_cap=4.0):
    n, W = B.shape
    xs = np.arange(W)
    D = np.zeros((3, W))
    bpx = np.zeros((n, 3, W), np.int16); bph = np.zeros((n, 3, W), np.int8)
    for i in range(n):
        e = np.where(B[i], w_wt * np.minimum(1.0, np.abs(Wm[i] - w_t[i]) / max(w_t[i], 1.0)), miss[i])
        if np.isfinite(prior[i]):
            e = e + p_wt * np.minimum(p_cap, np.abs(xs - prior[i]) / p_scale)
        ND = np.full((3, W), np.inf)
        runs = _runs(B[i])
        for hn in (L_, V_, R_):
            T = D + KINK[:, hn][:, None]
            hp = T.argmin(0); M = T[hp, xs]
            dxs = range(-max_step, 0) if hn == L_ else ([-1, 0, 1] if hn == V_ else range(1, max_step + 1))
            for dx in dxs:            # short steps, also across gaps (x = x' + dx)
                cand = np.full(W, np.inf); src = np.full(W, -1)
                if dx >= 0:
                    cand[dx:] = M[:W - dx] + step_c * abs(dx); src[dx:] = xs[:W - dx]
                else:
                    cand[:dx] = M[-dx:] + step_c * abs(dx); src[:dx] = xs[-dx:]
                better = cand < ND[hn]
                ND[hn][better] = cand[better]; bpx[i, hn][better] = src[better]; bph[i, hn][better] = hp[src[better]]
            if hn != V_:              # long slanted / sideways moves along the band's ink
                val, arg = _sweep(M, runs, side_c, right=(hn == R_))
                better = val < ND[hn]
                ND[hn][better] = val[better]; bpx[i, hn][better] = arg[better]; bph[i, hn][better] = hp[arg[better]]
            ND[hn] += e
        D = ND - np.min(ND)
    h, x = np.unravel_index(np.argmin(D), D.shape)
    out = np.zeros(n, np.int64); side = np.zeros(n, bool)
    for i in range(n - 1, -1, -1):
        out[i] = x
        px = int(bpx[i, h, x]); side[i] = abs(px - x) > max_step
        h, x = int(bph[i, h, x]), px
    return out, side


def trace_appearance(ink: np.ndarray, fy: int, prior: np.ndarray | None = None, block: int = 700,
                     passes: int = 2, **kw) -> dict:
    """Trace one curve through bool ``ink`` (rows x cols). ``prior``: x per sample (NaN = none), the
    curve identity (e.g. the raster tracer's pick). Returns dict(x, sideways, width, gap_frac)."""
    H, W = ink.shape
    n = -(-H // fy)
    B = band_ink(ink, fy, n, dilate=1)
    Wm = width_map(ink, fy, n)
    pr = np.full(n, np.nan) if prior is None else np.asarray(prior, float)[:n]
    if pr.size < n:
        pr = np.r_[pr, np.full(n - pr.size, np.nan)]
    # appearance learnt from the prior first, then from the previous pass per block
    ref = np.where(np.isfinite(pr), np.clip(np.nan_to_num(pr), 0, W - 1), -1).astype(int)
    ok = (ref >= 0)
    on = np.zeros(n, bool); on[ok] = B[np.nonzero(ok)[0], ref[ok]]
    w0 = float(np.median(Wm[np.nonzero(on)[0], ref[on]])) if on.any() else 2.0
    w_t = np.full(n, w0); miss = np.ones(n)
    x = side = None
    for _ in range(passes):
        x, side = _viterbi(B, Wm, pr, w_t, miss, **kw)
        hit = B[np.arange(n), x]
        ww = Wm[np.arange(n), x]
        for s in range(0, n, block):
            h = hit[s:s + block]
            if h.sum() >= 20:
                w_t[s:s + block] = np.median(ww[s:s + block][h])
            gap = 1 - h.mean() if h.size else 0
            miss[s:s + block] = float(np.clip(1.0 - 0.6 * gap, 0.4, 1.0))   # dashed: gaps are expected
    return dict(x=x.astype(float), sideways=side, width=w_t, gap_frac=1 - miss)


def apply_appearance(T: dict, ink: np.ndarray, **kw) -> dict:
    """Re-trace the picked curve of a raster trace ``T`` (in place) with :func:`trace_appearance`, its x
    as the prior; samples where the raster trace is off-track (wrap) keep the raster x and flags."""
    cands = [T]
    fy = int(T["fy"])
    for t in cands:
        tx = np.asarray(t["x"], float); n = len(tx)
        off = np.asarray(t["off_left"]) | np.asarray(t["off_right"])
        pr = tx.copy(); pr[off] = np.nan
        r = trace_appearance(ink, fy, pr, **kw)
        nx = tx.copy(); m = ~off
        nx[m] = r["x"][:n][m]
        t["x"] = nx
        fx, pad = t["fx"], t["pad"]
        pp = np.asarray(t["path_padded"]).copy()
        pp[m] = np.clip((nx[m] // fx).astype(int) + pad, pad, t["cost"].shape[1] - pad - 1)
        t["path_padded"] = pp; t["path"] = np.clip(pp - pad, 0, t["cost"].shape[1] - 2 * pad - 1)
        t["appearance_sideways"] = int(r["sideways"][:n][m].sum())
    return T


def _retrace(t: dict, ink: np.ndarray, **kw) -> dict:
    tx = np.asarray(t["x"], float); n = len(tx)
    off = np.asarray(t["off_left"]) | np.asarray(t["off_right"])
    pr = tx.copy(); pr[off] = np.nan
    r = trace_appearance(ink, int(t["fy"]), pr, **kw)
    nx = tx.copy(); m = ~off
    nx[m] = r["x"][:n][m]
    return dict(x=nx, m=m, sideways=r["sideways"][:n], width=r["width"][:n])


def _set_x(t: dict, nx: np.ndarray, m: np.ndarray):
    t["x"] = nx
    fx, pad = t["fx"], t["pad"]
    pp = np.asarray(t["path_padded"]).copy()
    pp[m] = np.clip((nx[m] // fx).astype(int) + pad, pad, t["cost"].shape[1] - pad - 1)
    t["path_padded"] = pp; t["path"] = np.clip(pp - pad, 0, t["cost"].shape[1] - 2 * pad - 1)


def stroke_band(x: np.ndarray, width: np.ndarray, fy: int, shape: tuple, grow: float = 1.0) -> np.ndarray:
    """Pixels covered by a traced stroke (each sample's ``fy`` rows, +-(width/2 + grow) around x,
    joined to the next sample so slanted strokes are covered too)."""
    H, W = shape
    out = np.zeros((H, W), bool)
    n = len(x)
    for i in range(n):
        if not np.isfinite(x[i]):
            continue
        h = float(width[i]) / 2 + grow
        a = x[i] if i + 1 >= n or not np.isfinite(x[i + 1]) else min(x[i], x[i + 1])
        b = x[i] if i + 1 >= n or not np.isfinite(x[i + 1]) else max(x[i], x[i + 1])
        out[i * fy:min(H, (i + 1) * fy), max(0, int(a - h)):min(W, int(np.ceil(b + h)) + 1)] = True
    return out


def apply_identity(T: dict, curve_ink: np.ndarray, cross_px: float | None = None, **kw) -> dict:
    """Joint identity re-trace of all candidates of a (multi-)curve trace ``T`` (in place) on
    ``curve_ink`` (e.g. scan ink the segmenter calls curve, grid and text removed). Candidates are
    traced in order (strongest first); each later candidate does not see the strokes already
    claimed, except within ``cross_px`` of them (crossings, where both curves share ink), so two
    curves cannot both follow one stroke and identities do not swap. Each candidate keeps its own
    appearance (stroke width, dash gaps, heading) from its raster prior; off-track (wrap) samples keep
    the raster x, so the backup-scale transition is handled by the raster wrap logic."""
    cands = T.get("candidates") or [T]
    fy = int(T["fy"]); H, W = curve_ink.shape
    claimed = np.zeros((H, W), bool); done = []
    for t in cands:
        tx = np.asarray(t["x"], float)
        ink = curve_ink
        if claimed.any():
            near = np.zeros((H, W), bool)          # samples where this curve's prior meets a claimed one
            for d in done:
                cp = float(cross_px if cross_px is not None else max(4.0, 1.5 * np.nanmedian(d["width"])))
                close = np.abs(tx - d["x"]) <= cp
                near |= np.repeat(close, fy)[:H, None] if len(close) * fy >= H else \
                    np.r_[np.repeat(close, fy), np.zeros(H - len(close) * fy, bool)][:, None]
            ink = curve_ink & (~claimed | near)
        r = _retrace(t, ink, **kw)
        _set_x(t, r["x"], r["m"])
        t["appearance_sideways"] = int(r["sideways"][r["m"]].sum())
        done.append(r)
        claimed |= stroke_band(r["x"], r["width"], fy, (H, W))
    return T
