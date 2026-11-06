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


def find_seeds(curve_ink: np.ndarray, fy: int, n: int, start: int = 0, window: int = 80, min_sep: float = 20.0,
               edge: int | None = None):
    """Independent seeds for ``n`` curves: the first sample (from ``start``, within ``window`` samples)
    whose band shows >= ``n`` separate curve-ink runs at least ``min_sep`` px apart and away from the
    track edges (``edge`` px, default 2 % of the width: border-hugging ink is clipping, not a seed);
    returns (sample, [(x, width), ...] widest first) or None."""
    H, W = curve_ink.shape
    edge = int(edge if edge is not None else max(6, round(0.02 * W)))
    s0 = start * fy
    B = band_ink(curve_ink[s0:s0 + window * fy], fy, dilate=0)
    Wm = width_map(curve_ink[s0:s0 + window * fy], fy, dilate=0)
    for i in range(B.shape[0]):
        rs = [(a, b) for a, b in _runs(B[i]) if b - a >= 2 and a >= edge and b <= W - edge]
        if len(rs) < n:
            continue
        c = [((a + b - 1) / 2, float(Wm[i, a:b].max())) for a, b in rs]
        c.sort(key=lambda t: -t[1])
        pick = []
        for x, w in c:
            if all(abs(x - p[0]) >= min_sep for p in pick):
                pick.append((x, w))
            if len(pick) == n:
                return start + i, pick
    return None


def _stretches(m: np.ndarray, min_len: int):
    p = np.r_[0, m.astype(np.int8), 0]; d = np.diff(p)
    return [(a, b) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]) if b - a >= min_len]


def _rows_mask(m: np.ndarray, fy: int, H: int) -> np.ndarray:
    r = np.repeat(m, fy)[:H]
    return np.r_[r, np.zeros(H - r.size, bool)] if r.size < H else r


def seeded_multi_trace(curve_ink: np.ndarray, fy: int, n: int = 2, seed_len: int = 6,
                       cross_px: float | None = None, cross_len: int = 15, collapse_px: float = 6.0,
                       collapse_len: int = 40, class_inks: list | None = None, **kw):
    """Trace ``n`` curves on curve ink from independent seeds (not from another tracer's picks).
    Curves are traced widest first. A later curve may use ink already claimed by an earlier one only
    inside short windows (<= ``cross_len`` samples) where its own first pass comes within
    ``cross_px`` of that curve (a true crossing); longer shared stretches are excluded. Where a later
    curve still collapses onto an earlier one (within ``collapse_px`` for >= ``collapse_len``
    samples) or loses ink that long, it is re-seeded from free ink at the start of that stretch and
    re-traced there. With ``class_inks`` (one ink mask per curve class from the identity segmenter)
    each curve is seeded and traced on its own class ink, in the given order. Returns a list of dict(x, width, gap_frac, seed_x, seed_sample, reseeds)."""
    H, W = curve_ink.shape
    ns = -(-H // fy)
    if class_inks is not None:        # per-curve ink from a curve-identity segmenter: one seed per class
        n = len(class_inks); seeds = []
        for ci in class_inks:
            sd1 = find_seeds(ci, fy, 1, window=ns)
            if sd1 is None:
                return []
            seeds.append((sd1[0], sd1[1][0]))
    else:
        sd = find_seeds(curve_ink, fy, n)
        if sd is None:
            return []
        seeds = [(sd[0], p_) for p_ in sd[1]]
    claimed = np.zeros((H, W), bool); out = []
    for k_, (s, (x0, w0)) in enumerate(seeds):
        if class_inks is not None:
            curve_ink = class_inks[k_]
        B = band_ink(curve_ink, fy, ns, dilate=1)
        pr = np.full(ns, np.nan); pr[s:s + seed_len] = x0
        reseeds = 0
        if not out:
            r = trace_appearance(curve_ink, fy, pr, **kw)
        else:
            r0 = trace_appearance(curve_ink & ~claimed, fy, pr, **kw)
            near = np.zeros(ns, bool)
            for d in out:
                cp = float(cross_px if cross_px is not None else max(4.0, 1.5 * np.nanmedian(d["width"])))
                near |= np.abs(r0["x"] - d["x"]) <= cp
            win = np.zeros(ns, bool)
            for a, b in _stretches(near, 1):
                if b - a <= cross_len:
                    win[a:b] = True
            pr2 = r0["x"].copy(); pr2[win] = np.nan; pr2[:s] = np.nan; pr2[s:s + seed_len] = x0
            allowed = curve_ink & (~claimed | _rows_mask(win, fy, H)[:, None])
            r = trace_appearance(allowed, fy, pr2, **kw)
            # re-seed collapsed / ink-less stretches from free ink
            bad = np.zeros(ns, bool)
            for d in out:
                bad |= np.abs(r["x"] - d["x"]) < collapse_px
            on = B[np.arange(ns), np.clip(r["x"].astype(int), 0, W - 1)]
            bad |= ~on
            st = _stretches(bad, collapse_len)
            if st:
                pr3 = r["x"].copy()
                for a, b in st:
                    pr3[a:b] = np.nan
                    sd2 = find_seeds(curve_ink & ~claimed, fy, 1, start=a, window=max(1, b - a))
                    if sd2 is not None:
                        k = sd2[0]; pr3[k:min(b, k + seed_len)] = sd2[1][0][0]; reseeds += 1
                if reseeds:
                    r = trace_appearance(allowed, fy, pr3, **kw)
        x = r["x"].astype(float)
        out.append(dict(x=x, width=r["width"], gap_frac=float(np.mean(r["gap_frac"])), seed_x=x0, seed_sample=s,
                        reseeds=reseeds))
        claimed |= stroke_band(x, r["width"], fy, (H, W))
    return out


def role_by_style(seeded: list, want: str) -> int:
    """Index of the seeded trace matching a line style when no legend is readable: ``want`` =
    "dashed" (largest gap share) or "solid" (smallest); "thick" / "thin" by median stroke width; "left" / "right" by median position."""
    if want in ("dashed", "solid"):
        g = [t["gap_frac"] for t in seeded]
        return int(np.argmax(g) if want == "dashed" else np.argmin(g))
    if want in ("left", "right"):
        xs = [float(np.nanmedian(t["x"])) for t in seeded]
        return int(np.argmin(xs) if want == "left" else np.argmax(xs))
    w = [float(np.median(t["width"])) for t in seeded]
    return int(np.argmax(w) if want == "thick" else np.argmin(w))


def match_to(cands: list, seeded: list) -> list:
    """Order seeded traces to the candidates they overlap most (labels only: which seeded trace
    plays which candidate role); returns seeded list reordered to len(cands)."""
    if len(seeded) < 2 or len(cands) < 2:
        return seeded[:len(cands)]
    d = lambda c, s: float(np.nanmedian(np.abs(np.asarray(c["x"], float) - s["x"][:len(c["x"])])))
    keep = d(cands[0], seeded[0]) + d(cands[1], seeded[1]); swap = d(cands[0], seeded[1]) + d(cands[1], seeded[0])
    return seeded if keep <= swap else [seeded[1], seeded[0]] + seeded[2:]


def wrap_backup(curve_ink: np.ndarray, x: np.ndarray, fy: int, width: float, side: str = "right",
                edge: int | None = None, near: float = 0.85, min_len: int = 3, **kw):
    """Explicit wrap transition: samples where the traced curve is near the ``side`` edge and the
    curve ink in its band runs into that edge (the curve leaves the track) start a wrap; during the
    wrap the backup curve is traced entering from the opposite edge (prior pinned there), and the
    wrap ends when that ink no longer touches the edge. Returns [(a, b, xb)] with xb the backup x
    for samples a..b-1 (value it on the backup scale)."""
    H, W = curve_ink.shape
    n = len(x); wi = int(round(width))
    edge = int(edge if edge is not None else max(6, round(0.02 * wi)))   # the edge zone (border line is grid, not curve)
    B = band_ink(curve_ink, fy, n, dilate=0)
    if side == "right":
        touch = B[:, max(0, wi - edge):wi].any(1); enter = B[:, 0:edge].any(1)
        close = np.asarray(x, float) >= near * width
    else:
        touch = B[:, 0:edge].any(1); enter = B[:, max(0, wi - edge):wi].any(1)
        close = np.asarray(x, float) <= (1 - near) * width
    # main-side contact must be attached to the traced stroke: same run from trace to the edge
    att = np.zeros(n, bool); runs_att = {}
    for i in np.nonzero(touch & close)[0]:
        xi = int(np.clip(x[i], 0, W - 1))
        for a_, b_ in _runs(B[i, :wi]):          # the run under (or within 4 px of) the trace reaches the edge
            if a_ - 4 <= xi < b_ + 4 and ((side == "right" and b_ >= wi - edge) or (side != "right" and a_ < edge)):
                att[i] = True; runs_att[i] = (a_, b_)
    on = att.copy()
    # extend each contact while backup ink still enters from the opposite edge
    i = 0; spans = []
    while i < n:
        if not on[i]:
            i += 1; continue
        a = i
        while i < n and (on[i] or enter[i]):
            i += 1
        if i - a >= min_len and enter[a:i].mean() > 0.3:
            spans.append((a, i))
    out = []
    for a, b in spans:          # backup trace over the full track width, pinned at the entry edge only
        sub = curve_ink[a * fy:b * fy, :wi].copy()
        for i in range(a, b):                 # the main stroke's own run into the edge is not the backup
            if i in runs_att:
                a_, b_ = runs_att[i]
                sub[(i - a) * fy:(i - a + 1) * fy, max(0, a_ - 2):min(wi, b_ + 2)] = False
        x0 = 0.0 if side == "right" else wi - 1.0
        pr = np.full(b - a, np.nan); pr[:2] = x0
        r = trace_appearance(sub, fy, pr, passes=1, **kw)
        xb = r["x"][:b - a]
        Bs = band_ink(sub, fy, b - a, dilate=1)
        xi = np.clip(xb.astype(int), 2, wi - 3)
        on_ink = float(np.mean([Bs[k, xi[k] - 2:xi[k] + 3].any() for k in range(b - a)]))
        if on_ink < 0.6 or abs(xb[0] - x0) > 3 * edge:   # no backup stroke really entered: false wrap
            continue
        out.append((a, b, xb))
    return out


def spike_excursions(curve_ink: np.ndarray, x: np.ndarray, fy: int, width: int, side: str = "right",
                     edge: int | None = None, min_reach: float = 0.3, base_tol: float = 15.0, pad: int = 1) -> dict:
    """Fast off-scale spikes drawn as near-horizontal strokes (too quick for the row-wise tracer,
    which bridges them with a straight connector). At sample i, a curve-ink run that starts within
    ``base_tol`` px of the trace and runs into the ``side`` edge means the curve left the main scale.
    If a separate run enters from the opposite edge in the same rows (the backup curve), the spike
    peak is that run's far end: returns {i: fraction of width on the backup scale}. With no backup
    run the curve is clipped: {i: nan}. Bands use +-``pad`` samples so short strokes are caught."""
    H, W = curve_ink.shape; n = len(x)
    wi = min(int(width), W)
    edge = int(edge if edge is not None else max(6, round(0.02 * wi)))
    out = {}
    ez = curve_ink[:, wi - edge:wi].any(1) if side == "right" else curve_ink[:, :edge + 1].any(1)
    Be = band_ink(ez[:, None], fy, n, dilate=pad)[:, 0]          # only bands touching the wrap edge can spike
    for i in np.flatnonzero(Be[:n]):
        if not np.isfinite(x[i]):
            continue
        r0, r1 = max(0, (i - pad) * fy), min(H, (i + 1 + pad) * fy)
        if r0 >= r1:
            continue
        rs = [r_ for rr in curve_ink[r0:r1, :wi] for r_ in _runs(rr)]     # per pixel row: strokes stay apart
        if side == "right":
            att = [(a, b) for a, b in rs if b >= wi - edge and a > edge and a <= x[i] + base_tol and b - a >= min_reach * wi]
            bk = [(a, b) for a, b in rs if a <= edge and b < wi - edge and b - a >= 3]
            if att:
                out[i] = (max(b for a, b in bk) - 1) / (wi - 1) if bk else np.nan
        else:
            att = [(a, b) for a, b in rs if a <= edge and b < wi - edge and b >= x[i] - base_tol and b - a >= min_reach * wi]
            bk = [(a, b) for a, b in rs if b >= wi - edge and a > edge and b - a >= 3]
            if att:
                out[i] = min(a for a, b in bk) / (wi - 1) if bk else np.nan
    return out
