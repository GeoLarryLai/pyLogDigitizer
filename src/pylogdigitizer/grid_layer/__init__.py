"""Grid as a background layer (idea A).

Fits the regular grid of a rectified track panel - vertical value lines (linear or log spacing),
horizontal depth lines, heavy lines - with a per-line, per-block width and darkness, renders the
predicted grid image and returns the residual: ink the grid does not explain. Ink on a grid line is
kept as curve evidence when it is wider than the line there (a stroke crossing it or running along
it). KGS scans are 1-bit, so "darkness" is the ink coverage of the line in the block (0..1); on a
grayscale panel (uint8, dark = ink) the same code uses the mean darkness.
"""
from __future__ import annotations

import numpy as np

__all__ = ["run_lengths", "fit_grid", "grid_residual"]


def run_lengths(ink: np.ndarray, axis: int = 1) -> np.ndarray:
    """Length of the ink run through each pixel along ``axis`` (0 = vertical, 1 = horizontal); 0 off ink."""
    a = np.asarray(ink, bool)
    if axis == 0:
        return run_lengths(a.T, 1).T
    H, W = a.shape
    out = np.zeros((H, W), np.int32)
    p = np.zeros((H, W + 2), np.int8); p[:, 1:-1] = a
    d = np.diff(p, axis=1)
    rs, cs = np.nonzero(d == 1); re, ce = np.nonzero(d == -1)     # same order: row-major
    L = ce - cs
    # paint lengths with a cumulative trick on a flat index
    flat = np.zeros(H * (W + 1), np.int64)
    s = rs * (W + 1) + cs; e = re * (W + 1) + ce
    np.add.at(flat, s, L); np.add.at(flat, e, -L)
    out = np.cumsum(flat).reshape(H, W + 1)[:, :W].astype(np.int32)
    out[~a] = 0
    return out


def _bands(mask1d: np.ndarray):
    p = np.r_[0, mask1d.astype(np.int8), 0]
    d = np.diff(p)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _spacing_kind(centers: np.ndarray) -> str:
    if len(centers) < 4:
        return "unknown"
    d = np.diff(centers).astype(float)
    cv = d.std() / max(d.mean(), 1e-9)
    if cv < 0.15:
        return "linear"
    r = d[1:] / np.maximum(d[:-1], 1e-9)
    return "log" if (np.mean(r < 1) > 0.7 or np.mean(r > 1) > 0.7) else "irregular"


def fit_grid(pan: np.ndarray, block: int = 2000, v_cover: float = 0.5, v_block: float = 0.6,
             h_cover: float = 0.5, min_vrun: int = 15) -> dict:
    """Fit the grid lattice of a rectified panel (bool ink, or uint8 gray with dark ink).

    Vertical lines: columns whose ink (in vertical runs >= ``min_vrun``) covers >= ``v_cover`` of
    the panel; per block a line is drawn only where it covers >= ``v_block`` of the block rows, with
    the width measured there. Horizontal lines: rows whose ink outside the vertical lines covers
    >= ``h_cover`` of the width; their spacing is regularised (missing lines under the curve are
    filled at the lattice period). Returns dict(vlines, hlines, h_period, v_kind, heavy_v, heavy_h)."""
    g = np.asarray(pan)
    dark = (g.astype(np.float32) if g.dtype == bool else (255 - g.astype(np.float32)) / 255.0)
    ink = dark > 0.5
    H, W = ink.shape
    vr = run_lengths(ink, 0) >= min_vrun
    cov = vr.mean(0)
    vlines = []
    for a, b in _bands(cov >= v_cover):
        lo, hi = max(0, a - 1), min(W, b + 1)
        blocks = []
        for r in range(0, H, block):
            sub = vr[r:r + block, lo:hi]
            cc = sub.mean(0)
            on = cc >= v_block
            if on.any():
                cols = np.nonzero(on)[0] + lo
                blocks.append(dict(r0=r, r1=min(H, r + block), c0=int(cols[0]), c1=int(cols[-1]) + 1,
                                   width=int(cols[-1] - cols[0] + 1),
                                   dark=float(dark[r:r + block, cols[0]:cols[-1] + 1].mean())))
        if blocks:
            vlines.append(dict(center=0.5 * (a + b - 1), width=float(np.median([k["width"] for k in blocks])),
                               blocks=blocks))
    if len(vlines) >= 4:     # lattice check: drop a "line" off the regular spacing (a straight curve stretch)
        c = np.array([v["center"] for v in vlines]); d = np.diff(c)
        per = float(np.median(d))
        if np.mean(np.abs(d - per) < 0.2 * per) >= 0.5:
            ph = np.median(np.mod(c, per))
            off = np.abs(((c - ph) / per) - np.round((c - ph) / per)) * per
            vlines = [v for v, o in zip(vlines, off) if o < 0.2 * per]
    vmask = np.zeros(W, bool)
    for v in vlines:
        vmask[int(np.floor(v["center"] - v["width"])):int(np.ceil(v["center"] + v["width"])) + 1] = True
    free = ~vmask
    hr = run_lengths(ink[:, free], 1)
    hcov = (hr >= max(8, free.sum() // 20)).mean(1) if free.any() else np.zeros(H)
    hl = [(a, b) for a, b in _bands(hcov >= h_cover)]
    hc = np.array([0.5 * (a + b - 1) for a, b in hl])
    period = float(np.median(np.diff(hc))) if len(hc) > 2 else 0.0
    hlines = [dict(center=float(c), width=int(b - a), dark=float(dark[a:b].mean()), filled=False)
              for c, (a, b) in zip(hc, hl)]
    if period > 4:          # fill lattice positions with no detected line (curve on top, faded)
        wmed = int(np.median([h["width"] for h in hlines]))
        have = hc
        for c in np.arange(hc[0], H, period):
            if np.min(np.abs(have - c)) > 0.3 * period:
                hlines.append(dict(center=float(c), width=wmed, dark=float("nan"), filled=True))
        for c in np.arange(hc[0] - period, 0, -period):
            if np.min(np.abs(have - c)) > 0.3 * period:
                hlines.append(dict(center=float(c), width=wmed, dark=float("nan"), filled=True))
        hlines.sort(key=lambda h: h["center"])
    vw = np.array([v["width"] for v in vlines]) if vlines else np.zeros(0)
    hw = np.array([h["width"] for h in hlines if not h["filled"]]) if hlines else np.zeros(0)
    return dict(vlines=vlines, hlines=hlines, h_period=period,
                v_kind=_spacing_kind(np.array([v["center"] for v in vlines])),
                heavy_v=[i for i, v in enumerate(vlines) if vw.size and v["width"] > np.median(vw) + 1],
                heavy_h=[i for i, h in enumerate(hlines) if not h["filled"] and hw.size and h["width"] > np.median(hw) + 1],
                shape=(H, W))


def grid_residual(pan: np.ndarray, G: dict | None = None, slack: int = 1, **kw) -> dict:
    """Predicted grid image and residual curve ink. A grid-line pixel stays in the residual when the
    ink run through it across the line is longer than the line width + 2*``slack`` (a stroke
    crossing), or along a vertical line when the row's run is wider than the line there (a curve
    riding on the line). Returns dict(grid, residual, G)."""
    g = np.asarray(pan)
    ink = g if g.dtype == bool else (g < 128)
    G = G or fit_grid(g, **kw)
    H, W = ink.shape
    grid = np.zeros((H, W), bool)
    keep = np.zeros((H, W), bool)
    hrun = run_lengths(ink, 1)
    hband = np.zeros(H, bool)
    for h in G["hlines"]:
        a = int(np.floor(h["center"] - h["width"] / 2 - slack)); b = int(np.ceil(h["center"] + h["width"] / 2 + slack)) + 1
        hband[max(0, a):max(0, min(H, b))] = True
    vband = np.zeros((H, W), bool)
    for v in G["vlines"]:
        for k in v["blocks"]:
            r0, r1, c0, c1 = k["r0"], k["r1"], k["c0"], k["c1"]
            a, b = max(0, c0 - slack), min(W, c1 + slack)
            sub = ink[r0:r1, a:b]
            grid[r0:r1, a:b] |= sub; vband[r0:r1, a:b] = True
            keep[r0:r1, a:b] |= sub & (hrun[r0:r1, a:b] > (c1 - c0) + 2 * slack) & ~hband[r0:r1, None]
    vrun = None
    for h in G["hlines"]:
        a = int(np.floor(h["center"] - h["width"] / 2 - slack)); b = int(np.ceil(h["center"] + h["width"] / 2 + slack)) + 1
        a, b = max(0, a), min(H, b)
        if b <= a:
            continue
        if vrun is None:
            vrun = run_lengths(ink, 0)
        sub = ink[a:b]
        grid[a:b] |= sub
        keep[a:b] |= sub & (vrun[a:b] > (b - a)) & ~vband[a:b]
    grid &= ~keep
    return dict(grid=grid, residual=ink & ~grid, G=G)
