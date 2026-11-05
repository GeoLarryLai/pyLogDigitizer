"""Synthetic log-scan tiles with exact per-pixel labels (idea B).

Draws real KGS LAS curves in legend styles (solid / dashed / dotted, 1-5 px) with crossings and
backup wraps onto synthetic grids (linear or log value lines, heavy lines, variable thickness),
then degrades the sheet: ink bleed, fading, blur, skew, stamps / text, paper texture and an optional
1-bit threshold like the KGS TIFFs. Labels: 0 paper, 1 grid, 2 curve A, 3 curve B, 4 text; plus the
curve direction (radians, NaN off curve). numpy + OpenCV only.
"""
from __future__ import annotations

import numpy as np

PAPER, GRID, CURVE_A, CURVE_B, TEXT = 0, 1, 2, 3, 4


def zigzag_score(v: np.ndarray) -> float:
    """Fraction of samples where the slope changes sign - high for the saw-tooth reference curves
    some KGS LAS files carry (digitised with interpolation artefacts)."""
    d = np.diff(np.asarray(v, float))
    d = d[d != 0]
    return float(np.mean(np.sign(d[1:]) != np.sign(d[:-1]))) if d.size > 2 else 1.0


def load_curves(las_paths, min_len: int = 400, max_zigzag: float = 0.45, smooth: int = 3) -> list[dict]:
    """Clean curves from LAS files: continuous runs of >= ``min_len`` samples; curves with a zig-zag
    score above ``max_zigzag`` are dropped, the rest get a light moving average."""
    import lasio
    out = []
    for p in las_paths:
        try:
            las = lasio.read(str(p))
        except Exception:
            continue
        dep = np.asarray(las.index, float)
        for c in las.curves[1:]:
            v = np.asarray(c.data, float)
            ok = np.isfinite(v) & (v > -900)
            if ok.sum() < min_len:
                continue
            p_ = np.r_[0, ok.astype(np.int8), 0]; dd = np.diff(p_)
            for a, b in zip(np.nonzero(dd == 1)[0], np.nonzero(dd == -1)[0]):
                if b - a < min_len:
                    continue
                seg = v[a:b]
                z = zigzag_score(seg)
                if z > max_zigzag or np.ptp(seg) == 0:
                    continue
                if smooth > 1:
                    seg = np.convolve(np.pad(seg, smooth // 2, mode="edge"), np.ones(smooth) / smooth, "valid")
                out.append(dict(name=f"{p.parent.name if hasattr(p, 'parent') else p}:{c.mnemonic}", depth=dep[a:b],
                                value=seg, zigzag=z))
    return out


def _grid(rng, H, W):
    g = np.zeros((H, W), np.float32)
    log = rng.random() < 0.3
    if log:
        cyc = int(rng.integers(2, 5)); xs = []
        for k in range(cyc):
            xs += [W * (k + np.log10(m)) / cyc for m in range(1, 10)]
        heavy = [W * k / cyc for k in range(cyc + 1)]
    else:
        nd = 10
        xs = [W * k / nd for k in range(nd + 1)]; heavy = xs[::5]
    wbase = rng.uniform(1, 3)
    for x in xs:
        w = max(1, int(round(wbase * rng.uniform(0.7, 1.4) + (2 if any(abs(x - h) < 1 for h in heavy) else 0))))
        xi = int(round(x)); g[:, max(0, xi - w // 2):min(W, xi - w // 2 + w)] = rng.uniform(0.6, 1.0)
    per = rng.uniform(18, 45); y0 = rng.uniform(0, per); k = 0
    for y in np.arange(y0, H, per):
        w = max(1, int(round(wbase * rng.uniform(0.7, 1.4) + (2 if k % 5 == 0 else 0)))); k += 1
        yi = int(y); g[yi:yi + w] = np.maximum(g[yi:yi + w], rng.uniform(0.6, 1.0))
    return g


def _curve(rng, c, H, W, px_per_ft, style, width, scale):
    i0 = int(rng.integers(0, max(1, len(c["value"]) - 10)))
    dep = c["depth"][i0:] - c["depth"][i0]; v = c["value"][i0:]
    m = dep * px_per_ft < H + 10
    y = dep[m] * px_per_ft; v = v[m]
    lo, hi = scale
    x = (v - lo) / (hi - lo) * W
    x = np.where(x > W, x - W, x)          # backup wrap: re-enters from the left on the backup scale
    x = np.where(x < 0, x + W, x)
    layer = np.zeros((H, W), np.uint8); dirm = np.full((H, W), np.nan, np.float32)
    pts = np.c_[x, y]
    seg_len = np.r_[0, np.hypot(*np.diff(pts, axis=0).T)]; arc = np.cumsum(seg_len)
    dash, gap = (1e9, 0) if style == "solid" else ((rng.uniform(8, 20), rng.uniform(4, 10)) if style == "dashed"
                                                    else (rng.uniform(1.5, 3), rng.uniform(3, 6)))
    for k in range(len(pts) - 1):
        if abs(pts[k + 1, 0] - pts[k, 0]) > W / 2:      # wrap jump: no stroke across the track
            continue
        if style != "solid" and (arc[k] % (dash + gap)) > dash:
            continue
        p0 = tuple(np.round(pts[k]).astype(int)); p1 = tuple(np.round(pts[k + 1]).astype(int))
        tmp = np.zeros((H, W), np.uint8)
        import cv2
        cv2.line(tmp, p0, p1, 1, int(width), lineType=cv2.LINE_8)
        layer |= tmp
        dirm[tmp > 0] = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    return layer.astype(bool), dirm


def render_tile(rng: np.random.Generator, curves: list[dict], H: int = 512, W: int = 512,
                binary: bool | None = None) -> dict:
    """One synthetic tile. Returns dict(image uint8 (255 paper), label uint8, direction float32, meta)."""
    import cv2
    grid = _grid(rng, H, W)
    ppf = rng.uniform(4, 12)
    styles = ["solid", "dashed", "dotted"]
    lab = np.zeros((H, W), np.uint8); lab[grid > 0] = GRID
    dirm = np.full((H, W), np.nan, np.float32)
    dark = grid * rng.uniform(0.5, 1.0)
    meta = dict(px_per_ft=ppf, curves=[])
    ncur = 2 if rng.random() < 0.7 else 1
    for k in range(ncur):
        c = curves[int(rng.integers(len(curves)))]
        lo, hi = np.percentile(c["value"], [2, rng.uniform(80, 99)])
        st = styles[0] if k == 0 and rng.random() < 0.7 else styles[int(rng.integers(3))]
        w = int(rng.integers(1, 6))
        m, d = _curve(rng, c, H, W, ppf, st, w, (lo, hi))
        lab[m] = CURVE_A + k; dirm[m] = d[m]
        dark = np.maximum(dark, m * rng.uniform(0.75, 1.0))
        meta["curves"].append(dict(name=c["name"], style=st, width=w))
    tmask = np.zeros((H, W), np.uint8)
    for _ in range(int(rng.integers(0, 4))):        # stamps / text / annotations
        txt = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .-"), int(rng.integers(2, 9))))
        org = (int(rng.integers(0, W - 40)), int(rng.integers(20, H)))
        cv2.putText(tmask, txt, org, int(rng.integers(0, 7)), rng.uniform(0.4, 1.4), 1, int(rng.integers(1, 3)))
    if rng.random() < 0.2:
        cv2.circle(tmask, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(20, 60)), 1, 2)
    lab[(tmask > 0) & (lab == PAPER)] = TEXT
    dark = np.maximum(dark, (tmask > 0) * rng.uniform(0.6, 1.0))
    if rng.random() < 0.6:                          # ink bleed
        k = int(rng.integers(2, 4)); dark = cv2.dilate(dark, np.ones((k, k), np.uint8)) * 0.5 + dark * 0.5
    fade = cv2.resize(rng.uniform(0.55, 1.0, (4, 4)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    dark = dark * np.clip(fade, 0.4, 1.0)
    dark = cv2.GaussianBlur(dark, (0, 0), rng.uniform(0.3, 1.2))
    paper = cv2.resize(rng.normal(0, 0.04, (H // 16, W // 16)).astype(np.float32), (W, H)) + rng.normal(0, 0.03, (H, W))
    img = np.clip(1 - dark + paper - 0.05, 0, 1)
    ang = rng.uniform(-1.5, 1.5)                    # skew
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    img = cv2.warpAffine(img.astype(np.float32), M, (W, H), borderValue=1.0)
    lab = cv2.warpAffine(lab, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    dirm = cv2.warpAffine(dirm, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=np.nan)
    dirm = np.where(np.isfinite(dirm), dirm - np.deg2rad(ang), np.nan).astype(np.float32)
    if binary if binary is not None else rng.random() < 0.7:
        img = (img > rng.uniform(0.45, 0.65)).astype(np.float32)
    meta.update(skew_deg=ang)
    return dict(image=(img * 255).astype(np.uint8), label=lab, direction=dirm, meta=meta)
