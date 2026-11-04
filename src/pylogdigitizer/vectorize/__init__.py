"""Vector-first separation: skeletonize all ink, split the skeleton into segments at junctions,
classify each segment as grid / curve / other, and rebuild a curve-only panel from the curve
segments. Grid segments are dropped as whole vectors; no curve pixel is erased by a grid filter
(pixels at crossings belong to the junction and are kept)."""
from __future__ import annotations

import numpy as np

GRID, CURVE, OTHER = 1, 2, 3


def skeletonize(ink: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen thinning (numpy) -> 1-px 8-connected centrelines."""
    im = np.pad(np.asarray(ink, bool), 1).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = im
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(np.int16) for n in nb)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (P[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                im[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return im[1:-1, 1:-1].astype(bool)


def _neighbours(sk: np.ndarray) -> np.ndarray:
    import cv2
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    return np.rint(cv2.filter2D(sk.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)).astype(np.int16)


def vectorize(ink: np.ndarray, grid: np.ndarray | None = None, min_len: int = 3) -> dict:
    """Ink -> skeleton segments. Returns dict(skel, junction, seg (int32 label map on the skeleton,
    0 = none), segs (list of per-segment feature dicts), width (local stroke width map))."""
    import cv2
    ink = np.asarray(ink, bool)
    sk = skeletonize(ink)
    nb = _neighbours(sk)
    junc = sk & (nb >= 3)
    junc_d = cv2.dilate(junc.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & sk
    body = sk & ~junc_d
    n, lab, st, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8), connectivity=8)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    order = np.argsort(ids, kind="stable")
    ys, xs, ids = ys[order], xs[order], ids[order]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    segs = [None] * n
    for yy, xx in zip(np.split(ys, cuts), np.split(xs, cuts)):
        i = int(lab[yy[0], xx[0]])
        L = len(yy)
        h = int(yy.max() - yy.min() + 1); w = int(xx.max() - xx.min() + 1)
        if L >= 3:
            cy, cx = yy - yy.mean(), xx - xx.mean()
            cov = np.array([[np.mean(cy * cy), np.mean(cy * cx)], [np.mean(cy * cx), np.mean(cx * cx)]])
            ev, evec = np.linalg.eigh(cov)
            straight = float(1.0 - np.sqrt(max(ev[0], 0) / max(ev[1], 1e-9)))
            d = evec[:, 1]
            ang = float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))   # 0 = vertical, 90 = horizontal
        else:
            straight, ang = 0.0, 45.0
        wd = 2.0 * dist[yy, xx]
        on_grid = float(grid[yy, xx].mean()) if grid is not None else 0.0
        segs[i] = dict(id=i, n=L, y0=int(yy.min()), y1=int(yy.max()), x0=int(xx.min()), x1=int(xx.max()),
                       h=h, w=w, straight=straight, angle=ang, width=float(np.median(wd)),
                       xmean=float(xx.mean()), on_grid=on_grid)
    return dict(skel=sk, junction=junc_d, seg=lab.astype(np.int32), segs=segs, dist=dist)


def grid_columns(grid: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Columns carrying a detected vertical grid line (fraction of rows flagged as grid)."""
    g = np.asarray(grid, bool)
    return g.mean(axis=0) >= frac


def classify_segments(V: dict, ink: np.ndarray, grid: np.ndarray, bands=(), other: np.ndarray | None = None,
                      long_v: int = 40, long_h: int | None = None, max_grid_width: float = 4.5,
                      col_tol: int = 2, col_cover: float = 0.3) -> np.ndarray:
    """Class per segment id (index = id; 0 unused). Grid: straight near-vertical/horizontal segments
    that are long, or that lie on a detected grid column / depth-line band; text blobs: OTHER;
    the rest: CURVE."""
    import cv2
    W = ink.shape[1]
    long_h = long_h or max(15, int(0.25 * W))
    cols = grid_columns(grid)
    colsd = cv2.dilate(cols.astype(np.uint8)[None], np.ones((1, 2 * col_tol + 1), np.uint8))[0].astype(bool)
    brow = np.zeros(ink.shape[0], bool)
    for a, e in bands:
        brow[max(0, a - 1):e + 1] = True
    # persistent vertical lines of any width (thin grid columns and thick reference lines): columns
    # where straight near-vertical segments cover >= ``col_cover`` of the panel height
    H = ink.shape[0]
    vcov = np.zeros(W, float)
    for s in V["segs"]:
        if s is not None and s["angle"] < 6 and s["straight"] > 0.9 and s["h"] >= 6 and s["w"] <= 3 + s["width"]:
            vcov[int(round(s["xmean"]))] += s["h"]
    vk = max(3, 2 * int(max_grid_width) + 1)
    vcov = np.convolve(vcov, np.ones(vk), mode="same")
    vcols = cv2.dilate((vcov >= col_cover * H).astype(np.uint8)[None], np.ones((1, vk), np.uint8))[0].astype(bool)
    cls = np.zeros(len(V["segs"]), np.int8)
    for s in V["segs"]:
        if s is None:
            continue
        i = s["id"]
        vert = s["angle"] < 8 and s["straight"] > 0.9
        hor = s["angle"] > 82 and s["straight"] > 0.9
        thin = s["width"] <= max_grid_width
        c = CURVE
        if vert and vcols[s["x0"]:s["x1"] + 1].all() and s["h"] >= 4:
            c = GRID      # piece of a persistent vertical line (thick reference lines too)
        elif vert and thin and (s["h"] >= long_v or (colsd[s["x0"]:s["x1"] + 1].all() and s["on_grid"] > 0.5)):
            c = GRID
        elif hor and (s["w"] >= long_h or (brow[s["y0"]:s["y1"] + 1].all() and s["w"] >= 6)):
            c = GRID
        elif s["on_grid"] > 0.8 and s["n"] >= 6 and (vert or hor):
            c = GRID      # short dash of a dashed grid line
        elif s["n"] < 3 and s["on_grid"] > 0.5:
            c = GRID
        if other is not None and c == CURVE:
            yy, xx = np.nonzero(V["seg"][s["y0"]:s["y1"] + 1, s["x0"]:s["x1"] + 1] == i)
            if len(yy) and other[yy + s["y0"], xx + s["x0"]].mean() > 0.6:
                c = OTHER
        cls[i] = c
    return cls


def style_groups(V: dict, cls: np.ndarray, n_curves: int) -> np.ndarray:
    """Group curve segments into ``n_curves`` stroke-width groups (length-weighted 1-D k-means);
    returns group per segment id (-1 = not a curve segment). Group 0 = thickest."""
    g = -np.ones(len(cls), np.int16)
    ids = [s["id"] for s in V["segs"] if s is not None and cls[s["id"]] == CURVE]
    if not ids:
        return g
    if n_curves <= 1:
        g[ids] = 0
        return g
    w = np.array([V["segs"][i]["width"] for i in ids]); n = np.array([V["segs"][i]["n"] for i in ids], float)
    c = np.quantile(w, np.linspace(0.85, 0.15, n_curves))
    for _ in range(30):
        a = np.argmin(np.abs(w[:, None] - c[None]), axis=1)
        for k in range(n_curves):
            if (a == k).any():
                c[k] = np.average(w[a == k], weights=n[a == k])
    order = np.argsort(-c); rank = np.empty_like(order); rank[order] = np.arange(n_curves)
    g[ids] = rank[a]
    return g


def curve_panel(ink: np.ndarray, V: dict, cls: np.ndarray, keep=(CURVE,)) -> np.ndarray:
    """Ink pixels whose nearest skeleton pixel is a kept segment or a junction -> curve-only panel."""
    import cv2
    ink = np.asarray(ink, bool)
    sk = V["skel"]
    keepseg = np.isin(cls, list(keep))
    good = np.zeros(ink.shape, bool)
    s = V["seg"]
    good[s > 0] = keepseg[s[s > 0]]
    # junction clusters: keep a whole cluster when it touches a kept segment (a curve crossing a
    # grid line, or the spur knots of a thick stroke)
    j = V["junction"]
    nj, jl = cv2.connectedComponents(j.astype(np.uint8), connectivity=8)
    near = cv2.dilate(good.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    hit = np.zeros(nj, bool); hit[np.unique(jl[j & near])] = True; hit[0] = False
    good |= hit[jl]
    # spread skeleton labels to the stroke pixels (nearest skeleton pixel)
    _, lab = cv2.distanceTransformWithLabels((~sk).astype(np.uint8), cv2.DIST_L2, 3,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    ky = np.zeros(lab.max() + 1, bool)
    sy, sx = np.nonzero(sk)
    ky[lab[sy, sx]] = good[sy, sx]
    return ink & ky[lab]


def vector_separate(ink: np.ndarray, n_curves: int = 1) -> dict:
    """Full vector-first pipeline on one track panel. Returns dict(curve_ink, V, cls, groups, grid)."""
    from pylogdigitizer import separation as sep
    ink = np.asarray(ink, bool)
    grid = sep.grid_mask(ink)
    bands = sep.hline_bands(ink)
    try:
        oth = sep.other_mask(ink, grid)
    except Exception:
        oth = None
    V = vectorize(ink, grid)
    cls = classify_segments(V, ink, grid, bands, oth)
    groups = style_groups(V, cls, n_curves)
    return dict(curve_ink=curve_panel(ink, V, cls), V=V, cls=cls, groups=groups, grid=grid, bands=bands)
