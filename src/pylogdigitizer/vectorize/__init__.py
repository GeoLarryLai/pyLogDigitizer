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


def _crossings(sk: np.ndarray) -> np.ndarray:
    """Crossing number: 0->1 transitions around each pixel's 8-ring (3+ = branch point)."""
    P = np.pad(sk.astype(np.int8), 1)
    ring = [P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:], P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]]
    # 4-neighbours bridging a diagonal count as one arm
    A = np.zeros(sk.shape, np.int16)
    for i in range(8):
        A += ((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.int16)
    return A


def vectorize(ink: np.ndarray, grid: np.ndarray | None = None, min_len: int = 3) -> dict:
    """Ink -> skeleton segments. Returns dict(skel, junction, seg (int32 label map on the skeleton,
    0 = none), segs (list of per-segment feature dicts), width (local stroke width map))."""
    import cv2
    ink = np.asarray(ink, bool)
    sk = skeletonize(ink)
    junc = sk & (_crossings(sk) >= 3)      # branch points (not staircase corners)
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


def _free_end(V: dict, s: dict, W: int, r: int = 3, edge: int = 4) -> bool:
    """True when an end of the horizontal segment ``s`` is a free stroke end (no junction or other
    skeleton pixel within ``r`` px) away from the track edges."""
    sk, j, lab = V["skel"], V["junction"], V["seg"]
    for xe in (s["x0"], s["x1"]):
        if xe < edge or xe > W - 1 - edge:
            continue
        y0, y1 = max(0, s["y0"] - r), s["y1"] + r + 1
        x0, x1 = max(0, xe - r), xe + r + 1
        win_l = lab[y0:y1, x0:x1]
        if not j[y0:y1, x0:x1].any() and not ((win_l > 0) & (win_l != s["id"])).any():
            return True
    return False


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
        elif hor and s["w"] < 0.9 * W and _free_end(V, s, W):
            c = CURVE     # horizontal stroke ending in paper inside the track: a spike, not a grid row
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


# ---------------------------------------------------------------- segment linking (v2)

def _seg_pixels(V: dict) -> list:
    """Per segment id: (ys, xs) skeleton pixels, cached in V."""
    if "pix" in V:
        return V["pix"]
    lab = V["seg"]
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    o = np.argsort(ids, kind="stable"); ys, xs, ids = ys[o], xs[o], ids[o]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    pix = [None] * len(V["segs"])
    for yy, xx in zip(np.split(ys, cuts), np.split(xs, cuts)):
        if len(yy):
            pix[int(lab[yy[0], xx[0]])] = (yy, xx)
    V["pix"] = pix
    return pix


def endpoints(V: dict, tan_len: int = 10) -> list:
    """Two ends per segment: dict(seg, end (0/1), y, x, t (unit outward tangent, (dy, dx)), width).
    Ends are the extreme pixels along the segment's main axis; the tangent is fitted on the
    ``tan_len`` pixels nearest each end."""
    pix = _seg_pixels(V)
    out = []
    for s in V["segs"]:
        if s is None or pix[s["id"]] is None:
            continue
        yy, xx = pix[s["id"]]
        p = np.stack([yy, xx], 1).astype(float)
        c = p.mean(0)
        if len(p) >= 2:
            u = np.linalg.svd(p - c, full_matrices=False)[2][0]
        else:
            u = np.array([1.0, 0.0])
        if u[0] < 0:
            u = -u                                   # main axis points down
        proj = (p - c) @ u
        o = np.argsort(proj)
        for end, idx in ((0, o[:tan_len]), (1, o[::-1][:tan_len])):
            q = p[idx]
            tip = q[0]
            if len(q) >= 3:
                v = tip - q[1:].mean(0)
            else:
                v = -u if end == 0 else u
            n = np.hypot(*v)
            v = v / n if n > 0 else (-u if end == 0 else u)
            out.append(dict(seg=s["id"], end=end, y=float(tip[0]), x=float(tip[1]), t=v, width=s["width"]))
    return out


def link_segments(V: dict, eligible: np.ndarray, reach: float = 22.0, max_cost: float = 1.4) -> list:
    """Mutual-best greedy links between segment ends of ``eligible`` segments (bool per id).
    Cost = gap/reach + misalignment of both tangents with the gap vector + width mismatch.
    Returns links [(cost, end_a, end_b, gap)] with end dicts from :func:`endpoints`."""
    E = [e for e in endpoints(V) if eligible[e["seg"]]]
    if not E:
        return []
    Y = np.array([e["y"] for e in E]); X = np.array([e["x"] for e in E])
    cell = {}
    for i, (y, x) in enumerate(zip(Y, X)):
        cell.setdefault((int(y // reach), int(x // reach)), []).append(i)
    cand = []
    for i, a in enumerate(E):
        cy, cx = int(Y[i] // reach), int(X[i] // reach)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for j in cell.get((cy + dy, cx + dx), ()):
                    if j <= i or E[j]["seg"] == a["seg"]:
                        continue
                    b = E[j]
                    g = np.array([b["y"] - a["y"], b["x"] - a["x"]]); d = float(np.hypot(*g))
                    if d > reach:
                        continue
                    if d < 2.5:
                        mis = 1.0 - float(-(a["t"] @ b["t"]))          # ends should face each other
                    else:
                        gu = g / d
                        mis = 0.5 * ((1.0 - float(a["t"] @ gu)) + (1.0 - float(-(b["t"] @ gu))))
                    wd = abs(a["width"] - b["width"]) / max(a["width"], b["width"], 1.0)
                    c = d / reach + 2.0 * mis + 1.5 * wd
                    if c <= max_cost:
                        cand.append((c, i, j, d))
    cand.sort(key=lambda t: t[0])
    used = set(); links = []
    for c, i, j, d in cand:
        if i in used or j in used:
            continue
        used.add(i); used.add(j)
        links.append((c, E[i], E[j], d))
    return links


def rescue_vertical(V: dict, cls: np.ndarray, links: list, min_cos: float = 0.94) -> np.ndarray:
    """Grid-labelled straight segments linked at BOTH ends to curve segments with tangents within
    ~20 deg of their own direction are curve (a steep curve stretch beside a grid column)."""
    cls = cls.copy()
    ends = {}
    for c, a, b, d in links:
        ends.setdefault(a["seg"], []).append((a, b)); ends.setdefault(b["seg"], []).append((b, a))
    for sid, lst in ends.items():
        if cls[sid] != GRID or len(lst) < 2:
            continue
        ok = set()
        for me, other in lst:
            if cls[other["seg"]] == CURVE and float(-(me["t"] @ other["t"])) >= min_cos:
                ok.add(me["end"])
        if ok == {0, 1}:
            cls[sid] = CURVE
    return cls


def build_chains(V: dict, cls: np.ndarray, links: list, ink: np.ndarray | None = None) -> dict:
    """Union-find over curve links -> chains. Each chain: segs, pixels (incl. straight bridges over
    link gaps), n (px), span (rows), width, gaps (bridged link gaps > 2.5 px), dash (gaps per 100
    rows), dash_reg (1 - CV of gap lengths) and dashed (bool)."""
    pix = _seg_pixels(V)
    par = {}

    def f(a):
        while par.setdefault(a, a) != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for s in V["segs"]:
        if s is not None and cls[s["id"]] == CURVE:
            f(s["id"])
    bridges = {}
    for c, a, b, d in links:
        if cls[a["seg"]] != CURVE or cls[b["seg"]] != CURVE:
            continue
        ra, rb = f(a["seg"]), f(b["seg"])
        if ra != rb:
            par[ra] = rb
        bridges.setdefault((a["seg"], b["seg"]), (a, b, d))
    # side branches: a curve segment with one free end whose other end sits at a junction next to a
    # chain (a spike leaving the curve) joins that chain
    import cv2
    linked = {(a["seg"], a["end"]) for c, a, b, d in links} | {(b["seg"], b["end"]) for c, a, b, d in links}
    nj, jl = cv2.connectedComponents(V["junction"].astype(np.uint8), connectivity=8)
    H, W = V["seg"].shape
    at = {}
    ends = [e for e in endpoints(V) if cls[e["seg"]] == CURVE]
    for e in ends:
        y, x = int(e["y"]), int(e["x"])
        w = jl[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        e["j"] = int(w[w > 0][0]) if (w > 0).any() else 0
        if e["j"]:
            at.setdefault(e["j"], []).append(e)
    byseg = {}
    for e in ends:
        byseg.setdefault(e["seg"], []).append(e)
    for sid, es in byseg.items():
        if len(es) != 2 or V["segs"][sid]["n"] < 2 * max(V["segs"][sid]["width"], 2.0):
            continue
        for e, o in ((es[0], es[1]), (es[1], es[0])):
            if e["j"] and not o["j"] and (o["seg"], o["end"]) not in linked:
                nb = [q for q in at.get(e["j"], []) if q["seg"] != sid]
                if nb:
                    best = max(nb, key=lambda q: V["segs"][q["seg"]]["n"])
                    ra, rb = f(sid), f(best["seg"])
                    if ra != rb:
                        par[ra] = rb
                break
    groups = {}
    for sid in list(par):
        groups.setdefault(f(sid), []).append(sid)
    chains = []
    cid = np.zeros(len(V["segs"]), np.int32)
    br_by_root = {}
    for (sa, sb), v in bridges.items():
        br_by_root.setdefault(f(sa), []).append(v)
    for root, segs in groups.items():
        ys = [pix[s][0] for s in segs if pix[s] is not None]; xs = [pix[s][1] for s in segs if pix[s] is not None]
        gaps = []
        for a, b, d in br_by_root.get(root, []):
            k = int(np.ceil(d)) + 1
            ys.append(np.rint(np.linspace(a["y"], b["y"], k)).astype(int)); xs.append(np.rint(np.linspace(a["x"], b["x"], k)).astype(int))
            if d > 2.5:
                gy, gx = ys[-1], xs[-1]
                on = float(ink[np.clip(gy, 0, ink.shape[0] - 1), np.clip(gx, 0, ink.shape[1] - 1)].mean()) if ink is not None else 0.0
                if on < 0.5:          # a real paper gap (dash), not a junction bridged over ink
                    gaps.append(d)
        if not ys:
            continue
        yy = np.concatenate(ys); xx = np.concatenate(xs)
        n = int(sum(V["segs"][s]["n"] for s in segs))
        span = int(yy.max() - yy.min() + 1)
        w = float(np.average([V["segs"][s]["width"] for s in segs], weights=[V["segs"][s]["n"] for s in segs]))
        dash = 100.0 * len(gaps) / max(span, 1)
        reg = float(1.0 - np.std(gaps) / np.mean(gaps)) if len(gaps) >= 3 else 0.0
        ch = dict(id=len(chains) + 1, segs=segs, y=yy, x=xx, n=n, span=span, width=w, gaps=len(gaps),
                  dash=dash, dash_reg=reg, dashed=bool(len(gaps) >= 3 and dash >= 1.5 and reg >= 0.3))
        chains.append(ch)
        cid[segs] = ch["id"]
    return dict(chains=chains, seg_chain=cid)


def text_fix(V: dict, cls: np.ndarray, C: dict, min_span: int = 60) -> np.ndarray:
    """Text must be compact clusters of short strokes: a segment flagged OTHER that belongs to a
    long linked chain (span >= ``min_span`` rows or >= 3 linked segments) is curve."""
    cls = cls.copy()
    by = {c["id"]: c for c in C["chains"]}
    for s in V["segs"]:
        if s is None or cls[s["id"]] != OTHER:
            continue
        c = by.get(int(C["seg_chain"][s["id"]]))
        if c is not None and (c["span"] >= min_span or len(c["segs"]) >= 3):
            cls[s["id"]] = CURVE
    return cls


def chain_groups(chains: list, n_curves: int, min_n: int = 30) -> np.ndarray:
    """Style group per chain (index = position in ``chains``; -1 = too short to judge): 1-D/2-D
    k-means on (dashed score, width), length weighted. Group 0 = solid/thickest."""
    g = -np.ones(len(chains), np.int16)
    idx = [i for i, c in enumerate(chains) if c["n"] >= min_n]
    if not idx:
        return g
    if n_curves <= 1:
        g[idx] = 0
        return g
    F = np.array([[chains[i]["width"], 3.0 * min(chains[i]["dash"], 3.0) * chains[i]["dashed"]] for i in idx])
    wts = np.array([chains[i]["n"] for i in idx], float)
    sc = F.std(0); sc[sc == 0] = 1; Z = F / sc
    key = Z[:, 0] - Z[:, 1]                          # solid & thick first
    cen = np.array([Z[np.argsort(key)[int(q * (len(Z) - 1))]] for q in np.linspace(1, 0, n_curves)])
    for _ in range(30):
        a = np.argmin(((Z[:, None] - cen[None]) ** 2).sum(-1), 1)
        for k in range(n_curves):
            if (a == k).any():
                cen[k] = np.average(Z[a == k], axis=0, weights=wts[a == k])
    order = np.argsort(-(cen[:, 0] - cen[:, 1])); rank = np.empty(n_curves, int); rank[order] = np.arange(n_curves)
    g[idx] = rank[a]
    return g


def chain_path(chains: list, groups: np.ndarray, shape, fy: int, target: int | None = None,
               switch: float = 0.35, none_cost: float = 0.9, style_pen: float = 0.8, spike: float = 0.25) -> tuple:
    """Viterbi over sample rows (step ``fy``) choosing one linked chain per sample. States per
    sample: (chain, leftmost x) and (chain, rightmost x) of each chain in the rows, plus 'none'.
    Long chains and the ``target`` style group are cheap; staying on a chain is free, switching
    costs ``switch`` + distance; the extreme farther from the chain's local median is favoured
    (keeps spike tips). Returns (x per sample, covered bool per sample, chain id per sample)."""
    H, W = shape
    ns = int(np.ceil(H / fy))
    per = [[] for _ in range(ns)]
    for i, c in enumerate(chains):
        if groups[i] < 0:
            continue
        s = np.clip(c["y"] // fy, 0, ns - 1)
        o = np.argsort(s, kind="stable"); s, xx = s[o], c["x"][o]
        cuts = np.flatnonzero(np.diff(s)) + 1
        ss = np.split(s, cuts); xs = np.split(xx, cuts)
        mid = np.array([np.median(x) for x in xs])
        k = 7
        med = np.array([np.median(mid[max(0, j - k):j + k + 1]) for j in range(len(mid))]) if len(mid) > 1 else mid
        base = 1.0 / (1.0 + c["n"] / 150.0) + (style_pen if target is not None and groups[i] != target else 0.0)
        for j, (sv, x) in enumerate(zip(ss, xs)):
            lo, hi = float(x.min()), float(x.max())
            for xv in ((lo,) if hi - lo < 2 else (lo, hi)):
                e = base - spike * min(1.0, abs(xv - med[j]) / 40.0)
                per[int(sv[0])].append((c["id"], xv, e))
    cost = np.array([0.0]); xs_prev = np.array([np.nan]); id_prev = np.array([0])
    back = []
    for t in range(ns):
        st = per[t]
        ids = np.array([0] + [s[0] for s in st]); xv = np.array([np.nan] + [s[1] for s in st])
        em = np.array([none_cost] + [s[2] for s in st])
        same = ids[:, None] == id_prev[None, :]
        dx = np.abs(xv[:, None] - xs_prev[None, :])
        tr = np.where(same & (ids[:, None] > 0), 0.0, switch + 0.004 * np.nan_to_num(dx, nan=0.0))
        tr[0, :] = 0.15; tr[:, id_prev == 0] = np.minimum(tr[:, id_prev == 0], 0.15); tr[0, id_prev == 0] = 0.0
        tot = cost[None, :] + tr
        b = np.argmin(tot, 1)
        cost = tot[np.arange(len(ids)), b] + em
        back.append(b); xs_prev, id_prev = xv, ids
        back[-1] = (b, ids, xv)
    x = np.full(ns, np.nan); cid = np.zeros(ns, np.int32)
    k = int(np.argmin(cost))
    for t in range(ns - 1, -1, -1):
        b, ids, xv = back[t]
        x[t] = xv[k]; cid[t] = ids[k]; k = int(b[k])
    return x, cid > 0, cid


def vector_separate_v2(ink: np.ndarray, n_curves: int = 1) -> dict:
    """v2: vectorize -> classify -> link -> rescue steep curve stretches from grid -> fix text ->
    chains with style -> style groups."""
    from pylogdigitizer import separation as sep
    ink = np.asarray(ink, bool)
    grid = sep.grid_mask(ink)
    bands = sep.hline_bands(ink)
    try:
        oth = sep.other_mask(ink, grid)
    except Exception:
        oth = None
    V = vectorize(ink, grid)
    cls0 = classify_segments(V, ink, grid, bands, oth)
    elig = np.isin(cls0, [CURVE, OTHER])
    vgrid = np.array([s is not None and cls0[s["id"]] == GRID and s["angle"] < 15 and s["h"] < 400 for s in
                      [None] + V["segs"][1:]])
    links = link_segments(V, elig | vgrid)
    cls = rescue_vertical(V, cls0, links)
    C = build_chains(V, np.where(cls == OTHER, CURVE, cls), links, ink)   # text candidates may join chains
    cls = text_fix(V, cls, C)
    C = build_chains(V, cls, links, ink)
    groups = chain_groups(C["chains"], n_curves)
    return dict(curve_ink=curve_panel(ink, V, cls), V=V, cls=cls, cls0=cls0, links=links, chains=C["chains"],
                seg_chain=C["seg_chain"], groups=groups, grid=grid, bands=bands,
                n_rescued=int(((cls0 == GRID) & (cls == CURVE)).sum()),
                n_text_fixed=int(((cls0 == OTHER) & (cls == CURVE)).sum()))


def apply_chains(T: dict, vec: dict, shape, n_curves: int = 1) -> dict:
    """Replace a tracer result's x by linked-chain paths. Single curve: one path over all chains.
    Several: one path per style group; each tracer candidate takes the group path that agrees with it
    most. Samples no chain covers keep the tracer's x (``vector_fallback`` counts them)."""
    cands = T.get("candidates") or [T]
    fy = int(T["fy"])
    ng = max(1, int(vec["groups"].max()) + 1) if len(vec["groups"]) else 1
    paths = [chain_path(vec["chains"], vec["groups"], shape, fy, target=(k if n_curves > 1 else None))
             for k in range(ng if n_curves > 1 else 1)]
    used = set()
    for t in cands:
        n = len(t["x"]); tx = np.asarray(t["x"], float)
        best, bk = None, 0
        for k, (x, cov, _) in enumerate(paths):
            m = cov[:n]
            if not m.any():
                continue
            agree = float(np.mean(np.abs(x[:n][m] - tx[:len(m)][m]) < 8)) - (0.5 if k in used else 0.0)
            if best is None or agree > best:
                best, bk = agree, k
        used.add(bk)
        x, cov, cid = paths[bk]
        m = cov[:n]
        fx = t["fx"]; pad = t["pad"]
        newx = tx.copy(); newx[m] = x[:n][m]
        t["x"] = newx
        pp = np.asarray(t["path_padded"]).copy()
        pp[m] = np.clip((newx[m] // fx).astype(int) + pad, pad, t["cost"].shape[1] - pad - 1)
        t["path_padded"] = pp; t["path"] = np.clip(pp - pad, 0, t["cost"].shape[1] - 2 * pad - 1)
        ol, orr = np.asarray(t["off_left"]).copy(), np.asarray(t["off_right"]).copy()
        ol[m] = False; orr[m] = False
        t["off_left"], t["off_right"] = ol, orr
        t["vector_group"] = bk; t["vector_rows"] = m; t["vector_fallback"] = int((~m).sum())
        t["vector_chain"] = cid[:n]
    return T
