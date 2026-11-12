# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""track_grid_detection: vertical track boundaries, grid lines, depth column, linear vs log grid."""
from __future__ import annotations
import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2].tolist(), e[1::2].tolist()))


def vertical_lines(ink: np.ndarray, min_frac: float = 0.5) -> list[dict]:
    """Columns that are ink over >= ``min_frac`` of the window, merged into lines
    {x0, x1, center, width, frac}."""
    cf = ink.mean(axis=0)
    return [dict(x0=a, x1=b, center=(a + b - 1) / 2, width=b - a, frac=float(cf[a:b].mean()))
            for a, b in _runs(cf >= min_frac)]


def horizontal_lines(ink: np.ndarray, min_frac: float = 0.5) -> list[dict]:
    """Rows that are ink over >= ``min_frac`` of the window width: {y0, y1, center, width, frac}."""
    rf = ink.mean(axis=1)
    return [dict(y0=a, y1=b, center=(a + b - 1) / 2, width=b - a, frac=float(rf[a:b].mean()))
            for a, b in _runs(rf >= min_frac)]


def find_header_bottom(ink: np.ndarray, min_run: int = 3000) -> int:
    """First row of the log body: start of the first long continuous run of the strongest
    vertical boundary line."""
    lines = vertical_lines(ink, 0.3)
    if not lines:
        return 0
    best = max(lines, key=lambda l: l["frac"] * l["width"])
    col = ink[:, best["x0"]:best["x1"]].any(axis=1)
    for a, b in _runs(col):
        if b - a >= min_run:
            return a
    return 0


def detect_tracks(ink: np.ndarray, min_track_width: int = 150, boundary_min_width: int = 5,
                  boundary_min_frac: float = 0.78, line_frac: float = 0.5, merge: bool = True) -> list[dict]:
    """Track layout from a body window (rows x cols), without header text.

    1. Vertical lines = columns with ink over >= ``line_frac`` of rows (dashed grid lines count).
    2. Boundaries = thick (>= ``boundary_min_width`` px) and nearly continuous lines.
    3. Each gap between boundaries is a panel: "depth" if it has no internal grid lines, else "track".
    4. Adjacent track panels with the same grid spacing are merged (heavy mid-track lines, e.g. the
       50 % line or the track 2/3 divider on a combined linear grid, are not track edges).
    Returns [{left, right, kind, grid_x}] (left/right = inner edges of boundary lines; grid_x =
    centers of all vertical lines from the left boundary to the right boundary).
    """
    lines = vertical_lines(ink, line_frac)
    bounds = [l for l in lines if l["width"] >= boundary_min_width and l["frac"] >= boundary_min_frac]
    panels = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b["x0"] - a["x1"] < 20:
            continue
        inner = [l["center"] for l in lines if a["x1"] <= l["x0"] and l["x1"] <= b["x0"]]
        gx = [a["center"]] + inner + [b["center"]]
        sp = float(np.median(np.diff(gx))) if inner else None
        panels.append(dict(left=int(a["x1"]), right=int(b["x0"]), kind="track" if inner else "depth",
                           grid_x=gx, spacing=sp))
    if merge:
        merged = []
        for p in panels:
            q = merged[-1] if merged else None
            if (q is not None and q["kind"] == p["kind"] == "track" and q["right"] + 15 >= p["left"]
                    and abs(q["spacing"] - p["spacing"]) <= 0.1 * q["spacing"]):
                q["right"] = p["right"]; q["grid_x"] = q["grid_x"] + p["grid_x"][1:]
                q["spacing"] = float(np.median(np.diff(q["grid_x"])))
            else:
                merged.append(dict(p))
        panels = merged
    return [p for p in panels if p["right"] - p["left"] >= min_track_width or p["kind"] == "depth"]


def classify_grid(track_or_grid_x, ink: np.ndarray | None = None) -> str:
    """'linear' if vertical grid spacing is uniform, 'log' if spacing shrinks within decades.
    Accepts a track dict / list of grid x positions, or a track image (then detects lines)."""
    if isinstance(track_or_grid_x, np.ndarray) and track_or_grid_x.ndim == 2:
        xs = [l["center"] for l in vertical_lines(track_or_grid_x, 0.5)]
    elif isinstance(track_or_grid_x, dict):
        xs = track_or_grid_x["grid_x"]
    else:
        xs = list(track_or_grid_x)
    d = np.diff(np.asarray(xs, float))
    if len(d) < 3:
        return "unknown"
    cv = d.std() / d.mean()
    if cv < 0.12:
        return "linear"
    # log: first gaps in a decade are larger than following ones (log10(2)=0.30, log10(3/2)=0.18 ...)
    return "log" if np.sum(np.diff(d[:4]) < 0) >= 2 else "unknown"


def depth_column(tracks: list[dict]) -> dict | None:
    """The region classified as the depth column (between track 1 and track 2)."""
    for t in tracks:
        if t["kind"] == "depth":
            return t
    return None


def grid_rows(track_ink: np.ndarray, min_frac: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal grid line centers inside one track and whether each is a heavy (major) line.
    Returns (centers, is_major)."""
    hl = horizontal_lines(track_ink, min_frac)
    if not hl:
        return np.zeros(0), np.zeros(0, bool)
    c = np.array([l["center"] for l in hl]); w = np.array([l["width"] for l in hl])
    return c, w >= max(3, np.median(w) + 2)


def follow_lines(ink: np.ndarray, x_start: list[float], block: int = 1500, search: int = 30,
                 min_frac: float = 0.6, rows: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Follow vertical lines (e.g. track boundaries) down a long scan whose x drifts.
    ``x_start`` = line centres in the first block. In each block the strongest column-sum peak
    within ``search`` px of the previous position is taken (kept if nothing is found).
    Returns (block_centre_rows, xs[n_blocks, n_lines])."""
    r0, r1 = rows or (0, ink.shape[0])
    prev = np.array(x_start, float)
    centers, xs = [], []
    for s in range(r0, r1, block):
        b = ink[s:min(r1, s + block)]
        if b.shape[0] < block // 3:
            break
        cf = b.mean(axis=0)
        cur = prev.copy()
        for i, x in enumerate(prev):
            a, e = max(0, int(x) - search), min(b.shape[1], int(x) + search + 1)
            seg = cf[a:e]
            if seg.size and seg.max() >= min_frac:
                m = seg >= max(min_frac, 0.8 * seg.max())
                idx = np.flatnonzero(m)
                # centre of the strongest run
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                best = max(runs, key=lambda r_: seg[r_].sum())
                cur[i] = a + (best[0] + best[-1]) / 2
        centers.append(s + b.shape[0] / 2); xs.append(cur); prev = cur
    return np.array(centers), np.array(xs)


def rectify(ink: np.ndarray, rows: tuple[int, int], centers: np.ndarray, left: np.ndarray, right: np.ndarray,
            width: int, chunk: int = 20000) -> np.ndarray:
    """Resample rows ``rows`` of a drifting panel to a fixed width: column j of the output is at
    x = left(y) + (j + 0.5) / width * (right(y) - left(y)) (nearest pixel), with left/right
    linearly interpolated between block centres. Output bool (rows x width)."""
    out = np.zeros((rows[1] - rows[0], width), dtype=bool)
    f = (np.arange(width) + 0.5) / width
    for s in range(rows[0], rows[1], chunk):
        e = min(rows[1], s + chunk)
        y = np.arange(s, e)
        L = np.interp(y, centers, left); R = np.interp(y, centers, right)
        X = np.clip(np.round(L[:, None] + f[None, :] * (R - L)[:, None]).astype(np.int64), 0, ink.shape[1] - 1)
        out[s - rows[0]:e - rows[0]] = ink[y[:, None], X]
    return out


def detect_tracks_by_spacing(ink: np.ndarray, line_frac: float = 0.45, merge_px: int = 8,
                             tol: float = 0.15, min_intervals: int = 4) -> list[dict]:
    """Fallback layout for logs without heavy track borders: tracks are runs of evenly spaced
    vertical grid lines; the depth column is the gap (>= 3 spacings, no lines) between runs.
    A missing line under a flat curve (gap of 2 spacings) does not break a run."""
    ls = vertical_lines(ink, line_frac)
    c = []
    for l in ls:
        if c and l["center"] - c[-1][0] <= merge_px:
            c[-1] = ((c[-1][0] * c[-1][1] + l["center"] * l["width"]) / (c[-1][1] + l["width"]), c[-1][1] + l["width"])
        else:
            c.append((l["center"], l["width"]))
    x = np.array([v[0] for v in c])
    if len(x) < 4:
        return []
    d = np.diff(x)
    dv = d[(d > 10) & (d < 200)]
    if dv.size == 0:                  # no plausible grid spacing
        return []
    s = float(np.median(dv))
    runs, cur = [], [x[0]]
    for xi, di in zip(x[1:], d):
        k = round(di / s)
        if k in (1, 2) and abs(di - k * s) <= tol * s * k:
            cur.append(xi)
        elif di < (1 - tol) * s:      # stray line inside a cell (curve fragment): skip
            continue
        else:
            runs.append(cur); cur = [xi]
    runs.append(cur)
    out = []
    tracks = [r for r in runs if (r[-1] - r[0]) / s >= min_intervals]
    for i, r in enumerate(tracks):
        n = int(round((r[-1] - r[0]) / s))
        out.append(dict(left=int(round(r[0])), right=int(round(r[-1])), kind="track",
                        grid_x=list(np.linspace(r[0], r[-1], n + 1)), spacing=s))
        if i + 1 < len(tracks):
            nxt = tracks[i + 1]
            out.append(dict(left=int(round(r[-1])), right=int(round(nxt[0])), kind="depth",
                            grid_x=[float(r[-1]), float(nxt[0])], spacing=None))
    return out


# --- block layout + probe (moved from workflows) ---------------------------------------------
def _valid_layout(tr: list[dict], width: int) -> bool:
    """Exactly one depth column with a track on each side, depth column 3-20 % of the sheet."""
    kinds = [t["kind"] for t in tr]
    if kinds.count("depth") != 1:
        return False
    i = kinds.index("depth")
    if i == 0 or i == len(tr) - 1:
        return False
    w = tr[i]["right"] - tr[i]["left"]
    return 0.03 * width <= w <= 0.2 * width


def detect_tracks_by_hgrid(ink: np.ndarray, run: int = 25, rel: float = 0.25, min_gap: int = 40,
                           merge_gap: int = 12) -> list[dict]:
    """Fallback for log (non-uniform) grids and depth columns crossed by grid lines: tracks are
    the x ranges covered by horizontal grid-line ink (short horizontal runs), the depth column
    is a gap in that coverage. Panel edges are snapped to the nearest strong vertical line."""
    import cv2
    h = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, run), np.uint8))
    p = h.mean(axis=0)
    if p.max() <= 0:
        return []
    k = np.ones(9) / 9
    ps = np.convolve(p, k, mode="same")
    on = ps > rel * np.percentile(ps[ps > 0], 75)
    segs = _runs(on)
    # merge small gaps (e.g. a vertical heavy line, a label)
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    segs = [(a, b) for a, b in merged if b - a >= 150]
    if len(segs) < 2:
        return []
    vl = vertical_lines(ink, 0.5)
    cx = np.array([l["center"] for l in vl]) if vl else np.array([])
    def snap(x):
        if len(cx):
            j = np.argmin(abs(cx - x))
            if abs(cx[j] - x) <= 15:
                return float(cx[j])
        return float(x)
    out = []
    for i, (a, b) in enumerate(segs):
        l, r = snap(a), snap(b)
        out.append(dict(left=int(round(l)), right=int(round(r)), kind="track", grid_x=[l, r], spacing=None))
        if i + 1 < len(segs) and segs[i + 1][0] - b >= min_gap:
            out.append(dict(left=int(round(r)), right=int(round(snap(segs[i + 1][0]))), kind="depth",
                            grid_x=[r, snap(segs[i + 1][0])], spacing=None))
    return out


def detect_tracks_thin(ink: np.ndarray, border_width: int = 3, boundary_min_frac: float = 0.78) -> list[dict]:
    """Track layout for digital plots whose track borders are thin (2-4 px) solid lines, e.g.
    the Eli / Gemini (Warrior) triple-combo prints: borders are the widest continuous lines
    (>= ``border_width`` px, wider than the 1 px grid). ``boundary_min_frac`` stays at the default
    rule's 0.78: depth numbers break the depth-column borders of a working image (69-92 % of the rows
    on STONE 5-19). :func:`workflows.analyze_layout` accepts this layout only when it adds splits to
    the one that already worked (:func:`only_adds_splits`). Same panel rules as :func:`detect_tracks`."""
    return detect_tracks(ink, boundary_min_width=border_width, boundary_min_frac=boundary_min_frac)


def detect_layout(block_ink: np.ndarray, thin_borders: bool = False) -> list[dict]:
    """Panels of one block: thick-border detection if it yields a valid layout (one depth column
    between tracks), else the grid-spacing fallback; [] if neither is valid.
    ``thin_borders``: try :func:`detect_tracks_thin` first (thin-bordered digital plots).
    Each returned panel carries ``"detector"``: the name of the function that produced it."""
    w = block_ink.shape[1]
    fns = (detect_tracks, detect_tracks_by_spacing, detect_tracks_by_hgrid)
    if thin_borders:
        fns = (detect_tracks_thin,) + fns
    for fn in fns:
        try:
            tr = fn(block_ink)
        except Exception:
            tr = []
        if _valid_layout(tr, w):
            for t in tr:
                t.setdefault("detector", fn.__name__)
            return tr
    return []


def boundaries(tracks: list[dict]) -> list[float]:
    """x of every panel edge of a layout (left and right of each track / depth panel)."""
    return sorted({float(t[k]) for t in tracks for k in ("left", "right")})


def only_adds_splits(base: list[dict], new: list[dict], tol: int = 15) -> bool:
    """True when every panel edge of ``base`` is also an edge of ``new`` (within ``tol`` px): ``new``
    may split panels of ``base`` but never moves or drops one of its edges."""
    nb = boundaries(new)
    return bool(nb) and all(min(abs(x - y) for y in nb) <= tol for x in boundaries(base))


def same_depth_column(a: list[dict], b: list[dict], tol: int = 15) -> bool:
    """One depth column in each layout, at the same position (within ``tol`` px)."""
    da = [t for t in a if t["kind"] == "depth"]; db = [t for t in b if t["kind"] == "depth"]
    return (len(da) == len(db) == 1 and abs(da[0]["left"] - db[0]["left"]) <= tol
            and abs(da[0]["right"] - db[0]["right"]) <= tol)


def same_layout(a: list[dict], b: list[dict], tol: int = 15) -> bool:
    """Same depth-column position (within ``tol`` px) and same number of tracks."""
    return (same_depth_column(a, b, tol)
            and sum(t["kind"] == "track" for t in a) == sum(t["kind"] == "track" for t in b))


def find_probe(ink: np.ndarray, block: int = 800, stride: int | None = None, confirm: int = 2,
               thin_borders: bool = False) -> tuple[int, list[dict]]:
    """First block from the top of the sheet (just below the header) whose layout is valid and
    confirmed by the next ``confirm`` blocks. Raises LayoutError if none."""
    stride = stride or block // 2
    h = ink.shape[0]
    cache = {}
    def lay(s):
        if s not in cache:
            cache[s] = detect_layout(ink[s:s + block], thin_borders)
        return cache[s]
    for s in range(0, max(1, h - (confirm + 2) * block), stride):
        tr = lay(s)
        if tr and sum(same_layout(tr, lay(s + k * block)) for k in range(1, confirm + 2)) >= confirm:
            # refine upwards: the log body (and the first depth labels) may start above s;
            # single-block detections are noisy (curves hide grid lines), so allow one miss
            top, miss = s, 0
            for s2 in range(s - stride // 2, -1, -(stride // 2)):
                t2 = detect_layout(ink[s2:s2 + block], thin_borders)
                if t2 and same_layout(tr, t2):
                    top, miss = s2, 0
                else:
                    miss += 1
                    if miss >= 3:
                        break
            return top, tr
    raise LayoutError("no stable track layout (depth column between tracks) found on this sheet; "
                      "check that it is a depth-indexed log, or pass x_range/rows overrides")


class LayoutError(RuntimeError):
    """Track layout could not be detected."""
