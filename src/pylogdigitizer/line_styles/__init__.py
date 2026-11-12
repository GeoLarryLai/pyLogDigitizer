# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Black-ink curves told apart by line style (stroke width and dash / dot pattern).

A black-and-white log often draws two to four curves in one track, each in its own style:
solid, dashed, dotted or dash-dot, thin or thick. This module labels the curve ink of each track
by style, so that every style becomes its own *class* - exactly like an ink colour on a colour
scan. From there the colour workflow is reused unchanged: one working image per curve
(:func:`pylogdigitizer.curve_ink.make_curve_ink`), traced in an assisted session
(:func:`pylogdigitizer.workflows.open_color_curve`), written to LAS.

How the ink is labelled (per track, inside the log body):

1. The printed grid is removed (:func:`pylogdigitizer.separation.remove_grid`: curve pixels on
   grid lines are bridged back).
2. Connected pieces of what is left are measured: length (box diagonal), stroke thickness
   (area / length; ridge of the distance transform for long pieces) and centroid.
3. The curve ink is split by stroke width (a morphological opening keeps the heavy strokes),
   then in each width layer long pieces are *solid* line; short pieces are *dashes* (longer) or *dots* (shorter).
   Pieces are kept only when they have neighbours of their own kind nearby (a dashed or dotted
   line is a chain). Dots with many neighbours are *fill* (stippled shading between curves) and
   isolated specks (grid nodes, dust) are dropped. Dashes and dots that alternate along one
   chain are *dash-dot*.
4. One style per pattern; styles whose stroke is clearly heavier (``thick_ratio``) than the
   lightest style in the track are *thick*, the others *thin* (no weight in the name when all
   have the same weight).

A style is proposed for a track when its pieces cover enough of the depth range
(``min_coverage``). The proposal is a starting point the user confirms (one curve per style).
Header legend samples (:func:`pylogdigitizer.header_ocr.legend_styles`) can be matched to the
proposed styles with :func:`match_legend`.
"""
from __future__ import annotations

import numpy as np

PATTERNS = ("solid", "dashed", "dotted", "dash-dot")
WEIGHTS = ("thin", "thick")
#: tracer settings for one-style working images: black-and-white defaults (grid handled by the
#: tracer), but no specks filter so dots and dashes are kept
LINE_TRACE_PRESET = dict(min_area=0)
#: dash gaps closed per pattern when building the working image (px at 300 dpi)
GAP_CLOSE_PX = {"solid": 5, "dashed": 11, "dotted": 9, "dash-dot": 13}

_SOLID, _DASH, _DOT, _FILL, _DASHDOT = 1, 2, 3, 4, 5


def style_name(pattern: str, weight: str | None = None) -> str:
    """Class name of a style, e.g. 'solid thick', 'dotted thin', 'dashed'."""
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}, got {pattern!r}")
    if weight is not None and weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    return f"{pattern} {weight}" if weight else pattern


def _neighbour_counts(shape, pts_from, pts_to, radius: float) -> np.ndarray:
    """For each point of ``pts_from``: how many points of ``pts_to`` lie within ``radius``
    (a disc-filtered point image; a point counts itself when it is in both sets)."""
    import cv2
    if len(pts_from) == 0:
        return np.zeros(0, int)
    img = np.zeros(shape, np.float32)
    if len(pts_to):
        yy = np.clip(np.round(pts_to[:, 1]).astype(int), 0, shape[0] - 1)
        xx = np.clip(np.round(pts_to[:, 0]).astype(int), 0, shape[1] - 1)
        np.add.at(img, (yy, xx), 1.0)
    r = int(np.ceil(radius))
    k = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)
    cnt = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_CONSTANT)
    yy = np.clip(np.round(pts_from[:, 1]).astype(int), 0, shape[0] - 1)
    xx = np.clip(np.round(pts_from[:, 0]).astype(int), 0, shape[1] - 1)
    return np.round(cnt[yy, xx]).astype(int)


def _ridge_width(m: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    """Mean stroke width (2 x distance-transform ridge) of each labelled piece."""
    import cv2
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = m & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)))
    s = np.bincount(lab[ridge], weights=2.0 * dt[ridge], minlength=n)
    c = np.bincount(lab[ridge], minlength=n)
    return s / np.maximum(c, 1)


def classify_pieces(curve_ink: np.ndarray, px_scale: float = 1.0, dot_len: float = 15.5,
                    long_len: float = 45.0, fill_neighbours: int = 5, grid: np.ndarray | None = None,
                    max_on_grid: float = 0.7) -> dict:
    """Label the pieces of grid-free curve ink of ONE track by pattern.

    Returns dict(lab (piece labels), kind (per piece: 0 none, 1 solid, 2 dash, 3 dot, 4 fill,
    5 dash-dot), width (per piece stroke width px), length (per piece px)). Lengths are px at
    300 dpi times ``px_scale``. ``grid``: mask of the printed grid lines (same shape); pieces
    lying mostly on it (> ``max_on_grid``) are leftovers of the grid, not curve."""
    import cv2
    m = np.asarray(curve_ink, bool)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    w_, h_ = st[:, 2].astype(float), st[:, 3].astype(float)
    ln = np.hypot(w_, h_)
    area = st[:, 4].astype(float)
    thick = area / np.maximum(ln, 1.0)
    width = _ridge_width(m, lab, n)
    dl, ll = dot_len * px_scale, long_len * px_scale
    kind = np.zeros(n, np.int8)
    ok = area >= max(3.0, 3.0 * px_scale ** 2)
    # grid slivers left by the grid removal: straight, at most 3 px across, elongated
    sliver = (np.minimum(w_, h_) <= 3 * px_scale) & (np.maximum(w_, h_) >= 4 * np.minimum(w_, h_)) & (width <= 3.2 * px_scale)
    ok &= ~sliver
    if grid is not None:
        g = np.asarray(grid, bool) & m
        ok &= np.bincount(lab[g], minlength=n) <= max_on_grid * np.maximum(area, 1)
    ok[0] = False
    is_long = ok & (ln >= ll)
    is_dot = ok & (ln < dl)
    is_dash = ok & (ln >= dl) & (ln < ll)
    kind[is_long] = _SOLID
    shape = m.shape
    di, ai = np.flatnonzero(is_dot), np.flatnonzero(is_dash)
    # dots: chain (1..fill-1 neighbours within ~1.5 dot spacings: a line of dots has two) vs fill
    # (stippled shading: a 2-D lattice, many neighbours) vs specks (none)
    nd = _neighbour_counts(shape, cen[di], cen[di], 18.0 * px_scale) - 1
    fill = nd >= fill_neighbours
    kind[di[fill]] = _FILL
    dot_ok = di[~fill & (nd >= 1)]
    # dashes: neighbours within dash length + gap (dashes or chain dots)
    lone = di[~fill & (nd == 0)]
    na = _neighbour_counts(shape, cen[ai], np.r_[cen[ai], cen[dot_ok], cen[lone]].reshape(-1, 2), 36.0 * px_scale) - 1
    na_dash = _neighbour_counts(shape, cen[ai], cen[ai], 36.0 * px_scale) - 1
    dash_ok = ai[na >= 1]
    # printed dashes have one length; much shorter / longer pieces are merged dots or broken line
    if len(dash_ok) >= 10:
        med = float(np.median(ln[dash_ok]))
        odd = np.abs(ln[dash_ok] - med) > 0.35 * med
        near_dots = _neighbour_counts(shape, cen[dash_ok], cen[dot_ok], 30.0 * px_scale) if len(dot_ok) else np.zeros(len(dash_ok), int)
        dot_ok = np.r_[dot_ok, dash_ok[odd & (near_dots >= 1)]].astype(int)
        dash_ok = dash_ok[~odd]
    # one pass of majority relabelling along chains: a piece surrounded by the other kind joins it
    if len(dot_ok) and len(dash_ok):
        r = 30.0 * px_scale
        dots_near_dot = _neighbour_counts(shape, cen[dot_ok], cen[dot_ok], r) - 1
        dash_near_dot = _neighbour_counts(shape, cen[dot_ok], cen[dash_ok], r)
        dots_near_dash = _neighbour_counts(shape, cen[dash_ok], cen[dot_ok], r)
        dash_near_dash = _neighbour_counts(shape, cen[dash_ok], cen[dash_ok], r) - 1
        to_dash = dash_near_dot > 2 * np.maximum(dots_near_dot, 0) + 1
        to_dot = dots_near_dash > 2 * np.maximum(dash_near_dash, 0) + 1
        dot_ok, dash_ok = np.r_[dot_ok[~to_dash], dash_ok[to_dot]].astype(int), np.r_[dash_ok[~to_dot], dot_ok[to_dash]].astype(int)
    kind[dot_ok] = _DOT
    kind[dash_ok] = _DASH
    # dash-dot: a dash with one or two dots right beside it (one gap away), and those dots
    # (a dash-dot line's dots are too far apart to chain on their own)
    cand_dots = np.r_[dot_ok, lone].astype(int)
    if len(cand_dots) and len(dash_ok):
        r = 24.0 * px_scale
        a_d = _neighbour_counts(shape, cen[dash_ok], cen[cand_dots], r)
        # ... and no other dash within one dashed-line period (a plain dashed line has one)
        dd = dash_ok[(a_d >= 1) & (a_d <= 2) & np.isin(dash_ok, ai[na_dash == 0])]
        if len(dd) >= 10:
            d_a = _neighbour_counts(shape, cen[cand_dots], cen[dd], r)
            # a chained dot beside a dash is part of a dotted line crossing it, unless it is lone
            take = (d_a >= 1) & np.r_[np.zeros(len(dot_ok), bool), np.ones(len(lone), bool)]
            if take.sum() >= 10:
                kind[dd] = _DASHDOT
                kind[cand_dots[take]] = _DASHDOT
    return dict(lab=lab, kind=kind, width=np.where(kind == _SOLID, width, thick), length=ln)


def _coverage(mask: np.ndarray, bin_rows: int = 20) -> float:
    h = mask.shape[0] // bin_rows * bin_rows
    if h == 0:
        return 0.0
    return float(mask[:h].reshape(-1, bin_rows, mask.shape[1]).any(axis=(1, 2)).mean())


def _kind_pattern(k: int) -> str:
    return {_SOLID: "solid", _DASH: "dashed", _DOT: "dotted", _DASHDOT: "dash-dot"}[k]


def width_layers(curve_ink: np.ndarray, heavy_px: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Split curve ink by stroke width: (heavy, light). Heavy = what survives a morphological
    opening with a ``heavy_px`` disc (strokes at least that wide), grown back by one pixel onto
    the ink; light = the rest (thin lines, dots, stipple)."""
    import cv2
    m = np.asarray(curve_ink, bool)
    k = max(3, int(round(heavy_px)) | 1)
    op = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    heavy = m & cv2.dilate(op, np.ones((3, 3), np.uint8)).astype(bool)
    return heavy, m & ~heavy


def track_styles(curve_ink: np.ndarray, px_scale: float = 1.0, min_coverage: float = 0.25,
                 heavy_px: float = 5.0, thick_ratio: float = 1.35, **kw) -> tuple[np.ndarray, list[dict]]:
    """Styles in ONE track's grid-free curve ink: (per-pixel style index map, 0 = none; styles).

    The ink is first split by stroke width (:func:`width_layers`), then each layer's pieces are
    labelled by pattern (:func:`classify_pieces`): a style is (pattern, weight). Each style:
    dict(name, pattern, weight, width_px, coverage, pieces). Index k in the map is styles[k-1]."""
    layers = dict(zip(("thick", "thin"), width_layers(curve_ink, heavy_px * px_scale)))
    found = []          # (pattern, weight, mask, width, pieces)
    for weight, layer in layers.items():
        if not layer.any():
            continue
        P = classify_pieces(layer, px_scale=px_scale, **kw)
        lab, kind, width = P["lab"], P["kind"], P["width"]
        for k in (_SOLID, _DASH, _DOT, _DASHDOT):
            ids = np.flatnonzero(kind == k)
            if len(ids) == 0:
                continue
            sel = np.zeros(len(kind), bool); sel[ids] = True
            m = sel[lab]
            cov = _coverage(m)
            if cov < min_coverage:
                continue
            a = P["length"][ids]
            found.append((_kind_pattern(k), weight, m, float(np.average(width[ids], weights=a)), int(len(ids)), cov))
    # the same pattern in both layers (dots near the width threshold, a dashed line partly heavy)
    # is one style: merge, weight from the mean stroke width
    def _merge(entries, key):
        out = {}
        for pat, weight, m, w, n, cov in entries:
            k = key(pat, weight)
            if k in out:
                p0, wt0, m0, w0, n0, _ = out[k]
                out[k] = (p0, wt0, m0 | m, (w0 * n0 + w * n) / (n0 + n), n0 + n, None)
            else:
                out[k] = (pat, weight, m, w, n, None)
        return [(p_, wt, m, w, n, _coverage(m)) for p_, wt, m, w, n, _ in out.values()]
    # solid pieces stay apart per layer (a thick and a thin solid curve); other patterns merge
    found = _merge(found, lambda pat, wt: (pat, wt) if pat == "solid" else (pat,))
    if found:
        wmin = min(f[3] for f in found)
        found = [(pat, "thick" if w >= thick_ratio * wmin else "thin", m, w, n, cov) for pat, _, m, w, n, cov in found]
        # the weight is re-read from the stroke width: two entries may now share a name (both solid
        # layers within thick_ratio): one style
        found = _merge(found, lambda pat, wt: (pat, wt))
    weights_used = {f[1] for f in found}
    styles, masks = [], []
    for pat, weight, m, w, n, cov in found:
        wname = weight if len(weights_used) > 1 else None
        styles.append(dict(pattern=pat, weight=wname, width_px=round(w, 2), coverage=round(cov, 3), pieces=n,
                           name=style_name(pat, wname)))
        masks.append(m)
    idx = np.zeros(curve_ink.shape, np.uint8)
    order = sorted(range(len(styles)), key=lambda i: -styles[i]["coverage"])
    for new, old in enumerate(order, start=1):
        idx[masks[old] & (idx == 0)] = new
    return idx, [styles[i] for i in order]


def grid_lines_mask(ink: np.ndarray, col_frac: float = 0.3, row_frac: float = 0.4, grow: int = 2) -> np.ndarray:
    """Printed grid of one track window: columns inked over >= ``col_frac`` of the rows and rows
    inked over >= ``row_frac`` of the width, each grown by ``grow`` px."""
    ink = np.asarray(ink, bool)
    k = np.ones(2 * grow + 1)
    cols = np.convolve((ink.mean(axis=0) >= col_frac).astype(float), k, "same") > 0
    rows = np.convolve((ink.mean(axis=1) >= row_frac).astype(float), k, "same") > 0
    return cols[None, :] | rows[:, None]


def suggest_line_styles(ink: np.ndarray, tracks: dict, body_rows=None, px_scale: float = 1.0,
                        min_coverage: float = 0.25, margin: int = 4, **kw) -> dict:
    """Label black curve ink by line style in every track.

    ``ink``: 1-bit sheet (True = ink). ``tracks``: {track name: (x_left, x_right)} on the sheet.
    ``body_rows``: (first, last) row of the log body (header and footer are left alone).

    Returns dict(labels, names, styles, curve_ink) where ``labels``/``names`` follow the colour
    convention (``names[labels]``: 'white' = paper, 'black' = grid / text / anything that is not
    a proposed curve, then one class per style name), ``styles`` = {track: [style dicts]} and
    ``curve_ink`` = the grid-free curve ink inside the tracks."""
    from pylogdigitizer.separation import remove_grid
    ink = np.asarray(ink, bool)
    r0, r1 = body_rows or (0, ink.shape[0])
    names = ["white", "black"]
    labels = np.where(ink, 1, 0).astype(np.uint8)
    curve_all = np.zeros_like(ink)
    out = {}
    for tname, (xl, xr) in tracks.items():
        x0, x1 = int(round(xl)) + margin, int(round(xr)) - margin
        if x1 - x0 < 10:
            continue
        win = ink[r0:r1, x0:x1]
        cur = remove_grid(win)
        idx, styles = track_styles(cur, px_scale=px_scale, min_coverage=min_coverage, grid=grid_lines_mask(win), **kw)
        curve_all[r0:r1, x0:x1] |= cur
        sub = labels[r0:r1, x0:x1]
        for k, s in enumerate(styles, start=1):
            if s["name"] not in names:
                names.append(s["name"])
            sub[idx == k] = names.index(s["name"])
            s["track"] = tname
        out[tname] = styles
    return dict(labels=labels, names=names, styles=out, curve_ink=curve_all)


def reassign_off_trace(sug: dict, style: str, to: str, x_range, rows, xs, body_rows=None,
                       half_width: float = 7.0, margin: int = 4, px_scale: float = 1.0,
                       heavy_px: float = 5.0) -> dict:
    """Split ink shared by two touching curves along a confirmed trace.

    Where two curves of one track cross, or run side by side with shading between them, their ink
    is one connected piece, and the piece classifier gives the whole piece the style of its longest
    run (usually *solid*): the dashes of the other curve vanish from their own class. Once ``style``
    has been traced (``rows``, ``xs``: sheet row and x of its trace), only its ink along the trace
    (within ``half_width`` px) is kept for it; the rest of the curve ink in the track that is not
    another proposed style, shading or a speck, and that has the pattern of style ``to`` (or is a
    long run of it), is given to ``to``, which keeps its own pieces.

    Returns a new suggestion dict (``labels`` copied); ``sug`` is not changed."""
    import cv2
    names = sug["names"]
    if style not in names or to not in names:
        raise ValueError(f"styles must be in {names[2:]}, got {style!r} and {to!r}")
    labels = sug["labels"].copy()
    r0, r1 = body_rows or (0, labels.shape[0])
    x0, x1 = int(round(x_range[0])) + margin, int(round(x_range[1])) - margin
    lab = labels[r0:r1, x0:x1]
    cur = np.asarray(sug["curve_ink"][r0:r1, x0:x1], bool)
    i_s, i_t = names.index(style), names.index(to)
    corridor = np.zeros(lab.shape, np.uint8)
    pts = np.c_[np.asarray(xs, float) - x0, np.asarray(rows, float) - r0]
    ok = np.isfinite(pts).all(axis=1)
    hw = max(1, int(round(half_width * px_scale)))
    for seg in np.split(pts, np.flatnonzero(~ok)):
        seg = seg[np.isfinite(seg).all(axis=1)]
        if len(seg) >= 2:
            cv2.polylines(corridor, [np.round(seg).astype(np.int32).reshape(-1, 1, 2)], False, 1,
                          thickness=2 * hw + 1)
    corridor = corridor.astype(bool)
    others = (lab >= 2) & (lab != i_s) & (lab != i_t)
    cand = cur & ~corridor & ~others
    weight = to.split(" ")[1] if " " in to else None
    heavy, light = width_layers(cand, heavy_px * px_scale)
    layer = heavy if weight == "thick" else light if weight == "thin" else cand
    P = classify_pieces(layer, px_scale=px_scale)
    # pieces of the target's own pattern, plus long runs (its dashes merged end to end); dots of the
    # shading are not taken by a dashed curve
    own = {"solid": (_SOLID,), "dashed": (_SOLID, _DASH), "dotted": (_SOLID, _DOT),
           "dash-dot": (_SOLID, _DASH, _DOT, _DASHDOT)}[to.split(" ")[0]]
    keep = np.isin(P["kind"], own)[P["lab"]]
    lab[(lab == i_s) & ~corridor] = 1
    lab[keep] = i_t
    return dict(sug, labels=labels)


def route_hidden_along_trace(ink: np.ndarray, sug: dict, style: str, rows, xs, near_px: float = 60.0,
                             window: int = 15, min_run: int = 15, half_width: float = 3.0,
                             px_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Find where ``style``'s ink is hidden under a traced curve, and route its trace along it there.

    Two curves of one track can run on top of each other for feet at a time (a dashed neutron curve
    under a solid density curve): the dashes merge into the solid stroke, so the dashed curve's
    working image has nothing there and its tracer wanders off to the next ink in the track. At
    every trace sample (``rows``, ``xs``: sheet row and x of the *covering*, already traced curve)
    this checks for ink labelled ``style`` within ``near_px`` px and ``window`` rows. Stretches of at
    least ``min_run`` samples with none are taken as hidden. The covering trace is drawn into the
    working image there (``half_width`` px either side) only so the trace stays connected through
    them; it is not a reading of this curve. Pass the intervals to
    :meth:`AssistedSession.set_hidden_interval`, which writes those depths as null (the same rule as
    a curve clipped off the track) and lists each one as a tricky spot in Trace and fix.

    Returns ``(new_ink, hidden, intervals)``: ``hidden`` a boolean per trace sample, ``intervals``
    the hidden stretches as (first, last) sheet rows. ``ink`` is not changed."""
    import cv2
    names = sug["names"]
    if style not in names:
        raise ValueError(f"style must be in {names[2:]}, got {style!r}")
    own = (sug["labels"] == names.index(style)).astype(np.int32)
    H, W = own.shape
    ii = np.zeros((H + 1, W + 1), np.int64)
    ii[1:, 1:] = own.cumsum(0).cumsum(1)
    rows = np.asarray(rows, float)
    xs = np.asarray(xs, float)
    ok = np.isfinite(rows) & np.isfinite(xs)
    near = max(1, int(round(near_px * px_scale)))
    win = max(0, int(round(window * px_scale)))
    hidden = np.zeros(len(rows), bool)
    r = np.clip(np.round(np.where(ok, rows, 0)).astype(int), 0, H - 1)
    x = np.clip(np.round(np.where(ok, xs, 0)).astype(int), 0, W - 1)
    a, b = np.clip(r - win, 0, H), np.clip(r + win + 1, 0, H)
    c, d = np.clip(x - near, 0, W), np.clip(x + near + 1, 0, W)
    cnt = ii[b, d] - ii[a, d] - ii[b, c] + ii[a, c]
    empty = ok & (cnt == 0)
    canvas = np.zeros((H, W), np.uint8)
    hw = max(1, int(round(half_width * px_scale)))
    intervals = []
    edges = np.flatnonzero(np.diff(np.r_[0, empty.astype(np.int8), 0]))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        if s1 - s0 < max(2, min_run):
            continue
        hidden[s0:s1] = True
        intervals.append((float(rows[s0]), float(rows[s1 - 1])))
        seg = np.c_[xs[s0:s1], rows[s0:s1]]
        cv2.polylines(canvas, [np.round(seg).astype(np.int32).reshape(-1, 1, 2)], False, 1,
                      thickness=2 * hw + 1)
    return np.asarray(ink, bool) | canvas.astype(bool), hidden, intervals


def make_style_base(ink: np.ndarray, sug: dict) -> np.ndarray:
    """Black base for the working images: the sheet ink minus all curve ink inside the tracks
    (styled or not). Grid, borders, depth numbers and header stay, so each working image still
    has its layout, and no other curve is left for the tracer to follow."""
    return np.asarray(ink, bool) & ~sug["curve_ink"] & ~(sug["labels"] >= 2)


def make_style_ink(base: np.ndarray, sug: dict, style: str, x_range, body_rows=None,
                   pattern: str | None = None, **kw) -> np.ndarray:
    """Working image for ONE style: the base plus this style's pieces inside the track, dash gaps
    closed (``GAP_CLOSE_PX`` by pattern) and the stroke thickened. Same function as the colour path
    (:func:`pylogdigitizer.curve_ink.make_curve_ink`)."""
    from pylogdigitizer.curve_ink import make_curve_ink
    pat = pattern or style.split(" ")[0]
    kw.setdefault("gap_close_px", GAP_CLOSE_PX.get(pat, 9))
    kw.setdefault("thicken_px", 2)
    return make_curve_ink(base, sug["labels"], sug["names"], style, x_range, body_rows, **kw)


def legend_pattern(entry: dict) -> str:
    """Pattern of one header legend sample (:func:`header_ocr.legend_styles`)."""
    if not entry.get("dashed"):
        return "solid"
    p = float(entry.get("dash_period") or 0.0)
    return "dotted" if 0 < p <= 14 or entry.get("ink", 1.0) < 0.45 else "dashed"


def match_legend(styles: list[dict], legend: list[dict]) -> list[int | None]:
    """For each legend entry (top to bottom), the index of the proposed style with the same
    pattern (and closest stroke width), or None. A style is used at most once."""
    used, out = set(), []
    for e in legend:
        pat = legend_pattern(e)
        cand = [i for i, s in enumerate(styles) if s["pattern"] == pat and i not in used]
        if not cand:
            out.append(None)
            continue
        i = min(cand, key=lambda i: abs(styles[i]["width_px"] - float(e.get("width", 0))))
        used.add(i); out.append(i)
    return out
