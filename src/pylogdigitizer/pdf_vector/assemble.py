# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Classify grid vs curve strokes, map x→value / y→depth, assemble dense traces (no smoothing)."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from pylogdigitizer.depth_calibration import DepthMap
from pylogdigitizer.scale_calibration import ScaleMap
from pylogdigitizer.pdf_vector.extract import PageVectors, Stroke
from pylogdigitizer.pdf_vector.calibrate import (
    CurveHeader,
    TrackFrame,
    colour_key,
    is_near_black,
    is_near_gray,
    is_near_white,
    depth_map_from_labels,
    detect_depth_axis,
    parse_curve_headers,
    legend_styles_from_strokes,
    tracks_from_grids,
    style_distance,
)


@dataclass
class CurveTrace:
    """Dense connected points for one curve (no smoothing)."""
    mnemonic: str
    depth: np.ndarray
    value: np.ndarray
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    unit: str = ""
    kind: str = "linear"
    page_index: int | None = None
    info: dict = field(default_factory=dict)


def classify_strokes(strokes: list[Stroke]) -> tuple[list[Stroke], list[Stroke], list[Stroke]]:
    """Return (grid_h, grid_v, body_curve_strokes). Curves keep full style (colour/width/dash)."""
    grid_h: list[Stroke] = []
    grid_v: list[Stroke] = []
    curves: list[Stroke] = []
    for s in strokes:
        if len(s.pts) < 2:
            continue
        dx = float(np.ptp(s.pts[:, 0]))
        dy = float(np.ptp(s.pts[:, 1]))
        if is_near_black(s.color) and dy < 1.5 and dx > 30 and not s.dash:
            grid_h.append(s)
        elif is_near_black(s.color) and dx < 1.5 and dy > 30 and not s.dash:
            grid_v.append(s)
        elif is_near_white(s.color):
            if len(s.pts) >= 8 and dy > 30:
                curves.append(s)
        elif is_near_gray(s.color) and not s.dash and s.width <= 1.5 and dy < 8:
            continue
        elif (not is_near_black(s.color) or bool(s.dash)) and len(s.pts) >= 5 and dy > 20:
            curves.append(s)
        elif is_near_black(s.color) and not s.dash and s.width >= 2.5 and dy > 40 and dx > 1.5:
            curves.append(s)
    return grid_h, grid_v, curves


def _points_to_series(
    pts: np.ndarray,
    depth_map: DepthMap,
    scale: ScaleMap,
    clip_tol: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    rows = -pts[:, 1]
    depth = np.asarray(depth_map(rows), float)
    value = np.asarray(scale(pts[:, 0]), float)
    lo, hi = (scale.x_left, scale.x_right) if scale.x_left < scale.x_right else (scale.x_right, scale.x_left)
    clipped = (pts[:, 0] <= lo + clip_tol) | (pts[:, 0] >= hi - clip_tol)
    value = value.copy()
    value[clipped] = np.nan
    order = np.argsort(depth)
    return depth[order], value[order]


def unwrap_log_wraps(
    pts: np.ndarray,
    scale: ScaleMap,
    edge_tol: float = 2.0,
) -> np.ndarray:
    """For log tracks: when a stroke wraps right→left (or left→right), shift by one decade.

    Returns a value array with decade offsets applied; edge samples stay NaN (clipped).
    Assumes the stroke is ordered roughly by depth (descending PDF y).
    """
    if scale.kind != "log" or scale.v_left <= 0 or scale.v_right <= 0:
        return np.asarray(scale(pts[:, 0]), float)

    lo, hi = (scale.x_left, scale.x_right) if scale.x_left < scale.x_right else (scale.x_right, scale.x_left)
    span = max(hi - lo, 1e-9)
    decade = scale.v_right / scale.v_left
    if decade <= 1:
        decade = scale.v_left / scale.v_right

    # Order by descending y (increasing depth on typical plots).
    order = np.argsort(-pts[:, 1])
    xs = pts[order, 0]
    base = np.asarray(scale(xs), float)
    edge_l = xs <= lo + edge_tol
    edge_r = xs >= hi - edge_tol
    wrap_offset = 0
    out = base.copy()
    for i in range(len(xs)):
        if i > 0:
            dx = xs[i] - xs[i - 1]
            # Jump from near-right to near-left → +1 decade wrap.
            if edge_r[i - 1] and edge_l[i] and dx < -0.4 * span:
                wrap_offset += 1
            elif edge_l[i - 1] and edge_r[i] and dx > 0.4 * span:
                wrap_offset -= 1
        if edge_l[i] or edge_r[i]:
            out[i] = np.nan
        else:
            out[i] = base[i] * (decade ** wrap_offset)
    # Restore original point order.
    restored = np.empty_like(out)
    restored[order] = out
    return restored



def unwrap_continuity(
    depth: np.ndarray,
    value: np.ndarray,
    v_left: float,
    v_right: float,
    kind: str = "linear",
) -> np.ndarray:
    """Resolve linear (or log) scale wraps by depth continuity — no fixed offsets.

    Walks deep→shallow. At each sample chooses wrap count k so the value stays
    continuous with the deeper neighbour. Scale span comes from the header
    (``|v_right - v_left|`` or the log decade ratio). Deepest sample prefers an
    in-scale reading when one exists.
    """
    d = np.asarray(depth, float)
    v = np.asarray(value, float)
    out = v.copy()
    finite = np.isfinite(d) & np.isfinite(v)
    if finite.sum() < 2:
        return out
    lo, hi = (min(v_left, v_right), max(v_left, v_right))
    order = np.argsort(d)[::-1]  # deep → shallow
    order = order[finite[order]]

    if kind == "log":
        if v_left <= 0 or v_right <= 0:
            return out
        decade = hi / lo if hi > lo else lo / hi
        if decade <= 1:
            return out

        def candidates(raw: float):
            return [raw * (decade ** k) for k in range(-3, 4)]

        def in_scale(x: float) -> bool:
            return lo / (decade ** 0.05) <= x <= hi * (decade ** 0.05)
    else:
        span = abs(v_right - v_left)
        if span < 1e-9:
            return out

        def candidates(raw: float):
            return [raw + k * span for k in range(-3, 4)]

        def in_scale(x: float) -> bool:
            return lo - 0.02 * span <= x <= hi + 0.02 * span

    # Scale span for jump penalty (linear only reaches here with a finite span).
    span = abs(v_right - v_left) if kind != "log" else None
    prev = None
    for i in order:
        cands = candidates(float(v[i]))
        if prev is None:
            in_sc = [c for c in cands if in_scale(c)]
            pick = min(in_sc or cands, key=lambda c: abs(c - 0.5 * (lo + hi)))
        else:
            def cost(c: float, _p=prev) -> float:
                # Walking deep→shallow: a sudden rise of ~1 scale span is a wrap,
                # not a real jump (mud-temp / tension wraps).
                d = abs(c - _p)
                if span is not None and c > _p + 0.3 * span:
                    d += 0.5 * span
                return d
            pick = min(cands, key=cost)
        out[i] = pick
        prev = pick
    return out


def _points_to_series_log_wrap(
    pts: np.ndarray,
    depth_map: DepthMap,
    scale: ScaleMap,
    clip_tol: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    rows = -pts[:, 1]
    depth = np.asarray(depth_map(rows), float)
    if scale.kind == "log":
        value = unwrap_log_wraps(pts, scale, edge_tol=clip_tol)
    else:
        value = np.asarray(scale(pts[:, 0]), float)
        lo, hi = (scale.x_left, scale.x_right) if scale.x_left < scale.x_right else (scale.x_right, scale.x_left)
        clipped = (pts[:, 0] <= lo + clip_tol) | (pts[:, 0] >= hi - clip_tol)
        value = value.copy()
        value[clipped] = np.nan
    order = np.argsort(depth)
    return depth[order], value[order]


def _body_segs(segs: list[Stroke]) -> list[Stroke]:
    return [s for s in segs if float(np.ptp(s.pts[:, 1])) > 40 and len(s.pts) >= 20]


def _frame_for_header(h: CurveHeader, frames: list[TrackFrame], page_width: float) -> TrackFrame:
    """Pick a plausible track for this header (prefer narrow frames over page-wide).

    Scale labels sit near the *left* of their track, so prefer the frame whose
    left edge is closest to (and ≤) the header x.
    """
    containing = [
        fr for fr in frames
        if fr.x_left - 40 <= h.x <= fr.x_right + 10
        or any(c.mnemonic == h.mnemonic for c in fr.curves)
    ]
    narrow = [fr for fr in containing if (fr.x_right - fr.x_left) < 400]
    if narrow:
        # Prefer frames where header sits near the left edge (typical scale text).
        def score(fr: TrackFrame) -> tuple:
            left_dist = h.x - fr.x_left
            # Good if header is just inside from the left (0..40 pt).
            left_fit = abs(left_dist - 5.0) if -20 <= left_dist <= 50 else 200 + abs(left_dist)
            return (left_fit, fr.x_right - fr.x_left)
        return min(narrow, key=score)
    if containing:
        # Split a giant frame: left headers → [x_left, mid], right → [mid, x_right].
        fr = min(containing, key=lambda fr: fr.x_right - fr.x_left)
        if (fr.x_right - fr.x_left) > 400:
            # Use nearest v-edge from other frames if any; else 180-pt default.
            edges = sorted({f.x_left for f in frames} | {f.x_right for f in frames})
            right_edge = next((e for e in edges if e > h.x + 30), None)
            if right_edge and right_edge - max(fr.x_left, h.x - 5) < 400:
                return TrackFrame(x_left=max(fr.x_left, 0.0), x_right=float(right_edge))
            x0 = max(fr.x_left, h.x - 5) if h.x < 200 else h.x - 5
            return TrackFrame(x_left=x0, x_right=min(page_width, x0 + 180.0))
        return fr
    x0 = max(0.0, h.x - 5)
    return TrackFrame(x_left=x0, x_right=min(page_width, x0 + 180.0))


def _best_chain_for_header(
    h: CurveHeader,
    fr: TrackFrame,
    curves: list[Stroke],
    claimed: set[int],
) -> Stroke | None:
    """Pick the longest unclaimed body stroke in this track matching header style.

    White strokes with matching width/dash are accepted when the legend colour is
    black (Schlumberger draws some wrap/backup curves in white — e.g. AT20).
    Non-white matches still win when both exist.
    """
    candidates: list[tuple[float, Stroke]] = []
    for s in _body_segs(curves):
        if id(s) in claimed:
            continue
        x_med = float(np.median(s.pts[:, 0]))
        if not (fr.x_left - 5 <= x_med <= fr.x_right + 5):
            continue
        white = is_near_white(s.color)
        if white:
            # White wrap/backup is used on log resistivity tracks (e.g. AT20).
            if h.kind != "log" or h.color is None or not is_near_black(h.color):
                continue
            if h.dash is not None and bool(s.dash) != bool(h.dash):
                continue
            if h.width is not None and abs(s.width - h.width) > 1.5:
                continue
            if not s.dash and s.width < 2.5:
                continue
        else:
            if h.color is not None and not is_near_black(h.color):
                if sum((float(a) - float(b)) ** 2 for a, b in zip(s.color, h.color)) > 0.05:
                    continue
            elif h.color is not None and is_near_black(h.color):
                if not is_near_black(s.color):
                    continue
            if h.dash is not None and bool(s.dash) != bool(h.dash):
                continue
            if h.width is not None and abs(s.width - h.width) > 1.5:
                continue
        dist = style_distance(
            s.color if not white else (0.0, 0.0, 0.0), s.width, tuple(s.dash),
            h.color, h.width, h.dash,
        )
        dx = float(np.ptp(s.pts[:, 0]))
        # TENS (linear, black dash) is a narrow strip; AT20 (log, same dash) spans the track.
        if h.kind == "linear" and h.dash and dx > 80:
            continue
        if h.kind == "log" and h.dash and not white and dx < 40:
            continue
        score = float(len(s.pts))
        score += 0.5 * dx
        score += 5000.0 * (1.0 / (1.0 + dist))
        if h.width is not None:
            score += 200.0 * (1.0 - min(abs(s.width - h.width) / 5.0, 1.0))
        # Prefer strokes near the header's scale column (TENS vs AT* share black dash).
        score -= 2.5 * abs(x_med - h.x)
        if white:
            score -= 500.0  # prefer inked stroke when both match
        candidates.append((score, s))
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    return candidates[0][1]


def _apply_explicit_styles(
    headers: list[CurveHeader],
    mnemonic_colors: dict[str, tuple[float, float, float]] | None,
    mnemonic_styles: dict[str, dict] | None,
) -> None:
    """Apply explicit overrides (fallback). Styles win over colour-only."""
    if mnemonic_styles:
        for h in headers:
            st = mnemonic_styles.get(h.mnemonic)
            if not st:
                continue
            if "color" in st:
                h.color = tuple(float(c) for c in st["color"])
            if "width" in st:
                h.width = float(st["width"])
            if "dash" in st:
                h.dash = tuple(float(x) for x in st["dash"])
    if mnemonic_colors:
        for h in headers:
            if h.mnemonic in mnemonic_colors and h.color is None:
                h.color = tuple(float(c) for c in mnemonic_colors[h.mnemonic])



def _hgrid_ys(strokes: list[Stroke], min_dx: float = 200.0) -> list[float]:
    """Unique y positions of long near-horizontal grid lines (track edges)."""
    ys: list[float] = []
    for s in strokes:
        if len(s.pts) < 2:
            continue
        dx = float(np.ptp(s.pts[:, 0]))
        dy = float(np.ptp(s.pts[:, 1]))
        if dx >= min_dx and dy < 2.0 and is_near_black(s.color) and not s.dash:
            ys.append(float(np.median(s.pts[:, 1])))
    if not ys:
        return []
    ys = sorted(ys)
    out = [ys[0]]
    for y in ys[1:]:
        if y - out[-1] > 3.0:
            out.append(y)
    return out


def _horizontal_curve_strokes(strokes: list[Stroke]) -> list[Stroke]:
    """Long polylines that run with depth (x) across a track band (small dy)."""
    out: list[Stroke] = []
    for s in strokes:
        if len(s.pts) < 40:
            continue
        dx = float(np.ptp(s.pts[:, 0]))
        dy = float(np.ptp(s.pts[:, 1]))
        if dx < 80:
            continue
        if dy > 0.6 * dx:  # not a horizontal track curve
            continue
        # Skip white/grey fills; pure black ink curves are kept (is_near_gray
        # also matches black, so require non-black before dropping).
        if is_near_white(s.color):
            continue
        if is_near_gray(s.color) and not is_near_black(s.color):
            continue
        out.append(s)
    return out


def _track_for_y(y: float, bounds: list[float]) -> tuple[float, float] | None:
    if len(bounds) < 2:
        return None
    for i in range(len(bounds) - 1):
        y0, y1 = bounds[i], bounds[i + 1]
        if y0 - 2 <= y <= y1 + 2:
            return y0, y1
    # nearest band centre
    mids = [0.5 * (bounds[i] + bounds[i + 1]) for i in range(len(bounds) - 1)]
    k = int(np.argmin([abs(y - m) for m in mids]))
    return bounds[k], bounds[k + 1]


def traces_from_page_horizontal(
    page: PageVectors,
    depth_map: DepthMap | None = None,
    headers: list[CurveHeader] | None = None,
) -> list[CurveTrace]:
    """Assemble traces for horizontal digital prints (depth along +x, value along +y).

    USGS GeoLog Locator PDFs stack tracks vertically; each track's lower/upper grid
    line is ``v_left`` / ``v_right``. Curve identity is by y-band proximity to the
    scale label (these prints are typically monochrome — no colour legend).
    """
    if depth_map is None:
        depth_map = depth_map_from_labels(page.texts, axis="x")
    if depth_map is None:
        return []
    page_headers = parse_curve_headers(page.texts)
    if headers is None:
        headers = page_headers
    else:
        have = {h.mnemonic for h in headers}
        headers = list(headers) + [h for h in page_headers if h.mnemonic not in have]
    bounds = _hgrid_ys(page.strokes)
    curves = _horizontal_curve_strokes(page.strokes)
    if not bounds or not curves:
        return []

    skip = {"TIME_1900", "VDL", "MAIN", "REPEAT", "OPTICAL_IMAGE", "ACOUSTIC_IMAGE",
            "GAMMA_IMAGE", "DIP_IN_DEGREES", "WATER_LEVEL", "TRANS"}
    # Group headers by track band; within a band match left→right headers to
    # bottom→top strokes (USGS amb/pmp share a track).
    by_band: dict[tuple[float, float], list[CurveHeader]] = defaultdict(list)
    for h in headers:
        if h.mnemonic in skip:
            continue
        band = _track_for_y(h.y, bounds)
        if band is None:
            continue
        y0, y1 = band
        if not (y0 - 15 <= h.y <= y1 + 5):
            band = _track_for_y(h.y + 20, bounds) or band
        by_band[band].append(h)

    claimed: set[int] = set()
    out: list[CurveTrace] = []
    for (y0, y1), band_headers in by_band.items():
        band_strokes: list[Stroke] = []
        for s in curves:
            if id(s) in claimed:
                continue
            y_med = float(np.median(s.pts[:, 1]))
            if y0 - 2 <= y_med <= y1 + 2:
                band_strokes.append(s)
        if not band_strokes:
            continue
        # Left header → lower-mean-y stroke (smaller values when v_left at bottom).
        hs = sorted(band_headers, key=lambda h: h.x)
        ss = sorted(band_strokes, key=lambda s: float(np.median(s.pts[:, 1])))
        for h, chain in zip(hs, ss):
            claimed.add(id(chain))
            scale = ScaleMap(y0, y1, h.v_left, h.v_right, h.kind)
            depth = np.asarray(depth_map(chain.pts[:, 0]), float)
            value = np.asarray(scale(chain.pts[:, 1]), float)
            lo, hi = ((scale.x_left, scale.x_right)
                      if scale.x_left < scale.x_right else (scale.x_right, scale.x_left))
            clip_tol = 1.5
            clipped = (chain.pts[:, 1] <= lo + clip_tol) | (chain.pts[:, 1] >= hi - clip_tol)
            value = value.copy()
            value[clipped] = np.nan
            order = np.argsort(depth)
            depth, value = depth[order], value[order]
            if len(depth) >= 2:
                _, uniq_idx = np.unique(np.round(depth, 6), return_index=True)
                uniq_idx.sort()
                depth, value = depth[uniq_idx], value[uniq_idx]
            out.append(CurveTrace(
                mnemonic=h.mnemonic, depth=depth, value=value,
                color=tuple(float(c) for c in colour_key(chain.color)[:3]),
                unit=h.unit, kind=h.kind, page_index=page.page_index,
                info={"n_pts": int(len(depth)), "width": chain.width, "dash": chain.dash,
                      "track": (y0, y1), "layout": "horizontal",
                      "v_left": h.v_left, "v_right": h.v_right},
            ))
        # Leftover strokes in band with no header: still emit as anonymous curves.
        for chain in ss[len(hs):]:
            if id(chain) in claimed:
                continue
            claimed.add(id(chain))
    return out


def traces_from_page(
    page: PageVectors,
    depth_map: DepthMap | None = None,
    frames: list[TrackFrame] | None = None,
    headers: list[CurveHeader] | None = None,
    mnemonic_colors: dict[str, tuple[float, float, float]] | None = None,
    mnemonic_styles: dict[str, dict] | None = None,
    unwrap_log: bool = True,
) -> list[CurveTrace]:
    """Build one CurveTrace per curve header that has a matching stroke on this page.

    Style identity (colour + width + dash) comes from the page legend by default.
    ``mnemonic_colors`` / ``mnemonic_styles`` remain as explicit overrides.

    Horizontal digital prints (USGS GeoLog Locator style) are routed to
    :func:`traces_from_page_horizontal` when depth ticks run along +x.
    """
    axis = detect_depth_axis(page.texts)
    if axis == "x":
        return traces_from_page_horizontal(page, depth_map=depth_map, headers=headers)

    grid_h, grid_v, curves = classify_strokes(page.strokes)
    if depth_map is None:
        depth_map = depth_map_from_labels(page.texts, axis="y")
    if depth_map is None:
        return []
    page_headers = parse_curve_headers(page.texts)
    if headers is None:
        headers = page_headers
    else:
        have = {h.mnemonic for h in headers}
        headers = list(headers) + [h for h in page_headers if h.mnemonic not in have]

    _apply_explicit_styles(headers, mnemonic_colors, mnemonic_styles)
    # Legend fills styles for headers still missing them.
    need = [h for h in headers if h.color is None]
    if need and sum(1 for h in headers if h.color is not None) < 3:
        got = legend_styles_from_strokes(list(need), page.strokes)
        if len(got) < 3:
            for h in need:
                if h.mnemonic in got:
                    h.color = h.width = h.dash = None
    elif need and sum(1 for h in headers if h.color is not None) >= 3:
        pass  # already have a legend; do not false-match on body pages

    if frames is None:
        vxs = [float(np.mean(s.pts[:, 0])) for s in grid_v]
        hspans = []
        for s in grid_h:
            x0 = float(s.pts[:, 0].min())
            x1 = float(s.pts[:, 0].max())
            vxs.append(x0)
            vxs.append(x1)
            hspans.append((x0, x1))
        vxs.append(0.0)
        depth_x = depth_map.info.get("x_col")
        frames = tracks_from_grids(
            vxs, headers, page.width, depth_x=depth_x, hgrid_spans=hspans,
        )

    skip = {"TIME_1900", "VDL", "MAIN", "REPEAT"}
    claimed: set[int] = set()
    out: list[CurveTrace] = []
    ordered = sorted(
        headers,
        key=lambda h: (h.color is None, h.mnemonic in skip, -(h.width or 0), -h.y),
    )
    for h in ordered:
        if h.mnemonic in skip:
            continue
        fr = _frame_for_header(h, frames, page.width)
        chain = _best_chain_for_header(h, fr, curves, claimed)
        if chain is None:
            continue
        claimed.add(id(chain))
        scale = ScaleMap(fr.x_left, fr.x_right, h.v_left, h.v_right, h.kind)
        if unwrap_log and h.kind == "log":
            depth, value = _points_to_series_log_wrap(chain.pts, depth_map, scale)
        else:
            depth, value = _points_to_series(chain.pts, depth_map, scale)
        if len(depth) >= 2:
            _, uniq_idx = np.unique(np.round(depth, 6), return_index=True)
            uniq_idx.sort()
            depth, value = depth[uniq_idx], value[uniq_idx]
        col = colour_key(chain.color)
        out.append(CurveTrace(
            mnemonic=h.mnemonic, depth=depth, value=value,
            color=tuple(float(c) for c in col[:3]),
            unit=h.unit, kind=h.kind, page_index=page.page_index,
            info={"n_pts": int(len(depth)), "width": chain.width, "dash": chain.dash,
                  "track": (fr.x_left, fr.x_right),
                  "v_left": h.v_left, "v_right": h.v_right,
                  "legend_style": {"color": h.color, "width": h.width, "dash": h.dash}},
        ))
    return out


def stitch_traces(traces: list[CurveTrace]) -> list[CurveTrace]:
    by: dict[str, list[CurveTrace]] = defaultdict(list)
    for t in traces:
        by[t.mnemonic].append(t)
    out: list[CurveTrace] = []
    for mnem, group in by.items():
        d = np.concatenate([t.depth for t in group])
        v = np.concatenate([t.value for t in group])
        order = np.argsort(d)
        d, v = d[order], v[order]
        _, idx = np.unique(np.round(d, 5), return_index=True)
        idx.sort()
        d, v = d[idx], v[idx]
        t0 = group[0]
        v_left = t0.info.get("v_left")
        v_right = t0.info.get("v_right")
        if v_left is None:
            for g in group:
                if "v_left" in g.info:
                    v_left, v_right = g.info["v_left"], g.info["v_right"]
                    break
        if v_left is not None and v_right is not None and t0.kind == "linear":
            v = unwrap_continuity(d, v, float(v_left), float(v_right), kind="linear")
        out.append(CurveTrace(
            mnemonic=mnem, depth=d, value=v, color=t0.color, unit=t0.unit,
            kind=t0.kind,
            info={"n_pages": len(group), "n_pts": int(len(d)),
                  "v_left": v_left, "v_right": v_right},
        ))
    return out


def apply_clip_nulls(value: np.ndarray, scale: ScaleMap, x: np.ndarray | None = None,
                     tol_frac: float = 0.01) -> np.ndarray:
    v = np.asarray(value, float).copy()
    if x is not None:
        lo, hi = (scale.x_left, scale.x_right) if scale.x_left < scale.x_right else (scale.x_right, scale.x_left)
        span = max(hi - lo, 1e-9)
        edge = (x <= lo + tol_frac * span) | (x >= hi - tol_frac * span)
        v[edge] = np.nan
        return v
    lo, hi = sorted((scale.v_left, scale.v_right))
    if scale.kind == "log":
        edge = (v <= lo * 1.02) | (v >= hi / 1.02)
    else:
        span = hi - lo if hi != lo else 1.0
        edge = (v <= lo + tol_frac * abs(span)) | (v >= hi - tol_frac * abs(span))
    v[edge] = np.nan
    return v
