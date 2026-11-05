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
        elif is_near_black(s.color) and not s.dash and s.width >= 2.5 and dy > 40 and dx > 5:
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
    """Pick a plausible track for this header (prefer narrow frames over page-wide)."""
    containing = [
        fr for fr in frames
        if fr.x_left - 40 <= h.x <= fr.x_right + 10
        or any(c.mnemonic == h.mnemonic for c in fr.curves)
    ]
    # Prefer tracks of typical width (<280 pt). Giant frames are last resort.
    narrow = [fr for fr in containing if (fr.x_right - fr.x_left) < 400]
    if narrow:
        return min(narrow, key=lambda fr: abs((fr.x_left + fr.x_right) / 2 - h.x))
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
    """Pick the longest unclaimed body stroke in this track matching header style."""
    candidates: list[tuple[float, Stroke]] = []
    for s in _body_segs(curves):
        if id(s) in claimed:
            continue
        if is_near_white(s.color):
            continue
        x_med = float(np.median(s.pts[:, 0]))
        if not (fr.x_left - 5 <= x_med <= fr.x_right + 5):
            continue
        # Hard filters when legend style is known — length must not override style.
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
            s.color, s.width, tuple(s.dash),
            h.color, h.width, h.dash,
        )
        score = float(len(s.pts))
        score += 0.5 * float(np.ptp(s.pts[:, 0]))
        score += 5000.0 * (1.0 / (1.0 + dist))
        if h.width is not None:
            score += 200.0 * (1.0 - min(abs(s.width - h.width) / 5.0, 1.0))
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
    """
    grid_h, grid_v, curves = classify_strokes(page.strokes)
    if depth_map is None:
        depth_map = depth_map_from_labels(page.texts)
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
        t0 = group[0]
        out.append(CurveTrace(
            mnemonic=mnem, depth=d[idx], value=v[idx], color=t0.color, unit=t0.unit,
            kind=t0.kind, info={"n_pages": len(group), "n_pts": int(len(idx))},
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
