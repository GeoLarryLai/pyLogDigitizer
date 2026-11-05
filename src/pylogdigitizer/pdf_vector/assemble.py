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
    legend_colors_from_strokes,
    tracks_from_grids,
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


def classify_strokes(strokes: list[Stroke]) -> tuple[list[Stroke], list[Stroke], dict[tuple, list[Stroke]]]:
    grid_h: list[Stroke] = []
    grid_v: list[Stroke] = []
    curves: dict[tuple, list[Stroke]] = defaultdict(list)
    for s in strokes:
        if len(s.pts) < 2:
            continue
        dx = float(np.ptp(s.pts[:, 0]))
        dy = float(np.ptp(s.pts[:, 1]))
        col = colour_key(s.color)
        if is_near_black(s.color) and dy < 1.5 and dx > 30 and not s.dash:
            grid_h.append(s)
        elif is_near_black(s.color) and dx < 1.5 and dy > 30 and not s.dash:
            grid_v.append(s)
        elif is_near_white(s.color):
            if len(s.pts) >= 8 and dy > 30:
                curves[col].append(s)
        elif is_near_gray(s.color) and not s.dash and s.width <= 1.5 and dy < 8:
            continue
        elif (not is_near_black(s.color) or bool(s.dash)) and len(s.pts) >= 5 and dy > 20:
            curves[col].append(s)
        elif is_near_black(s.color) and not s.dash and s.width >= 2.5 and dy > 40 and dx > 5:
            curves[col].append(s)
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


def _color_dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def _body_segs(segs: list[Stroke]) -> list[Stroke]:
    return [s for s in segs if float(np.ptp(s.pts[:, 1])) > 40 and len(s.pts) >= 20]


def _frame_for_header(h: CurveHeader, frames: list[TrackFrame], page_width: float) -> TrackFrame:
    for fr in frames:
        if h in fr.curves or any(c.mnemonic == h.mnemonic for c in fr.curves):
            return fr
        if fr.x_left - 40 <= h.x <= fr.x_right + 10:
            return fr
    # Fabricate from header x: Schlumberger scale text sits near the left of its track.
    # Guess track width ~180 pt (common on these plots).
    width = 180.0
    x0 = max(0.0, h.x - 5)
    # If header is mid-page, track starts near header x.
    if h.x > 150:
        x0 = min([fr.x_left for fr in frames if fr.x_left >= 150], default=h.x - 5)
        # Prefer known right-track left edge from frames.
        rights = [fr for fr in frames if fr.x_left >= 150]
        if rights:
            return min(rights, key=lambda fr: abs(fr.x_left - h.x))
    return TrackFrame(x_left=x0, x_right=min(page_width, x0 + width))


def _best_chain_for_header(
    h: CurveHeader,
    fr: TrackFrame,
    curves: dict[tuple, list[Stroke]],
    claimed: set[int],
) -> Stroke | None:
    """Pick the longest unclaimed body stroke in this track matching the header colour."""
    want = colour_key(h.color) if h.color is not None else None
    candidates: list[tuple[float, Stroke]] = []
    for col, segs in curves.items():
        if is_near_white(col):
            continue
        if want is not None and not is_near_black(want):
            if _color_dist(col, want) > 0.05:
                continue
        elif want is not None and is_near_black(want):
            if not is_near_black(col):
                continue
        elif want is None and is_near_black(col):
            # Uncoloured header: allow black only for curves typically drawn black.
            pass
        elif want is None and not is_near_black(col):
            # No colour hint — accept any non-white in track (weaker).
            pass
        for s in _body_segs(segs):
            if id(s) in claimed:
                continue
            x_med = float(np.median(s.pts[:, 0]))
            if not (fr.x_left - 5 <= x_med <= fr.x_right + 5):
                continue
            # Score: prefer matching colour, dashed black for backup styles, long chains, x span.
            score = float(len(s.pts))
            score += 0.5 * float(np.ptp(s.pts[:, 0]))
            if want is not None:
                score += 1000.0 * (1.0 - min(_color_dist(col, want), 1.0))
            if is_near_black(col) and s.dash:
                score += 50.0
            candidates.append((score, s))
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    return candidates[0][1]


def traces_from_page(
    page: PageVectors,
    depth_map: DepthMap | None = None,
    frames: list[TrackFrame] | None = None,
    headers: list[CurveHeader] | None = None,
    mnemonic_colors: dict[str, tuple[float, float, float]] | None = None,
) -> list[CurveTrace]:
    """Build one CurveTrace per curve header that has a matching stroke on this page."""
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

    # Apply explicit colours first; legend fills gaps without stealing explicit colours.
    if mnemonic_colors:
        for h in headers:
            if h.mnemonic in mnemonic_colors:
                h.color = tuple(float(c) for c in mnemonic_colors[h.mnemonic])
    legend_colors_from_strokes([h for h in headers if h.color is None], page.strokes)

    if frames is None:
        vxs = [float(np.mean(s.pts[:, 0])) for s in grid_v]
        for s in grid_h:
            vxs.append(float(s.pts[:, 0].min()))
            vxs.append(float(s.pts[:, 0].max()))
        vxs.append(0.0)
        depth_x = depth_map.info.get("x_col")
        frames = tracks_from_grids(vxs, headers, page.width, depth_x=depth_x)

    # Known curve names we care about — skip bogus header parses (TIME_1900 etc.).
    skip = {"TIME_1900", "VDL", "MAIN", "REPEAT"}
    claimed: set[int] = set()
    out: list[CurveTrace] = []
    # Prefer headers that have an explicit / legend colour.
    ordered = sorted(headers, key=lambda h: (h.color is None, h.mnemonic in skip, -h.y))
    for h in ordered:
        if h.mnemonic in skip:
            continue
        fr = _frame_for_header(h, frames, page.width)
        chain = _best_chain_for_header(h, fr, curves, claimed)
        if chain is None:
            continue
        claimed.add(id(chain))
        scale = ScaleMap(fr.x_left, fr.x_right, h.v_left, h.v_right, h.kind)
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
                  "track": (fr.x_left, fr.x_right)},
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
