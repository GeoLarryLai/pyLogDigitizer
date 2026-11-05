# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Depth and scale calibration from PDF text + legend styles (shared DepthMap / ScaleMap)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from pylogdigitizer.depth_calibration import DepthMap
from pylogdigitizer.scale_calibration import ScaleMap, linear_scale, log_scale
from pylogdigitizer.pdf_vector.extract import TextItem, Stroke


_DEPTH_RE = re.compile(r"^(\d{2,5})$")
_SCALE_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+([A-Za-z0-9./%°°-]+)\s*$"
)
_CURVE_PAREN_RE = re.compile(r"\(([A-Z][A-Z0-9_]{1,15})\)")
_CURVE_TOKEN_RE = re.compile(r"\b([A-Z]{2,}[A-Z0-9_]{0,12})\b")
_SKIP_TITLE = {
    "ARRAY", "INDUCTION", "TWO", "FOOT", "RESISTIVITY", "GAMMA", "RAY",
    "CABLE", "TENSION", "MUD", "TEMPERATURE", "TRANSIT", "TIME", "FOR",
    "AMPLITUDE", "SPONTANEOUS", "POTENTIAL", "MAIN", "REPEAT", "TO",
    "VARIABLE", "DENSITY", "LOG", "WITH", "LITHIOLOGY", "PLATFORM", "EXPRESS",
    "TOTAL", "COMPENSATED", "NEUTRON", "LITHO",
}


@dataclass
class CurveHeader:
    mnemonic: str
    v_left: float
    v_right: float
    unit: str
    kind: str  # 'linear' | 'log'
    x: float = 0.0
    y: float = 0.0
    descr: str = ""
    color: tuple[float, float, float] | None = None
    width: float | None = None
    dash: tuple[float, ...] | None = None


@dataclass
class TrackFrame:
    x_left: float
    x_right: float
    scale: ScaleMap | None = None
    curves: list[CurveHeader] = field(default_factory=list)


def parse_scale_label(text: str) -> tuple[float, float, str, str] | None:
    """Parse ``0 200 gAPI`` / ``0.2 2000 ohm.m`` → (v_left, v_right, unit, kind)."""
    m = _SCALE_RE.match(text.replace("\n", " ").strip())
    if not m:
        return None
    v0, v1 = float(m.group(1)), float(m.group(2))
    unit = m.group(3)
    unit_l = unit.lower().replace("°", "")
    span_ratio = max(abs(v0), abs(v1), 1e-12) / max(min(abs(v0), abs(v1)), 1e-12)
    logish = ("ohm" in unit_l) or (v0 > 0 and v1 > 0 and span_ratio >= 50)
    if logish and v0 > 0 and v1 > 0:
        return v0, v1, unit, "log"
    return v0, v1, unit, "linear"


def _depth_label_pairs(texts: list[TextItem]) -> list[tuple[float, float, float]]:
    out = []
    for t in texts:
        m = _DEPTH_RE.match(t.text.strip())
        if not m:
            continue
        d = float(m.group(1))
        if d < 50 or d > 50000:
            continue
        out.append((t.x, t.y, d))
    return out


def depth_map_from_labels(texts: list[TextItem], min_labels: int = 3) -> DepthMap | None:
    """DepthMap from numeric depth-column labels. Uses -y as row (PDF y up, depth down)."""
    pairs = _depth_label_pairs(texts)
    if len(pairs) < min_labels:
        return None
    xs = np.array([p[0] for p in pairs], float)
    x_med = float(np.median(xs))
    hist, edges = np.histogram(xs, bins=max(5, len(xs) // 2))
    if hist.max() >= min_labels:
        k = int(np.argmax(hist))
        x_med = 0.5 * (edges[k] + edges[k + 1])
    col = [p for p in pairs if abs(p[0] - x_med) < 25]
    if len(col) < min_labels:
        col = pairs
    col = sorted(col, key=lambda p: -p[1])
    ys = np.array([p[1] for p in col], float)
    ds = np.array([p[2] for p in col], float)
    keep = [0]
    for i in range(1, len(ds)):
        if ds[i] > ds[keep[-1]] + 0.05:
            keep.append(i)
    ys, ds = ys[keep], ds[keep]
    if len(ys) < min_labels:
        return None
    if len(ys) >= 5:
        coef = np.polyfit(-ys, ds, 1)
        pred = np.polyval(coef, -ys)
        resid = np.abs(ds - pred)
        thr = max(5.0, 3 * np.median(resid) * 1.4826)
        ok = resid <= thr
        if ok.sum() >= min_labels:
            ys, ds = ys[ok], ds[ok]
    rows = -ys
    order = np.argsort(rows)
    return DepthMap(rows=rows[order], depths=ds[order],
                    info={"n_labels": int(len(ds)), "x_col": x_med, "source": "pdf_text"})


def _mnemonic_from_title(text: str) -> tuple[str, str] | None:
    m = _CURVE_PAREN_RE.search(text)
    if m:
        return m.group(1), text
    toks = _CURVE_TOKEN_RE.findall(text)
    cand = [tok for tok in toks if tok not in _SKIP_TITLE and len(tok) >= 2]
    if not cand:
        return None
    return cand[0], text


def parse_curve_headers(texts: list[TextItem]) -> list[CurveHeader]:
    """Pair each curve title with the scale label immediately below it (same x band)."""
    items = sorted(texts, key=lambda t: (-t.y, t.x))
    headers: list[CurveHeader] = []
    used_scale_idx: set[int] = set()
    for i, t in enumerate(items):
        parsed = _mnemonic_from_title(t.text)
        if not parsed:
            continue
        if t.size and t.size < 9 and "(" not in t.text and " " not in t.text:
            continue
        mnem, title = parsed
        best = None
        for j in range(i + 1, min(len(items), i + 12)):
            if j in used_scale_idx:
                continue
            s = items[j]
            if t.y - s.y > 50 or s.y > t.y + 2:
                continue
            if abs(s.x - t.x) > 220:
                continue
            sc = parse_scale_label(s.text)
            if sc is None:
                continue
            dist = (t.y - s.y) + 0.05 * abs(s.x - t.x)
            if best is None or dist < best[0]:
                best = (dist, j, sc, s)
        if best is None:
            continue
        _, j, sc, s = best
        used_scale_idx.add(j)
        v0, v1, unit, kind = sc
        headers.append(CurveHeader(mnemonic=mnem, v_left=v0, v_right=v1, unit=unit,
                                   kind=kind, x=s.x, y=s.y, descr=title))
    seen: set[str] = set()
    out = []
    bad_prefix = ("TIME_", "DATE", "ONE", "LOG", "MAX_", "MIN_")
    for h in headers:
        if h.mnemonic in seen:
            continue
        if h.mnemonic.startswith(bad_prefix) or h.mnemonic in {"MAIN", "REPEAT", "TO"}:
            continue
        seen.add(h.mnemonic)
        out.append(h)
    return out


def legend_styles_from_strokes(
    headers: list[CurveHeader],
    strokes: list[Stroke],
) -> dict[str, dict]:
    """Map mnemonic → {color, width, dash} from horizontal legend swatches.

    Schlumberger (and similar) draw a coloured/dashed tick beside each curve's
    scale row. Matching uses (colour, width, dash), not colour alone — e.g. MTEM
    is solid black width 5 while TENS is dashed black width 3.
    """
    if not headers:
        return {}
    y_top = max(h.y for h in headers) + 50
    y_bot = min(h.y for h in headers) - 40
    swatches: list[tuple[float, float, tuple[float, float, float], float, tuple[float, ...]]] = []
    for s in strokes:
        if len(s.pts) < 2:
            continue
        rgb = (float(s.color[0]), float(s.color[1]), float(s.color[2]))
        if all(c > 0.9 for c in rgb):
            continue
        if max(rgb) - min(rgb) < 0.08 and min(rgb) > 0.4:
            continue
        dx = float(np.ptp(s.pts[:, 0]))
        dy = float(np.ptp(s.pts[:, 1]))
        cy = float(np.mean(s.pts[:, 1]))
        cx = float(np.mean(s.pts[:, 0]))
        if cy < y_bot or cy > y_top or dy > 6 or dx < 60:
            continue
        if s.width < 2.5 and not s.dash:
            continue
        swatches.append((cy, cx, rgb, float(s.width), tuple(float(x) for x in s.dash)))

    def style_key(rgb, w, dash):
        return (tuple(round(c, 2) for c in rgb), round(w, 1), tuple(round(d, 1) for d in dash))

    def same_side(sw_x: float, h_x: float) -> bool:
        # Left track headers (~x<200) vs right (~x>250).
        return (sw_x < 220) == (h_x < 220)

    # Score each (header, swatch) pair; greedily assign closest unused swatch
    # within a tight vertical window of the scale row.
    pairs: list[tuple[float, int, int]] = []  # (score, hi, si)
    for hi, h in enumerate(headers):
        if h.color is not None and h.width is not None:
            continue  # already styled
        target_y = h.y + 14.0
        for si, sw in enumerate(swatches):
            if not same_side(sw[1], h.x):
                continue
            dy = abs(sw[0] - target_y)
            if dy > 18:
                continue
            dx = abs(sw[1] - h.x)
            # Prefer distinctive styles slightly so plain black grid bars lose.
            distinct = (max(sw[2]) - min(sw[2]) > 0.1) or bool(sw[4]) or sw[3] >= 4.5
            score = dy + 0.02 * dx - (3.0 if distinct else 0.0)
            pairs.append((score, hi, si))
    pairs.sort()
    out: dict[str, dict] = {}
    used_h: set[int] = set()
    used_s: set[int] = set()
    used_style: set[tuple] = set()
    for score, hi, si in pairs:
        if hi in used_h or si in used_s:
            continue
        h = headers[hi]
        sw = swatches[si]
        key = style_key(sw[2], sw[3], sw[4])
        # Allow same style on opposite tracks (e.g. two black solids), but not twice
        # on the same track side.
        side = 0 if sw[1] < 220 else 1
        side_key = (key, side)
        if side_key in used_style:
            continue
        used_h.add(hi)
        used_s.add(si)
        used_style.add(side_key)
        style = {"color": sw[2], "width": sw[3], "dash": sw[4]}
        out[h.mnemonic] = style
        h.color = sw[2]
        h.width = sw[3]
        h.dash = sw[4]
    return out


# Back-compat alias
def legend_colors_from_strokes(headers, strokes):
    styles = legend_styles_from_strokes(headers, strokes)
    return {m: s["color"] for m, s in styles.items()}


def parse_well_header(texts: list[TextItem] | str) -> dict[str, str]:
    blob = texts if isinstance(texts, str) else "\n".join(t.text for t in texts)
    well: dict[str, str] = {}
    patterns = {
        "WELL": r"Well\s*:\s*([^\n|]+)",
        "COMP": r"Company\s*:\s*([^\n|]+)",
        "FLD": r"Field\s*:\s*([^\n|]+)",
        "CNTY": r"County\s*:\s*([^\n|,]+)",
        "DATE": r"(?:Creation\s*Date|Date)\s*:\s*([0-9]{1,2}[-/][A-Za-z]{3}[-/][0-9]{2,4}[^\n|]*)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, blob, re.I)
        if m:
            well[k] = m.group(1).strip()
    m = re.search(r"Lat\s*:\s*([-\d.]+)\s*,?\s*Long\s*:\s*([-\d.]+)", blob, re.I)
    if m:
        well["LOC"] = f"Lat {m.group(1)} Long {m.group(2)}"
    for key, pat in (
        ("EKB", r"K\.?B\.?\s*([-\d.]+)\s*ft"),
        ("EGL", r"G\.?L\.?\s*([-\d.]+)\s*ft"),
        ("EDF", r"D\.?F\.?\s*([-\d.]+)\s*ft"),
    ):
        m = re.search(pat, blob, re.I)
        if m:
            well[key] = m.group(1)
    return well


def tracks_from_grids(
    vgrid_xs: list[float],
    headers: list[CurveHeader],
    page_width: float,
    depth_x: float | None = None,
    hgrid_spans: list[tuple[float, float]] | None = None,
) -> list[TrackFrame]:
    """Build track frames from vertical grids and horizontal-grid spans.

    Horizontal depth-grid ticks usually span each track exactly (e.g. 0–180 and
    234–414). Those spans are the most reliable track bounds on Schlumberger PDFs.
    """
    spans = []
    if hgrid_spans:
        # Keep spans of typical track width.
        for a, b in hgrid_spans:
            # Linear tracks ~180 pt; log resistivity tracks can be ~360 pt.
            if 60 <= (b - a) <= 400:
                spans.append((round(float(a), 1), round(float(b), 1)))
    frames: list[TrackFrame] = []
    if spans:
        # Unique spans (cluster edges within 3 pt).
        uniq: list[tuple[float, float]] = []
        for a, b in sorted(set(spans)):
            if any(abs(a - ua) < 3 and abs(b - ub) < 3 for ua, ub in uniq):
                continue
            uniq.append((a, b))
        for a, b in uniq:
            fr = TrackFrame(x_left=float(a), x_right=float(b))
            for h in headers:
                # Scale text sits near the left of its track (or just inside).
                if a - 20 <= h.x <= b + 5:
                    fr.curves.append(h)
            if fr.curves:
                h0 = fr.curves[0]
                fr.scale = (log_scale if h0.kind == "log" else linear_scale)(
                    fr.x_left, fr.x_right, h0.v_left, h0.v_right
                )
            frames.append(fr)
        if frames:
            return frames

    # Fallback: vertical grid edges only.
    xs = sorted(set(round(float(x), 1) for x in vgrid_xs))
    xs = [0.0] + [x for x in xs if 5 < x < page_width - 5] + [float(page_width)]
    for a, b in zip(xs, xs[1:]):
        if b - a < 25:
            continue
        if depth_x is not None and a - 5 <= depth_x <= b + 5 and (b - a) < 80:
            continue
        fr = TrackFrame(x_left=float(a), x_right=float(b))
        for h in headers:
            if a - 40 <= h.x <= b + 10:
                fr.curves.append(h)
        if fr.curves:
            h0 = fr.curves[0]
            fr.scale = (log_scale if h0.kind == "log" else linear_scale)(
                fr.x_left, fr.x_right, h0.v_left, h0.v_right
            )
        frames.append(fr)
    return frames


def colour_key(rgb: tuple[float, ...], nd: int = 2) -> tuple[float, ...]:
    return tuple(round(float(c), nd) for c in rgb)


def is_near_black(rgb: tuple[float, float, float], thr: float = 0.15) -> bool:
    return all(c < thr for c in rgb)


def is_near_white(rgb: tuple[float, float, float], thr: float = 0.92) -> bool:
    return all(c > thr for c in rgb)


def is_near_gray(rgb: tuple[float, float, float], spread: float = 0.05) -> bool:
    return max(rgb) - min(rgb) < spread


def style_distance(
    color_a: tuple[float, ...], width_a: float, dash_a: tuple[float, ...],
    color_b: tuple[float, ...] | None, width_b: float | None, dash_b: tuple[float, ...] | None,
) -> float:
    """Lower = better match. Missing legend fields are ignored."""
    if color_b is None and width_b is None and dash_b is None:
        return 10.0
    d = 0.0
    if color_b is not None:
        d += sum((float(x) - float(y)) ** 2 for x, y in zip(color_a, color_b))
    if width_b is not None:
        d += 0.15 * abs(float(width_a) - float(width_b))
    if dash_b is not None:
        da, db = tuple(dash_a), tuple(dash_b)
        if bool(da) != bool(db):
            d += 1.0
        elif da and db:
            # Compare first dash/gap pair (lengths may differ).
            n = min(len(da), len(db), 2)
            d += 0.05 * sum(abs(da[i] - db[i]) for i in range(n))
    return d
