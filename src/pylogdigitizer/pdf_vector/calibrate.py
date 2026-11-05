# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Depth and scale calibration from PDF text + grid strokes (shared DepthMap / ScaleMap)."""
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
    """Return (x, y, depth) for tokens that look like depth numbers."""
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
    # Depth labels share one vertical column — keep the dominant x cluster.
    x_med = float(np.median(xs))
    # Robust: mode-ish via histogram
    hist, edges = np.histogram(xs, bins=max(5, len(xs) // 2))
    if hist.max() >= min_labels:
        k = int(np.argmax(hist))
        x_med = 0.5 * (edges[k] + edges[k + 1])
    col = [p for p in pairs if abs(p[0] - x_med) < 25]
    if len(col) < min_labels:
        col = pairs
    # Reject scale-endpoint leftovers: depths in a depth column are nearly evenly spaced.
    col = sorted(col, key=lambda p: -p[1])
    ys = np.array([p[1] for p in col], float)
    ds = np.array([p[2] for p in col], float)
    # Keep monotone increasing depth as y falls.
    keep = [0]
    for i in range(1, len(ds)):
        if ds[i] > ds[keep[-1]] + 0.05:
            keep.append(i)
    ys, ds = ys[keep], ds[keep]
    if len(ys) < min_labels:
        return None
    # Drop outliers that break the roughly-linear y↔depth relation (e.g. a lone "2000" from a scale).
    if len(ys) >= 5:
        # residual vs linear fit; drop points > 3σ
        coef = np.polyfit(-ys, ds, 1)
        pred = np.polyval(coef, -ys)
        resid = np.abs(ds - pred)
        thr = max(5.0, 3 * np.median(resid) * 1.4826)  # MAD→σ
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
        # Skip bare mnemonic tags drawn inside the track (small font, mid-page).
        if t.size and t.size < 9 and "(" not in t.text and " " not in t.text:
            continue
        mnem, title = parsed
        # Find nearest scale below this title (smaller y), similar x.
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
    # Deduplicate mnemonic (keep first = topmost).
    seen: set[str] = set()
    out = []
    # Reject tokens that are clearly not curve mnemonics (date/time marks, section titles).
    bad_prefix = ("TIME_", "DATE", "ONE", "LOG", "MAX_", "MIN_")
    for h in headers:
        if h.mnemonic in seen:
            continue
        if h.mnemonic.startswith(bad_prefix) or h.mnemonic in {"MAIN", "REPEAT", "TO"}:
            continue
        seen.add(h.mnemonic)
        out.append(h)
    return out


def legend_colors_from_strokes(
    headers: list[CurveHeader],
    strokes: list[Stroke],
    y_min: float | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Map mnemonic → RGB using short horizontal coloured swatches in the header band.

    Schlumberger plots draw a coloured tick/line next to each curve's scale row.
    """
    if not headers:
        return {}
    # Header band: near the highest scale labels.
    y_top = max(h.y for h in headers) + 40
    y_bot = min(h.y for h in headers) - 30
    if y_min is not None:
        y_bot = max(y_bot, y_min)
    swatches: list[tuple[float, float, tuple[float, float, float]]] = []  # y, x, color
    for s in strokes:
        if len(s.pts) < 2:
            continue
        rgb = s.color
        if all(c < 0.12 for c in rgb) or all(c > 0.92 for c in rgb):
            continue
        if max(rgb) - min(rgb) < 0.08:
            continue
        dx = float(np.ptp(s.pts[:, 0]))
        dy = float(np.ptp(s.pts[:, 1]))
        cy = float(np.mean(s.pts[:, 1]))
        cx = float(np.mean(s.pts[:, 0]))
        if cy < y_bot or cy > y_top:
            continue
        if dy > 8:  # not a horizontal swatch
            continue
        if dx < 20:  # too short
            continue
        swatches.append((cy, cx, (float(rgb[0]), float(rgb[1]), float(rgb[2]))))

    out: dict[str, tuple[float, float, float]] = {}
    used_cols: set[tuple[float, float, float]] = set()
    # Process top-to-bottom so each swatch row pairs with the nearest header.
    for h in sorted(headers, key=lambda hh: -hh.y):
        # Same side of the page as the scale label (left vs right track header).
        cands = [sw for sw in swatches if (sw[1] < 200) == (h.x < 200) or abs(sw[1] - h.x) < 120]
        if not cands:
            continue
        cands = sorted(cands, key=lambda sw: abs(sw[0] - (h.y + 12)) + 0.02 * abs(sw[1] - h.x))
        pick = None
        for sw in cands:
            if sw[2] not in used_cols:
                pick = sw[2]
                break
        if pick is None:
            continue
        used_cols.add(pick)
        out[h.mnemonic] = pick
        h.color = pick
    return out


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
) -> list[TrackFrame]:
    """Split page into tracks at vertical grid lines; skip the depth-label column."""
    xs = sorted(set(round(x, 1) for x in vgrid_xs))
    if len(xs) < 2:
        xs = [0.0, float(page_width)]
    frames: list[TrackFrame] = []
    for a, b in zip(xs, xs[1:]):
        if b - a < 25:
            continue
        mid = 0.5 * (a + b)
        if depth_x is not None and a - 5 <= depth_x <= b + 5 and (b - a) < 80:
            continue  # depth column
        fr = TrackFrame(x_left=float(a), x_right=float(b))
        for h in headers:
            # Scale label x is near the left edge of the track header.
            if a - 40 <= h.x <= b + 10:
                fr.curves.append(h)
        if fr.curves:
            # Default track scale from first curve; per-curve ScaleMap built at assemble time.
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
