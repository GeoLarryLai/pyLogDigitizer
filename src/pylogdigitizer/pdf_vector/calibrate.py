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


_DEPTH_RE = re.compile(r"^(\d{1,5})$")
_SCALE_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*([A-Za-z][A-Za-z0-9./%°°^_\-\s]*)\s*$"
)
_WAVEFORM_MNEM_RE = re.compile(
    r"^(?:WF\d*|PKSL|PKCH|NPK|SPJ|.*_RA_(?:LT|UT))$", re.I
)
_SCALE_SINGLE_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z0-9./%°°^_\-]*)\s*$"
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
    """Parse ``0 200 gAPI`` / ``0.2 2000 ohm.m`` → (v_left, v_right, unit, kind).

    Also handles USGS digital-print quirks such as ``48 In`` (extracted without a
    space) meaning ``4 8 In`` for shallow caliper tracks.
    """
    raw = text.replace("\n", " ").strip()
    m = _SCALE_RE.match(raw)
    if not m:
        m1 = _SCALE_SINGLE_RE.match(raw)
        if m1:
            num, unit = m1.group(1), m1.group(2)
            unit_l = unit.lower().replace("°", "")
            # Two-digit int + inch → split into left/right digit ends (4 8 In, 7 9 In).
            if num.isdigit() and len(num) == 2 and unit_l.startswith("in"):
                v0, v1 = float(num[0]), float(num[1])
                if v0 < v1:
                    return v0, v1, unit, "linear"
        return None
    v0, v1 = float(m.group(1)), float(m.group(2))
    unit = m.group(3)
    unit = unit.strip()
    unit_l = unit.lower().replace("°", "").replace(" ", "")
    span_ratio = max(abs(v0), abs(v1), 1e-12) / max(min(abs(v0), abs(v1)), 1e-12)
    # Log only for wide-span resistivity (decade+). Narrow ohm ranges (60–80) stay linear.
    ohmish = ("ohm" in unit_l) or unit_l in {"ohmm", "ohm.m", "ohm-m"}
    if ohmish and v0 > 0 and v1 > 0 and span_ratio >= 20:
        return v0, v1, unit, "log"
    if (not unit_l) and v0 > 0 and v1 > 0 and span_ratio >= 50:
        return v0, v1, unit, "log"
    return v0, v1, unit, "linear"


def _depth_label_pairs(
    texts: list[TextItem],
    min_depth: float = 0.0,
    max_depth: float = 50000.0,
) -> list[tuple[float, float, float]]:
    out = []
    for t in texts:
        m = _DEPTH_RE.match(t.text.strip())
        if not m:
            continue
        d = float(m.group(1))
        if d < min_depth or d > max_depth:
            continue
        out.append((t.x, t.y, d))
    return out


def detect_depth_axis(texts: list[TextItem], min_labels: int = 4) -> str:
    """Return ``"x"`` for horizontal digital prints (depth left→right) else ``"y"``.

    USGS GeoLog Locator exports often place depth along the page bottom; Schlumberger-style
    prints keep depth vertical. Detection uses the span of numeric
    depth tick labels only — no fudge offsets.
    """
    pairs = _depth_label_pairs(texts, min_depth=0.0)
    if len(pairs) < min_labels:
        return "y"
    xs = np.array([p[0] for p in pairs], float)
    ys = np.array([p[1] for p in pairs], float)
    if float(np.ptp(xs)) > 2.0 * max(float(np.ptp(ys)), 1.0) and float(np.ptp(xs)) > 80:
        return "x"
    return "y"


def depth_map_from_labels(
    texts: list[TextItem],
    min_labels: int = 3,
    axis: str | None = None,
) -> DepthMap | None:
    """DepthMap from numeric depth tick labels.

    Vertical prints (``axis="y"``, default): ``row = -y`` (PDF y up, depth down).
    Horizontal digital prints (``axis="x"``): ``row = x`` (depth left→right).
    """
    axis = axis or detect_depth_axis(texts, min_labels=min(min_labels, 4))
    # Shallow USGS wells label 0, 10, 20… — allow depths from 0 on either axis.
    pairs = _depth_label_pairs(texts, min_depth=0.0)
    # Keep oilfield vertical logs from grabbing random small integers by requiring
    # a deep-enough span when axis is vertical and many large depths exist.
    if axis == "y":
        deep = [p for p in pairs if p[2] >= 50]
        if len(deep) >= min_labels:
            pairs = deep
    if len(pairs) < min_labels:
        return None

    if axis == "x":
        # Depth ticks along the bottom: use x as the DepthMap row coordinate.
        col = sorted(pairs, key=lambda p: p[0])
        xs = np.array([p[0] for p in col], float)
        ds = np.array([p[2] for p in col], float)
        keep = [0]
        for i in range(1, len(ds)):
            if ds[i] > ds[keep[-1]] + 0.05:
                keep.append(i)
        xs, ds = xs[keep], ds[keep]
        if len(xs) < min_labels:
            return None
        order = np.argsort(xs)
        return DepthMap(
            rows=xs[order], depths=ds[order],
            info={"n_labels": int(len(ds)), "axis": "x", "source": "pdf_text",
                  "y_col": float(np.median([p[1] for p in col]))},
        )

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
                    info={"n_labels": int(len(ds)), "x_col": x_med, "axis": "y",
                          "source": "pdf_text"})


def _mnemonic_from_title(text: str) -> tuple[str, str] | None:
    m = _CURVE_PAREN_RE.search(text)
    if m:
        return m.group(1), text
    toks = _CURVE_TOKEN_RE.findall(text)
    cand = [tok for tok in toks if tok not in _SKIP_TITLE and len(tok) >= 2]
    if cand:
        return cand[0], text
    # USGS digital prints use mixed-case titles: "Gamma", "Fl cond amb", "Caliper".
    # Require lowercase so ALL-CAPS Schlumberger titles (CABLE DRAG, TOOL TOTAL, …)
    # do not invent extra mnemonics that steal legend styles from (TENS)/(MTEM).
    cleaned = text.strip()
    if not re.search(r"[a-z]", cleaned):
        return None
    if parse_scale_label(cleaned) is not None:
        return None
    if not re.search(r"[A-Za-z]{3,}", cleaned):
        return None
    if cleaned.lower() in {
        "depth", "optical image", "acoustic image", "gamma image",
        "dip, in degrees", "water level", "trans",
    }:
        return None
    if any(ch in cleaned for ch in "°"):
        return None
    mnem = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_").upper()
    if len(mnem) < 3 or len(mnem) > 24:
        return None
    return mnem, cleaned



def _unit_fits_mnemonic(mnemonic: str, unit: str) -> bool:
    """Reject obvious title/scale mispairs (generic, not well-specific)."""
    m = mnemonic.upper()
    u = (unit or "").lower().replace(" ", "")
    if not u:
        return True
    # Tension is force, not temperature / resistivity.
    if m in {"TENS", "CTEN", "CABLE_DRAG"} and any(x in u for x in ("deg", "ohm", "gapi", "api")):
        return False
    # Temperature not force / resistivity / gamma.
    if m in {"MTEM", "TEMP", "GTEM", "CTEM"} and (
        "lbf" in u or "ohm" in u or "gapi" in u or "api" in u or "cps" in u
    ):
        return False
    if m in {"RES", "AT20", "AT30", "AT60", "AT90"} and (
        "deg" in u or "lbf" in u or "gapi" in u
    ):
        return False
    # Gamma not porosity / sonic.
    if m in {"ECGR_STGC", "GR_STGC", "GR"} and any(x in u for x in ("us/ft", "m3/m3", "ft3/ft3", "ohm")):
        return False
    return True


def parse_curve_headers(texts: list[TextItem]) -> list[CurveHeader]:
    """Pair each curve title with the scale label immediately below it (same x band).

    Pass 1 prefers Schlumberger-style titles with ``(MNEM)`` or ALL-CAPS tokens.
    Pass 2 fills remaining scales with USGS mixed-case titles (``Gamma``, …
    ``Cable Drag`` must not steal the ``(TENS)`` scale).
    """
    items = sorted(texts, key=lambda t: (-t.y, t.x))

    def _is_usgs_title(text: str) -> bool:
        return bool(re.search(r"[a-z]", text)) and "(" not in text

    def _pair(pass_usgs: bool, used_scale_idx: set[int]) -> list[CurveHeader]:
        headers: list[CurveHeader] = []
        for i, t in enumerate(items):
            parsed = _mnemonic_from_title(t.text)
            if not parsed:
                continue
            usgs = _is_usgs_title(t.text)
            if pass_usgs != usgs:
                continue
            # Schlumberger titles are large; USGS digital-print titles often report size≈1.
            if t.size and t.size < 9 and "(" not in t.text and " " not in t.text and t.size >= 3:
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
                if not _unit_fits_mnemonic(mnem, sc[2]):
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
        return headers

    used: set[int] = set()
    headers = _pair(False, used) + _pair(True, used)
    seen: set[str] = set()
    out = []
    bad_prefix = ("TIME_", "DATE", "ONE", "LOG", "MAX_", "MIN_")
    for h in headers:
        if h.mnemonic in seen:
            continue
        if h.mnemonic.startswith(bad_prefix) or h.mnemonic in {"MAIN", "REPEAT", "TO", "DRAG"}:
            continue
        if _WAVEFORM_MNEM_RE.match(h.mnemonic):
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
    """Pull WELL/COMP/FLD/LOC/elevations/DATE from title-page text items."""
    if isinstance(texts, str):
        items = [TextItem(text=line, x=0, y=0, size=10) for line in texts.splitlines() if line.strip()]
    else:
        items = list(texts)
    blob = "\n".join(t.text for t in items)
    well: dict[str, str] = {}
    patterns = {
        "WELL": r"Well\s*:\s*([^\n|]+)",
        "COMP": r"Company\s*:\s*([^\n|]+)",
        "FLD": r"Field\s*:\s*([^\n|]+)",
        "CNTY": r"County\s*:\s*([^\n|,]+)",
        "CTRY": r"Country\s*:\s*([^\n|]+)",
        "DATE": r"(?:Creation\s*Date|Logging\s*Date|Date)\s*:?\s*([0-9]{1,2}[-/][A-Za-z]{3}[-/][0-9]{2,4})",
    }
    for k, pat in patterns.items():
        m_ = re.search(pat, blob, re.I)
        if m_ and m_.group(1).strip():
            well[k] = m_.group(1).strip()
    labels = {
        "well": "WELL", "company": "COMP", "field": "FLD", "county": "CNTY",
        "country": "CTRY", "logging date": "DATE", "date": "DATE",
    }
    for i, t_ in enumerate(items):
        key = t_.text.strip().rstrip(":").lower()
        if key in labels and labels[key] not in well and i + 1 < len(items):
            val = items[i + 1].text.strip()
            if not val or val.endswith(":"):
                continue
            # USGS form pages interleave labels; reject tool-run / long junk.
            if labels[key] == "CNTY" and ("tool" in val.lower() or len(val) > 30):
                continue
            well[labels[key]] = val
    if "DATE" not in well:
        m_ = re.search(r"\b([0-9]{2}-[A-Za-z]{3}-[0-9]{4})\b", blob)
        if m_:
            well["DATE"] = m_.group(1)
    m_ = re.search(r"Lat\s*:\s*([-\d.]+)\s*,?\s*Long\s*:\s*([-\d.]+)", blob, re.I)
    if m_:
        well["LOC"] = f"Lat {m_.group(1)} Long {m_.group(2)}"
    for key, pat in (
        ("EKB", r"K\.?B\.?\s*([-\d.]+)\s*ft"),
        ("EGL", r"G\.?L\.?\s*([-\d.]+)\s*ft"),
        ("EDF", r"D\.?F\.?\s*([-\d.]+)\s*ft"),
    ):
        m_ = re.search(pat, blob, re.I)
        if m_:
            well[key] = m_.group(1)
    # USGS GeoLog Locator header forms: label text beside value, not "Well:".
    if "WELL" not in well:
        for i, t_ in enumerate(items):
            lab = t_.text.strip().rstrip(":").lower()
            if lab in {"station name (c12)", "station name"} and i + 1 < len(items):
                well["WELL"] = items[i + 1].text.strip()
                break
        if "WELL" not in well:
            m_ = re.search(r"\b(Ro[- ]?\d+\s*\([^)]+\)|Ro[- ]?\d+)\b", blob)
            if m_:
                well["WELL"] = m_.group(1).strip()
    if "COMP" not in well and re.search(r"\bUSGS\b", blob):
        well["COMP"] = "USGS"
    if "SRVC" not in well and re.search(r"\bUSGS\b", blob):
        well["SRVC"] = "USGS"
    if "UWI" not in well:
        m_ = re.search(r"\b(\d{15})\b", blob)
        if m_:
            well["UWI"] = m_.group(1)
    if "DATE" not in well:
        m_ = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", blob)
        if m_:
            well["DATE"] = m_.group(1)
    if "CNTY" not in well:
        for i, t_ in enumerate(items):
            if t_.text.strip().rstrip(":").lower() == "county" and i + 1 < len(items):
                # Avoid grabbing a later "Tool run" label; take short token.
                val = items[i + 1].text.strip()
                if val and len(val) < 40 and "tool" not in val.lower():
                    well["CNTY"] = val
                break
    if "STAT" not in well:
        for i, t_ in enumerate(items):
            if t_.text.strip().rstrip(":").lower() == "state" and i + 1 < len(items):
                val = items[i + 1].text.strip()
                if val and len(val) <= 3:
                    well["STAT"] = val
                break
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
