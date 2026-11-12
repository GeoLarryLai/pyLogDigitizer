# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Vector well-log PDFs: read the curves as the exact paths drawn in the file.

For a PDF page whose curves are vector paths with selectable text (a digital print, not a scan):

1. the track borders come from the long vertical lines of the plot;
2. depth is registered from the printed depth numbers (a straight fit of depth against the
   height of each number, checked for a constant spacing);
3. each curve's name, scale values, unit and colour are read from the header text and the
   coloured scale bar above the plot;
4. every curve path is matched to its header entry (by track and colour) and its points are
   turned into depth and value exactly, with no clean-up, ink mask or tracing;
5. :func:`resample_path` puts each curve on a depth step you choose, and
   :meth:`DigitizedLog.to_las` writes the LAS file.

:func:`digitize_vector` runs all of it. Pages that do not have this layout fall back to
:func:`pylogdigitizer.pdf_vector.digitize_pdf`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

from pylogdigitizer.pdf_vector.extract import (Stroke, TextRun, apply_frame, extract_strokes, extract_text_runs,
                                               frame_transform, runs_in_frame)

_NUM = re.compile(r"^[-+\u2212]?(\d+(\.\d*)?|\.\d+)$")


def _num(t: str) -> float | None:
    t = t.strip().replace("\u2212", "-")
    return float(t) if _NUM.match(t) else None


@dataclass
class HeaderEntry:
    """One curve's header: printed name, scale (value at the left and right end), unit, scale type,
    the page x of both scale ends, the scale bar colour (if one is drawn) and the header line y."""
    name: str
    left: float
    right: float
    unit: str = ""
    kind: str = "linear"
    x0: float = 0.0
    x1: float = 0.0
    color: tuple[float, float, float] | None = None
    y: float = 0.0
    mnemonic: str = ""
    dash: tuple[float, ...] = ()

    def value(self, x: np.ndarray) -> np.ndarray:
        """Page x -> curve value on this scale (linear or log)."""
        f = (np.asarray(x, float) - self.x0) / max(self.x1 - self.x0, 1e-9)
        if self.kind == "log":
            a, b = np.log10(self.left), np.log10(self.right)
            return 10 ** (a + f * (b - a))
        return self.left + f * (self.right - self.left)


@dataclass
class VectorPage:
    """What was read from one vector page. Coordinates are PDF points, y up, in the *log frame*: the
    page turned by ``rotation`` degrees clockwise so the log reads upright (header on top, depth
    down). ``frame`` is the 3x3 matrix from PDF user space to that frame (identity for upright pages)."""
    page_index: int
    width: float
    height: float
    strokes: list[Stroke]
    texts: list[TextRun]
    borders: list[float] = field(default_factory=list)
    plot_top: float | None = None
    plot_bottom: float | None = None
    depth_labels: list[tuple[float, float]] = field(default_factory=list)   # (y centre, depth)
    depth_fit: tuple[float, float] | None = None                            # depth = a + b * y
    headers: list[HeaderEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rotation: int = 0
    frame: np.ndarray = field(default_factory=lambda: np.eye(3))
    raw_texts: list[TextRun] = field(default_factory=list, repr=False)   # page text as extracted
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def depth(self, y) -> np.ndarray:
        a, b = self.depth_fit
        return a + b * np.asarray(y, float)

    def y_of_depth(self, d) -> np.ndarray:
        a, b = self.depth_fit
        return (np.asarray(d, float) - a) / b


# ------------------------------------------------------------------ text helpers
def _merge_runs(runs: list[TextRun]) -> list[TextRun]:
    """Join runs on one baseline that touch (a word drawn in pieces, e.g. 'A' 'zimut' 'h')."""
    out: list[TextRun] = []
    for r in sorted(runs, key=lambda r: (round(-r.y, 1), r.x)):
        p = out[-1] if out else None
        if (p is not None and abs(p.y - r.y) < 0.2 * max(p.size, 1e-6) and abs(p.size - r.size) < 0.2 * p.size
                and -0.3 * p.size < r.x - p.x1 < 0.15 * p.size
                and not (_num(p.text) is not None and _num(r.text) is not None)):   # two scale numbers side by side
            out[-1] = TextRun(x=p.x, y=p.y, x1=max(p.x1, r.x1), text=p.text + r.text, size=p.size)
        else:
            out.append(r)
    return out


def mnemonic_of(name: str) -> str:
    """LAS-safe mnemonic of a printed curve name: 'Fl cond amb' -> 'FL_COND_AMB'."""
    m = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").upper()
    return m or "CURVE"


def _scale_kind(left: float, right: float, unit: str) -> str:
    lo, hi = sorted((abs(left), abs(right)))
    if left > 0 and right > 0 and hi / max(lo, 1e-12) >= 50:
        lg = np.log10([left, right])
        if np.allclose(lg, np.round(lg), atol=0.05) or "ohm" in unit.lower():
            return "log"
    return "linear"


# ------------------------------------------------------------------ layout
def _vertical_lines(strokes: list[Stroke]):
    out = []
    for s in strokes:
        p = s.pts
        if len(p) < 2:
            continue
        for a, b in zip(p[:-1], p[1:]):
            if abs(a[0] - b[0]) < 0.3 and abs(a[1] - b[1]) > 20:
                out.append((float(a[0]), float(min(a[1], b[1])), float(max(a[1], b[1])), bool(s.dash)))
    return out


def _cluster(xs, tol=1.0):
    xs = sorted(xs)
    groups: list[list[float]] = []
    for x in xs:
        if groups and x - groups[-1][-1] <= tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [float(np.mean(g)) for g in groups]


def _frame(vp: VectorPage) -> None:
    """Track borders and the plot top / bottom from the long solid vertical lines."""
    vl = [v for v in _vertical_lines(vp.strokes) if not v[3]]
    if not vl:
        vp.notes.append("no vertical track lines found")
        return
    longest = max(v[2] - v[1] for v in vl)
    tall = [v for v in vl if v[2] - v[1] >= 0.6 * longest]
    vp.borders = _cluster([v[0] for v in tall], tol=1.0)
    vp.plot_bottom = float(np.median([v[1] for v in tall]))
    vp.plot_top = float(np.median([v[2] for v in tall]))


def _depth(vp: VectorPage) -> None:
    """Fit depth against the height of the printed depth numbers (a column of plain numbers)."""
    lo, hi = (vp.plot_bottom - 2, vp.plot_top + 2) if vp.plot_top is not None else (-np.inf, np.inf)
    cand = [t for t in vp.texts if _num(t.text) is not None and lo <= t.y <= hi]
    cols: dict[float, list[TextRun]] = {}
    for t in cand:
        cx = 0.5 * (t.x + t.x1)
        k = next((c for c in cols if abs(c - cx) < 12), cx)
        cols.setdefault(k, []).append(t)
    best = None
    for col in cols.values():
        if len(col) < 3:
            continue
        col = sorted(col, key=lambda t: -t.y)
        d = np.array([_num(t.text) for t in col]); y = np.array([t.y + 0.36 * t.size for t in col])
        if len(set(d)) < len(d) or not (np.all(np.diff(d) > 0) or np.all(np.diff(d) < 0)):
            continue
        b, a = np.polyfit(y, d, 1)
        res = np.abs(a + b * y - d)
        step = np.median(np.abs(np.diff(d)))
        if res.max() > 0.05 * step:
            continue
        if best is None or len(col) > len(best[0]):
            best = (col, a, b, y, d)
    if best is None:
        vp.notes.append("no column of depth numbers found")
        return
    col, a, b, y, d = best
    # snap to horizontal depth lines when the print has them at the numbers
    hl = [float(s.pts[0, 1]) for s in vp.strokes if len(s.pts) == 2 and abs(s.pts[0, 1] - s.pts[1, 1]) < 0.2
          and abs(s.pts[0, 0] - s.pts[1, 0]) > 20 and lo < s.pts[0, 1] < hi]
    if hl:
        hl = np.array(hl); ys = y.copy()
        for i, yy in enumerate(y):
            j = np.argmin(np.abs(hl - yy))
            if abs(hl[j] - yy) < 0.5 * col[i].size:
                ys[i] = hl[j]
        b, a = np.polyfit(ys, d, 1); y = ys
    vp.depth_labels = [(float(yy), float(dd)) for yy, dd in zip(y, d)]
    vp.depth_fit = (float(a), float(b))


def _headers(vp: VectorPage) -> None:
    """Header entries: per printed scale line (left value, unit, right value), the name above it
    and the coloured scale bar next to it."""
    if vp.plot_top is not None:
        top = vp.plot_top
    elif vp.depth_labels:                      # no plot frame (text over a scan): above the depth numbers
        top = max(y for y, _ in vp.depth_labels) + 2.0
    else:
        top = -np.inf
    head = [t for t in vp.texts if t.y > top - 1]
    bars = []
    for s in vp.strokes:
        p = s.pts
        if len(p) == 2 and abs(p[0, 1] - p[1, 1]) < 0.3 and abs(p[0, 0] - p[1, 0]) > 10 and p[0, 1] > top + 1.0:
            x0, x1 = sorted((float(p[0, 0]), float(p[1, 0])))
            if vp.borders and x1 - x0 > 0.9 * (max(vp.borders) - min(vp.borders)) or x1 - x0 > 0.8 * vp.width:
                continue                       # the frame, not a scale bar
            bars.append((x0, x1, float(p[0, 1]), tuple(float(c) for c in s.color), tuple(s.dash)))
    lines: dict[float, list[TextRun]] = {}
    for t in head:
        k = next((y for y in lines if abs(y - t.y) < 0.25 * t.size), t.y)
        lines.setdefault(k, []).append(t)
    names = [t for t in head if _num(t.text) is None]
    entries: list[HeaderEntry] = []
    for y, toks in lines.items():
        toks = sorted(toks, key=lambda t: t.x)
        i = 0
        while i < len(toks):
            v0 = _num(toks[i].text)
            if v0 is None:
                i += 1; continue
            j = i + 1
            while j < len(toks) and _num(toks[j].text) is None:
                j += 1
            if j >= len(toks):
                break
            v1 = _num(toks[j].text)
            mid = toks[i + 1:j]
            span = (toks[i].x, toks[j].x1)
            unit = " ".join(t.text for t in mid)
            entries.append(HeaderEntry(name="", left=v0, right=v1, unit=unit, x0=span[0], x1=span[1], y=y))
            i = j + 1
    used: set[int] = set()
    for e in entries:
        sz = next((t.size for t in head if abs(t.y - e.y) < 0.5), 8.0)
        # scale bar: overlapping the numbers, within ~2 lines above or below
        best = None
        for b in bars:
            ov = min(b[1], e.x1) - max(b[0], e.x0)
            if ov > 0.5 * (e.x1 - e.x0) and abs(b[2] - e.y) < 2.2 * sz:
                if best is None or abs(b[2] - e.y) < abs(best[2] - e.y):
                    best = b
        if best is not None:
            e.x0, e.x1, e.color, e.dash = best[0], best[1], best[3], best[4]
        # snap the scale ends to the track borders they sit on
        if vp.borders:
            bd = np.array(vp.borders)
            l = bd[np.argmin(np.abs(bd - e.x0))]; r = bd[np.argmin(np.abs(bd - e.x1))]
            if best is not None:
                if abs(l - e.x0) < 1.5: e.x0 = float(l)
                if abs(r - e.x1) < 1.5: e.x1 = float(r)
            else:
                left = bd[bd <= e.x0 + 0.5]; right = bd[bd >= e.x1 - 0.5]
                if len(left): e.x0 = float(left.max())
                if len(right): e.x1 = float(right.min())
        # name: the nearest non-number text above, overlapping the scale span
        def ov(t):
            return (min(t.x1, e.x1) - max(t.x, e.x0)) / max(t.x1 - t.x, 1e-6)
        above = [(k, t) for k, t in enumerate(names) if t.y > e.y + 0.3 * sz and t.y - e.y < 3.5 * sz and k not in used
                 and ov(t) > 0.3]
        if above:
            k, t = min(above, key=lambda kt: (round(kt[1].y - e.y, 1), -ov(kt[1])))
            used.add(k); e.name = t.text
        elif e.unit:
            e.name, e.unit = e.unit, ""
        e.kind = _scale_kind(e.left, e.right, e.unit)
        e.mnemonic = mnemonic_of(e.name) if e.name else ""
    vp.headers = [e for e in entries if e.name]


def _read_in_frame(page_index: int, box, strokes, runs, rotation: int) -> VectorPage:
    M = frame_transform(box, rotation)
    w, h = float(box[2] - box[0]), float(box[3] - box[1])
    if rotation % 180:
        w, h = h, w
    st = [Stroke(pts=apply_frame(M, s.pts), color=s.color, width=s.width, dash=s.dash) for s in strokes]
    vp = VectorPage(page_index=page_index, width=w, height=h, strokes=st,
                    texts=_merge_runs(runs_in_frame(runs, M, rotation)), rotation=rotation, frame=M,
                    raw_texts=runs, box=tuple(box))
    _frame(vp); _depth(vp); _headers(vp)
    return vp


def _score(vp: VectorPage) -> tuple:
    return (vp.depth_fit is not None, len(vp.headers) > 0, vp.plot_top is not None, len(vp.headers), len(vp.depth_labels))


def page_rotations(pg, runs: list[TextRun]) -> list[int]:
    """Frames to try for a page, most likely first: the reading direction of most of its text
    (sideways logs, rotating page CTMs), then the page's own /Rotate, then the rest."""
    chars: dict[int, int] = {}
    for r in runs:
        if r.angle >= 0:
            chars[r.angle] = chars.get(r.angle, 0) + len(r.text)
    order = sorted(chars, key=lambda a: -chars[a])
    try:
        rot = int(pg.get("/Rotate", 0) or 0) % 360
    except Exception:
        rot = 0
    for a in [rot, 0, 90, 180, 270]:
        if a not in order:
            order.append(a)
    return order


def read_vector_page(path: str | Path, page: int, rotation: int | None = None) -> VectorPage:
    """Read one page's paths, text, track borders, depth numbers and curve headers.

    Pages turned sideways are read exactly too: the page /Rotate, rotating content transforms and
    logs drawn sideways on a landscape sheet are handled by reading in the frame where the log is
    upright. ``rotation`` (0, 90, 180, 270, clockwise) forces that frame; by default the frame of
    most of the text is tried first, then the others, and the one where a depth column and a
    header are found is kept (``vp.rotation``)."""
    from pypdf import PdfReader
    pg = PdfReader(str(path)).pages[page]
    b = pg.mediabox
    box = (float(b.left), float(b.bottom), float(b.right), float(b.top))
    strokes, runs = extract_strokes(pg), extract_text_runs(pg)
    if rotation is not None:
        return _read_in_frame(page, box, strokes, runs, int(rotation) % 360)
    best = None
    for k, rot in enumerate(page_rotations(pg, runs)):
        vp = _read_in_frame(page, box, strokes, runs, rot)
        if best is None or _score(vp) > _score(best):
            best = vp
        if vp.depth_fit is not None and vp.headers and vp.plot_top is not None:
            break                                       # a complete reading: no need to try more frames
    if best.rotation and best.depth_fit is not None:
        best.notes.append(f"page read turned {best.rotation} degrees clockwise (sideways log)")
    return best


# ------------------------------------------------------------------ curves
def _colour_dist(a, b) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))


def chain_strokes(strokes: list[Stroke], tol: float = 0.05) -> list[Stroke]:
    """Join paths of one style (colour, width, dash) whose ends meet into one polyline. Some
    exporters draw a curve as thousands of two-point segments; joined, it is one exact path again.
    Straight two-point lines that meet nothing stay as they are."""
    from collections import deque
    chains: list = []                        # [first stroke, deque of point arrays] (None once joined away)
    heads: dict = {}; tails: dict = {}       # (style, rounded point) -> chain index
    q = 1.0 / tol

    def key(st, pt):
        return (st, int(round(pt[0] * q)), int(round(pt[1] * q)))

    for s in strokes:
        p = s.pts
        if len(p) < 2:
            continue
        st = (tuple(np.round(s.color, 3)), round(float(s.width), 3), tuple(s.dash))
        k0, k1 = key(st, p[0]), key(st, p[-1])
        i = tails.pop(k0, None)
        if i is None and (i := tails.pop(k1, None)) is not None:
            p = p[::-1]; k0, k1 = k1, k0
        if i is not None:                    # append at the chain's end
            chains[i][1].append(p[1:]); tails[k1] = i
            continue
        i = heads.pop(k1, None)
        if i is None and (i := heads.pop(k0, None)) is not None:
            p = p[::-1]; k0, k1 = k1, k0
        if i is not None:                    # put in front of the chain's start
            chains[i][1].appendleft(p[:-1]); heads[k0] = i
            continue
        chains.append([s, deque([p])]); i = len(chains) - 1
        heads[k0] = i; tails[k1] = i
    res = []
    for s, parts in chains:
        res.append(s if len(parts) == 1 else
                   Stroke(pts=np.concatenate(list(parts)), color=s.color, width=s.width, dash=s.dash))
    return res


def _style_dist(s: Stroke, e: "HeaderEntry") -> float:
    """Colour distance, plus a penalty when one is dashed and the other solid."""
    return _colour_dist(s.color, e.color) + (0.5 if bool(s.dash) != bool(e.dash) else 0.0)


def vector_curves(vp: VectorPage) -> list:
    """Each header entry's curve as exact points: a :class:`CurveTrace` per curve with depth and
    value, plus the page ``x``/``y`` of every point in ``info``."""
    from pylogdigitizer.pdf_vector.assemble import CurveTrace
    if vp.depth_fit is None or vp.plot_top is None:
        return []
    top, bot = vp.plot_top, vp.plot_bottom
    H = top - bot
    borders = set(round(b, 1) for b in vp.borders)
    picks: dict[int, list[Stroke]] = {}
    how: dict[int, str] = {}
    orphans: list[Stroke] = []
    unmatched = 0
    for s in chain_strokes(vp.strokes):
        p = s.pts
        inside = (p[:, 1] >= bot - 0.5) & (p[:, 1] <= top + 0.5)
        if inside.sum() < 6 or np.ptp(p[inside, 1]) < 0.03 * H:
            continue
        q = p[inside]
        if np.ptp(q[:, 0]) < 0.3:
            continue                             # a straight vertical line (border or grid)
        if (np.isin(np.round(q[:, 0], 1), list(borders)) | (np.abs(q[:, 1] - top) < 0.5)
                | (np.abs(q[:, 1] - bot) < 0.5)).all():
            continue                             # the plot frame drawn as one joined outline
        xs = q[:, 0]
        cands = [k for k, e in enumerate(vp.headers) if np.mean((xs >= e.x0 - 1) & (xs <= e.x1 + 1)) >= 0.8]
        if not cands:
            orphans.append(s)
            continue
        if len(cands) > 1:
            col = [k for k in cands if vp.headers[k].color is not None]
            if col:
                d = {k: _style_dist(s, vp.headers[k]) for k in col}
                k = min(d, key=d.get)
                if d[k] < 0.25 and sum(v < 0.25 for v in d.values()) == 1:
                    picks.setdefault(k, []).append(s); how[k] = "colour"; continue
                if len(col) == len(cands) and d[k] >= 0.25:
                    unmatched += 1; continue     # every scale has a colour and none is this path's
            free = [k for k in cands if k not in picks]
            k = free[0] if free else cands[0]
            picks.setdefault(k, []).append(s); how.setdefault(k, "order"); continue
        picks.setdefault(cands[0], []).append(s); how.setdefault(cands[0], "track")
    if unmatched:
        vp.notes.append(f"{unmatched} path(s) in shared tracks match no scale colour; not digitized")
    for s in orphans:                            # a curve without a printed scale: say so, never guess one
        xm = float(np.median(s.pts[:, 0]))
        bd = np.array(vp.borders)
        l = bd[bd <= xm].max() if (bd <= xm).any() else xm - 15
        r = bd[bd >= xm].min() if (bd >= xm).any() else xm + 15
        nm = [t.text for t in sorted(vp.texts, key=lambda t: -t.y)
              if t.y > top and l <= 0.5 * (t.x + t.x1) <= r and _num(t.text) is None]
        vp.notes.append(f"{nm[0] if nm else 'a curve'} (x {np.min(s.pts[:, 0]):.0f}-{np.max(s.pts[:, 0]):.0f} pt): "
                        "path found but no scale printed for it; not digitized")
    out = []
    seen: dict[str, int] = {}
    for k, e in enumerate(vp.headers):
        if k not in picks:
            vp.notes.append(f"{e.name}: no curve path in its track (points or an image?)")
            continue
        xs, ys = [], []
        for s in picks[k]:
            p = s.pts
            m = (p[:, 1] >= bot - 0.01) & (p[:, 1] <= top + 0.01)
            # drop a closing run along a track border (outline of a filled area)
            on_edge = np.isin(np.round(p[:, 0], 1), list(borders))
            m &= ~(on_edge & np.r_[on_edge[1:], True] & np.r_[True, on_edge[:-1]])
            # and a last flat jump out to a border at either end (the pen closing the outline)
            for i, j in ((0, 1), (len(p) - 1, len(p) - 2)):
                if len(p) > 2 and on_edge[i] and abs(p[i, 1] - p[j, 1]) < 1.0 and abs(p[i, 0] - p[j, 0]) > 3.0:
                    m[i] = False
            if len(xs):
                xs.append([np.nan]); ys.append([np.nan])
            xs.append(p[m, 0]); ys.append(p[m, 1])
        x = np.concatenate(xs); y = np.concatenate(ys)
        depth = vp.depth(y); val = e.value(x)
        val[~np.isfinite(depth)] = np.nan
        mn = e.mnemonic
        if mn in seen:
            seen[mn] += 1; mn = f"{mn}_{seen[mn]}"
        else:
            seen[mn] = 1
        out.append(CurveTrace(mnemonic=mn, depth=depth, value=val, color=tuple(picks[k][0].color), unit=e.unit,
                              kind=e.kind, page_index=vp.page_index,
                              info=dict(name=e.name, x=x, y=y, left=e.left, right=e.right, x0=e.x0, x1=e.x1,
                                        matched_by=how.get(k, "track"), n_pts=int(np.isfinite(x).sum()),
                                        n_paths=len(picks[k]))))
    return out


def resample_path(depth, value, step: float = 0.5, max_gap: float | None = None, start: float | None = None):
    """Exact path points -> values on a regular depth grid of ``step`` (any step > 0, in the depth
    unit). Linear interpolation between neighbouring path points only, never smoothed. A grid depth
    is null where the path has a break (NaN) or its neighbours are more than ``max_gap`` apart
    (default: 4 steps or 2 % of the depth range, whichever is larger)."""
    step = float(step)
    if not step > 0:
        raise ValueError("depth step must be > 0")
    d = np.asarray(depth, float); v = np.asarray(value, float)
    ok = np.isfinite(d)
    d, v = d[ok], v[ok]
    if d.size < 2:
        return np.array([]), np.array([])
    o = np.argsort(d, kind="stable"); d, v = d[o], v[o]
    s0 = np.ceil(d[0] / step) * step if start is None else float(start)
    grid = np.round(np.arange(s0, d[-1] + 1e-9, step), 6)
    j = np.clip(np.searchsorted(d, grid), 1, len(d) - 1)
    d0, d1, v0, v1 = d[j - 1], d[j], v[j - 1], v[j]
    t = np.where(d1 > d0, (grid - d0) / np.where(d1 > d0, d1 - d0, 1), 0.0)
    out = v0 + t * (v1 - v0)
    exact = np.isclose(grid, d1)
    out = np.where(exact, v1, out)
    mg = max(4 * step, 0.02 * (d[-1] - d[0])) if max_gap is None else max_gap
    bad = (~np.isfinite(v0) | ~np.isfinite(v1) | (d1 - d0 > mg)) & ~(exact & np.isfinite(v1))
    out[bad] = np.nan
    return grid, out


def reading_order(runs: list[TextRun], box) -> list:
    """Page text as TextItems in reading order, each run read in its own direction (a header block
    printed sideways reads label, value, label, value like an upright one)."""
    from pylogdigitizer.pdf_vector.extract import TextItem
    out = []
    angles = sorted({r.angle for r in runs if r.angle in (0, 90, 180, 270)},
                    key=lambda a: -sum(r.angle == a for r in runs))
    for a in angles:
        rr = _merge_runs(runs_in_frame([r for r in runs if r.angle == a], frame_transform(box, a), a))
        out += [TextItem(x=r.x, y=r.y, text=r.text, size=r.size) for r in sorted(rr, key=lambda r: (round(-r.y, 1), r.x))]
    return out


def _pick(path):
    """(page, VectorPage, traces) of the first page with curve paths and a depth column (header
    pages are skipped); (None, None, []) when there is none."""
    from pypdf import PdfReader
    n = len(PdfReader(str(path)).pages)
    for i in range(n):
        try:
            vp = read_vector_page(path, i)
        except Exception:
            continue
        if vp.depth_fit is not None and vp.headers:
            tr = vector_curves(vp)
            if tr:
                return i, vp, tr
    return None, None, []


def _pick_page(path) -> int:
    """The first page with curve paths and a depth column (header pages are skipped)."""
    return _pick(path)[0]


def digitize_vector(path: str | Path, page: int | None = None, step: float = 0.5, fallback: bool = True):
    """Vector well-log PDF -> :class:`DigitizedLog` with exact curve points and a regular grid.

    ``page`` is 0-based (``None``: the first page with curves and depth numbers). ``step`` is the
    output depth step (any value > 0). The traces keep every path point (``log.traces[i].depth`` /
    ``.value``, page coordinates in ``.info``); ``log.depth`` / ``log.curves`` hold the resampled
    grid; ``log.to_las(path)`` writes the LAS. ``log.info['page']`` describes what was read.

    When the page has no readable depth column or header (another layout), and ``fallback`` is on,
    the older :func:`digitize_pdf` reader is used instead.
    """
    from pylogdigitizer.pdf_vector.digitize import DigitizedLog, digitize_pdf
    path = Path(path)
    asked = page
    vp, traces = None, []
    if page is None:
        page, vp, traces = _pick(path)
    page = 0 if page is None else page
    if vp is None:
        vp = read_vector_page(path, page)
        traces = vector_curves(vp)
    if not traces:
        if fallback:
            log = digitize_pdf(path, pages=None if asked is None else [asked], step=step)
            log.info["reader"] = "digitize_pdf (fallback)"; log.info["notes"] = vp.notes
            if log.traces:
                return log
        raise ValueError(f"{path.name} page {page + 1}: no vector curves with a depth column and header found ("
                         + "; ".join(vp.notes or ["unknown layout"]) + ")")
    well = {}
    try:
        from pylogdigitizer.pdf_vector.calibrate import parse_well_header
        if page == 0:
            runs, box = vp.raw_texts, vp.box
        else:
            from pypdf import PdfReader
            pg0 = PdfReader(str(path)).pages[0]; b = pg0.mediabox
            runs, box = extract_text_runs(pg0), (float(b.left), float(b.bottom), float(b.right), float(b.top))
        well = parse_well_header(reading_order(runs, box))
    except Exception:
        pass
    log = DigitizedLog(traces=traces, well=well, units={t.mnemonic: t.unit for t in traces}, step=step,
                       info=dict(reader="vector", page=page, notes=vp.notes,
                                 headers=[dict(name=e.name, mnemonic=e.mnemonic, left=e.left, right=e.right,
                                               unit=e.unit, kind=e.kind) for e in vp.headers],
                                 depth_labels=vp.depth_labels, depth_fit=vp.depth_fit))
    log.vector_page = vp
    resample_log(log, step)
    return log


def resample_log(log, step: float = 0.5) -> None:
    """Put every trace of ``log`` on one regular grid of ``step`` (fills ``log.depth`` / ``log.curves``)."""
    series = {t.mnemonic: resample_path(t.depth, t.value, step) for t in log.traces}
    series = {k: v for k, v in series.items() if len(v[0])}
    if not series:
        log.depth, log.curves = np.array([]), {}
        return
    d0 = min(g[0] for g, _ in series.values()); d1 = max(g[-1] for g, _ in series.values())
    grid = np.round(np.arange(np.ceil(d0 / step) * step, d1 + 1e-9, step), 6)
    curves = {}
    for k, (g, v) in series.items():
        out = np.full(grid.shape, np.nan)
        idx = np.clip(np.searchsorted(grid, g - 1e-7), 0, len(grid) - 1)
        okk = np.isclose(grid[idx], g, atol=1e-6)
        out[idx[okk]] = v[okk]
        curves[k] = out
    log.depth, log.curves, log.step = grid, curves, step
