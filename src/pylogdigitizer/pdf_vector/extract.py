# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Extract vector strokes (colour / width / dash) and positioned text from a PDF page via pypdf."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pypdf import PdfReader
from pypdf.generic import ArrayObject, ContentStream, NameObject


def _op_name(op: Any) -> str:
    if isinstance(op, bytes):
        return op.decode("latin-1")
    if isinstance(op, NameObject):
        return str(op)
    return str(op)


@dataclass
class Stroke:
    """One stroked path in page coordinates (origin bottom-left, y up)."""
    pts: np.ndarray  # (N, 2) float
    color: tuple[float, float, float]
    width: float
    dash: tuple[float, ...] = ()


@dataclass
class TextItem:
    x: float
    y: float
    text: str
    size: float = 0.0


@dataclass
class PageVectors:
    page_index: int
    width: float
    height: float
    strokes: list[Stroke] = field(default_factory=list)
    texts: list[TextItem] = field(default_factory=list)

    @property
    def plain_text(self) -> str:
        return "\n".join(t.text for t in self.texts)


def _apply_ctm(ctm: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = ctm @ np.array([x, y, 1.0], float)
    return float(v[0]), float(v[1])


def _sample_cubic(p0, p1, p2, p3, n: int = 4) -> list[tuple[float, float]]:
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1.0 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        out.append((x, y))
    return out


def _colour(operands) -> tuple[float, float, float] | None:
    """RGB from 1 (grey), 3 (RGB) or 4 (CMYK) numeric colour operands; pattern names are skipped."""
    vals = []
    for o in operands:
        try:
            vals.append(float(o))
        except (TypeError, ValueError):
            pass
    if len(vals) == 1:
        return (vals[0],) * 3
    if len(vals) == 3:
        return (vals[0], vals[1], vals[2])
    if len(vals) == 4:
        c, m, y, k = vals
        return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return None


def page_operations(page, contents=None) -> list:
    """The page's parsed content-stream operations, parsed once per page object (strokes, text
    and rendering all walk the same list; parsing is the slow part on big pages)."""
    ops = getattr(page, "_pld_ops", None)
    if ops is None:
        contents = page.get_contents() if contents is None else contents
        ops = [] if contents is None else ContentStream(contents, page).operations
        try:
            page._pld_ops = ops
        except Exception:
            pass
    return ops


def extract_strokes(page) -> list[Stroke]:
    """Parse the page content stream into stroked polylines with graphics state."""
    contents = page.get_contents()
    if contents is None:
        return []
    strokes: list[Stroke] = []
    pts: list[tuple[float, float]] = []
    color = (0.0, 0.0, 0.0)
    width = 1.0
    dash: tuple[float, ...] = ()
    ctm = np.eye(3)
    stack: list[tuple] = []

    def flush() -> None:
        nonlocal pts
        if len(pts) >= 2:
            strokes.append(Stroke(pts=np.asarray(pts, float), color=color, width=width, dash=dash))
        pts = []

    for operands, operator in page_operations(page, contents):
        op = _op_name(operator)
        if op == "q":
            stack.append((ctm.copy(), color, width, dash))
        elif op == "Q" and stack:
            ctm, color, width, dash = stack.pop()
            pts = []
        elif op == "cm" and len(operands) >= 6:
            a, b, c, d, e, f = (float(x) for x in operands[:6])
            ctm = ctm @ np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], float)
        elif op in ("RG", "G", "K", "SC", "SCN"):
            # stroking colour only (rg / g / k / sc / scn set the fill colour, which strokes never use)
            c = _colour(operands)
            if c is not None:
                color = c
        elif op == "w" and operands:
            width = float(operands[0])
        elif op == "d" and operands:
            arr = operands[0]
            dash = tuple(float(x) for x in arr) if isinstance(arr, ArrayObject) else ()
        elif op == "m" and len(operands) >= 2:
            flush()
            pts = [_apply_ctm(ctm, float(operands[0]), float(operands[1]))]
        elif op == "l" and len(operands) >= 2:
            pts.append(_apply_ctm(ctm, float(operands[0]), float(operands[1])))
        elif op == "c" and len(operands) >= 6:
            p0 = pts[-1] if pts else _apply_ctm(ctm, 0.0, 0.0)
            p1 = _apply_ctm(ctm, float(operands[0]), float(operands[1]))
            p2 = _apply_ctm(ctm, float(operands[2]), float(operands[3]))
            p3 = _apply_ctm(ctm, float(operands[4]), float(operands[5]))
            pts.extend(_sample_cubic(p0, p1, p2, p3))
        elif op == "v" and len(operands) >= 4 and pts:
            p0 = pts[-1]
            p1 = p0
            p2 = _apply_ctm(ctm, float(operands[0]), float(operands[1]))
            p3 = _apply_ctm(ctm, float(operands[2]), float(operands[3]))
            pts.extend(_sample_cubic(p0, p1, p2, p3))
        elif op == "y" and len(operands) >= 4 and pts:
            p0 = pts[-1]
            p1 = _apply_ctm(ctm, float(operands[0]), float(operands[1]))
            p3 = _apply_ctm(ctm, float(operands[2]), float(operands[3]))
            p2 = p3
            pts.extend(_sample_cubic(p0, p1, p2, p3))
        elif op == "S":
            flush()
        elif op == "s":
            if pts:
                pts.append(pts[0])
            flush()
        elif op == "h":
            if len(pts) >= 2:
                pts.append(pts[0])        # close the subpath; a following S strokes it
        elif op in ("B", "B*", "b", "b*"):
            if op.startswith("b") and len(pts) >= 2:
                pts.append(pts[0])
            flush()                       # fill and stroke: the outline is still a stroke
        elif op in ("f", "F", "f*", "n"):
            pts = []
        elif op == "re":
            pts = []  # rectangles are frames / fills, not curve traces
    flush()
    return strokes


def extract_texts(page) -> list[TextItem]:
    """Positioned text via pypdf visitor (handles custom encodings better than raw Tj)."""
    items: list[TextItem] = []

    def visitor(text, cm, tm, font_dict, font_size):  # noqa: ARG001
        if not text or not str(text).strip():
            return
        x, y = float(tm[4]), float(tm[5])
        a, b, c, d, e, f = (float(v) for v in cm)
        items.append(TextItem(x=a * x + c * y + e, y=b * x + d * y + f,
                              text=str(text).strip(), size=float(font_size or 0.0)))

    page.extract_text(visitor_text=visitor)
    return items


def extract_page(reader: PdfReader, page_index: int) -> PageVectors:
    page = reader.pages[page_index]
    box = page.mediabox
    return PageVectors(
        page_index=page_index,
        width=float(box.width),
        height=float(box.height),
        strokes=extract_strokes(page),
        texts=extract_texts(page),
    )


def extract_document(path: str | Path, pages: list[int] | None = None) -> list[PageVectors]:
    """Extract vector content for selected pages (0-based). ``pages=None`` → all pages."""
    reader = PdfReader(str(path))
    idxs = pages if pages is not None else list(range(len(reader.pages)))
    return [extract_page(reader, i) for i in idxs]


def rasterize_page(path: str | Path, page_index: int, scale: float = 2.0) -> np.ndarray:
    """Optional raster fallback (pypdfium2, Apache/BSD). Returns RGB uint8 array (H, W, 3)."""
    try:
        import pypdfium2 as pdfium
    except ImportError as e:  # pragma: no cover
        raise ImportError("rasterize_page needs pypdfium2 (Apache/BSD): pip install pypdfium2") from e
    doc = pdfium.PdfDocument(str(path))
    page = doc[page_index]
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil()
    return np.asarray(pil.convert("RGB"))


# ---------------------------------------------------------------------------------------------
# Text runs with their own positions and sizes (for header and depth-label reading)

@dataclass
class TextRun:
    """One run of text in page coordinates (PDF user space, y up): baseline start ``x, y``, end
    ``x1`` (for upright text), font size ``size`` (page units). A run ends at a large gap: a wide TJ
    offset, wide character spacing or a new text position, so ``0`` and ``250`` printed apart are
    two runs.

    ``angle`` is the reading direction on the page, snapped to 0, 90, 180 or 270 degrees
    (counter-clockwise; -1 for slanted text), and ``ux1, uy1`` the baseline end point, so text on
    a sideways page keeps its true place; :func:`runs_in_frame` turns runs into another frame."""
    x: float
    y: float
    x1: float
    text: str
    size: float
    angle: int = 0
    ux1: float | None = None
    uy1: float | None = None

    @property
    def end(self) -> tuple[float, float]:
        return (self.x1, self.y) if self.ux1 is None else (self.ux1, self.uy1)


def frame_transform(box: tuple[float, float, float, float], rotation: int) -> np.ndarray:
    """3x3 matrix taking page user space to a frame turned by ``rotation`` degrees clockwise
    (0/90/180/270) with its origin at the lower-left corner of the turned page box."""
    th = np.deg2rad(-rotation)
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    R[np.abs(R) < 1e-12] = 0.0
    x0, y0, x1, y1 = box
    c = R @ np.array([[x0, x0, x1, x1], [y0, y1, y0, y1], [1, 1, 1, 1]], float)
    T = np.eye(3); T[0, 2] = -c[0].min(); T[1, 2] = -c[1].min()
    return T @ R


def apply_frame(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    return pts @ M[:2, :2].T + M[:2, 2]


def runs_in_frame(runs: list["TextRun"], M: np.ndarray, rotation: int) -> list["TextRun"]:
    """Text runs moved into the frame of :func:`frame_transform`. Runs that read left to right in
    that frame keep their exact baseline; other runs (e.g. numbers printed upright on a sideways
    log) become an upright stand-in centred on the same place, so their numbers still count."""
    out = []
    for r in runs:
        a = apply_frame(M, [[r.x, r.y]])[0]; b = apply_frame(M, [list(r.end)])[0]
        turn = (r.angle - rotation) % 360 if r.angle >= 0 else -1      # reading direction in the frame
        if turn == 0:
            out.append(TextRun(x=float(a[0]), y=float(a[1]), x1=float(b[0]), text=r.text, size=r.size))
            continue
        L = float(np.hypot(*(b - a))); cx, cy = (a + b) / 2.0
        if turn == 180:                                                # upside down: the ink sits below
            cy = cy - 0.72 * r.size
        elif turn in (90, 270):                                        # turned a quarter: centre of the glyphs
            d = (b - a) / max(L, 1e-9); nrm = np.array([-d[1], d[0]])
            cx, cy = np.array([cx, cy]) + nrm * 0.36 * r.size
            cy = cy - 0.36 * r.size
        out.append(TextRun(x=float(cx - L / 2), y=float(cy), x1=float(cx + L / 2), text=r.text, size=r.size))
    return out


def _tounicode(font) -> dict[int, str]:
    """Code -> text from a font's /ToUnicode CMap (bfchar and bfrange)."""
    import re
    try:
        data = font["/ToUnicode"].get_object().get_data().decode("latin-1")
    except Exception:
        return {}
    out: dict[int, str] = {}

    def u(h: str) -> str:
        b = bytes.fromhex(h)
        try:
            return b.decode("utf-16-be")
        except Exception:
            return ""
    for blk in re.findall(r"beginbfchar(.*?)endbfchar", data, re.S):
        for a, b in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", blk):
            out[int(a, 16)] = u(b)
    for blk in re.findall(r"beginbfrange(.*?)endbfrange", data, re.S):
        for a, b, rest in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[^\]]*\]|<[0-9A-Fa-f]*>)", blk):
            lo, hi = int(a, 16), int(b, 16)
            if rest.startswith("["):
                for k, h in enumerate(re.findall(r"<([0-9A-Fa-f]*)>", rest)):
                    out[lo + k] = u(h)
            else:
                base = bytes.fromhex(rest[1:-1])
                for k in range(hi - lo + 1):
                    v = int.from_bytes(base, "big") + k
                    try:
                        out[lo + k] = v.to_bytes(max(2, len(base)), "big").decode("utf-16-be")
                    except Exception:
                        pass
    return out


class _Font:
    """Just enough of a PDF font to decode codes and advance the text position."""

    def __init__(self, font) -> None:
        font = font.get_object() if font is not None else {}
        self.two_byte = str(font.get("/Subtype", "")) == "/Type0"
        self.uni = _tounicode(font) if "/ToUnicode" in font else {}
        self.widths: dict[int, float] = {}
        self.default = 500.0
        try:
            if self.two_byte:
                d = font["/DescendantFonts"][0].get_object()
                self.default = float(d.get("/DW", 1000))
                w = d.get("/W", [])
                i = 0
                while i < len(w):
                    a = int(w[i])
                    nxt = w[i + 1]
                    if isinstance(nxt, (list, ArrayObject)):
                        for k, v in enumerate(nxt):
                            self.widths[a + k] = float(v)
                        i += 2
                    else:
                        for c in range(a, int(nxt) + 1):
                            self.widths[c] = float(w[i + 2])
                        i += 3
            elif "/Widths" in font:
                fc = int(font.get("/FirstChar", 0))
                for k, v in enumerate(font["/Widths"]):
                    self.widths[fc + k] = float(v)
                fd = font.get("/FontDescriptor")
                if fd is not None:
                    self.default = float(fd.get_object().get("/MissingWidth", 0) or 0) or 500.0
            else:
                self.default = 556.0          # standard-14 fonts: Helvetica-like digit width
        except Exception:
            pass

    def glyphs(self, raw: bytes):
        """(code, text, width in 1/1000 em, is_space) per glyph."""
        if self.two_byte:
            codes = [int.from_bytes(raw[i:i + 2], "big") for i in range(0, len(raw) - 1, 2)]
        else:
            codes = list(raw)
        for c in codes:
            if c in self.uni:
                t = self.uni[c]
            elif self.two_byte:
                t = ""
            else:
                t = bytes([c]).decode("cp1252", errors="replace")
            yield c, t, self.widths.get(c, self.default), (c == 32 and not self.two_byte)


def extract_text_runs(page) -> list[TextRun]:
    """Positioned text runs from the page content stream (own text-state machine: Tm, Td, TD, T*,
    Tc, Tw, Tz, TL, Ts, TJ offsets). Runs break at gaps wider than 0.3 em, so values printed apart
    on one line come out as separate runs. Form XObjects are not entered."""
    contents = page.get_contents()
    if contents is None:
        return []
    fonts = {}
    try:
        res = page.get("/Resources") or {}
        fonts = (res.get_object().get("/Font") or {})
        fonts = fonts.get_object() if fonts else {}
    except Exception:
        fonts = {}
    cache: dict[str, _Font] = {}
    runs: list[TextRun] = []
    ctm = np.eye(3); stack: list = []
    tm = np.eye(3); tlm = np.eye(3)
    font = _Font(None); size = 1.0; tc = tw = 0.0; tz = 1.0; tl = 0.0; rise = 0.0
    cur: dict | None = None

    def mat(a, b, c, d, e, f):
        return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], float)

    def end_run():
        nonlocal cur
        if cur is not None and cur["text"].strip():
            ang = cur["angle"]
            if ang == 0:
                runs.append(TextRun(x=cur["ux"], y=cur["uy"], x1=cur["ex"], text=cur["text"].strip(), size=cur["size"]))
            else:
                runs.append(TextRun(x=cur["ux"], y=cur["uy"], x1=cur["ex"], text=cur["text"].strip(), size=cur["size"],
                                    angle=ang, ux1=cur["ex"], uy1=cur["ey"]))
        cur = None

    def direction():
        m = ctm @ tm
        u = np.array([m[0, 0], m[1, 0]]); n = float(np.hypot(*u)) or 1.0
        u = u / n
        a = float(np.degrees(np.arctan2(u[1], u[0]))) % 360
        q = int(round(a / 90.0)) % 4 * 90
        snapped = q if min(abs(a - q), 360 - abs(a - q)) < 2.0 else -1
        return u, snapped

    def user(px: float, py: float) -> tuple[float, float]:
        m = ctm @ tm
        v = m @ np.array([px, py, 1.0])
        return float(v[0]), float(v[1])

    def eff_size() -> float:
        m = ctm @ tm
        return float(abs(size) * np.hypot(m[0, 1], m[1, 1]))

    def show(raw: bytes):
        nonlocal tm, cur
        u, ang = direction()
        v = np.array([-u[1], u[0]])
        for _, t, w, sp in font.glyphs(raw):
            ux, uy = user(0.0, rise)
            # along / across the reading direction (= x / y for upright text)
            x0, y0 = ux * u[0] + uy * u[1], ux * v[0] + uy * v[1]
            adv = (w / 1000.0 * size + tc + (tw if sp else 0.0)) * tz
            if cur is not None and (cur["angle"] != ang or abs(y0 - cur["y"]) > 0.2 * max(cur["size"], 1e-6)
                                    or x0 - cur["x1"] > 0.3 * cur["size"] or x0 < cur["x1"] - 0.5 * cur["size"]):
                end_run()
            if cur is None:
                cur = dict(x=x0, y=y0, x1=x0, ux=ux, uy=uy, ex=ux, ey=uy, angle=ang, text="", size=eff_size())
            cur["text"] += t
            # glyph end (without the extra character spacing): where the ink stops
            gx, gy = user(w / 1000.0 * size * tz, rise)
            cur["x1"] = gx * u[0] + gy * u[1]; cur["ex"], cur["ey"] = gx, gy
            tm = tm @ mat(1, 0, 0, 1, adv, 0)
            if tc * tz > 0.3 * abs(size) and t.strip():
                end_run()                       # wide character spacing: every glyph stands alone

    for operands, operator in page_operations(page, contents):
        op = _op_name(operator)
        try:
            if op == "q":
                stack.append(ctm.copy())
            elif op == "Q":
                if stack:
                    ctm = stack.pop()
            elif op == "cm" and len(operands) >= 6:
                ctm = ctm @ mat(*(float(v) for v in operands[:6]))
            elif op == "BT":
                tm = np.eye(3); tlm = np.eye(3); end_run()
            elif op == "ET":
                end_run()
            elif op == "Tf" and len(operands) >= 2:
                name = str(operands[0])
                if name not in cache:
                    cache[name] = _Font(fonts.get(name) if hasattr(fonts, "get") else None)
                font = cache[name]; size = float(operands[1])
            elif op == "Tc" and operands:
                tc = float(operands[0])
            elif op == "Tw" and operands:
                tw = float(operands[0])
            elif op == "Tz" and operands:
                tz = float(operands[0]) / 100.0
            elif op == "TL" and operands:
                tl = float(operands[0])
            elif op == "Ts" and operands:
                rise = float(operands[0])
            elif op in ("Td", "TD") and len(operands) >= 2:
                tx, ty = float(operands[0]), float(operands[1])
                if op == "TD":
                    tl = -ty
                tlm = tlm @ mat(1, 0, 0, 1, tx, ty); tm = tlm.copy()
            elif op == "Tm" and len(operands) >= 6:
                tlm = mat(*(float(v) for v in operands[:6])); tm = tlm.copy()
            elif op == "T*":
                tlm = tlm @ mat(1, 0, 0, 1, 0, -tl); tm = tlm.copy()
            elif op == "Tj" and operands:
                show(bytes(operands[0].original_bytes if hasattr(operands[0], "original_bytes") else operands[0]))
            elif op in ("'", '"') and operands:
                if op == '"' and len(operands) >= 3:
                    tw, tc = float(operands[0]), float(operands[1])
                tlm = tlm @ mat(1, 0, 0, 1, 0, -tl); tm = tlm.copy()
                o = operands[-1]
                show(bytes(o.original_bytes if hasattr(o, "original_bytes") else o))
            elif op == "TJ" and operands:
                for o in operands[0]:
                    if isinstance(o, (bytes, str)) or hasattr(o, "original_bytes"):
                        show(bytes(o.original_bytes if hasattr(o, "original_bytes") else o))
                    else:
                        d = -float(o) / 1000.0 * size * tz
                        tm = tm @ mat(1, 0, 0, 1, d, 0)
                        if cur is not None and -float(o) / 1000.0 > 0.3:
                            end_run()
        except Exception:
            continue
    end_run()
    return runs
