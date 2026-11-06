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


def extract_strokes(page) -> list[Stroke]:
    """Parse the page content stream into stroked polylines with graphics state."""
    contents = page.get_contents()
    if contents is None:
        return []
    cs = ContentStream(contents, page)
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

    for operands, operator in cs.operations:
        op = _op_name(operator)
        if op == "q":
            stack.append((ctm.copy(), color, width, dash))
        elif op == "Q" and stack:
            ctm, color, width, dash = stack.pop()
            pts = []
        elif op == "cm" and len(operands) >= 6:
            a, b, c, d, e, f = (float(x) for x in operands[:6])
            ctm = ctm @ np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], float)
        elif op in ("RG", "rg") and len(operands) >= 3:
            color = (float(operands[0]), float(operands[1]), float(operands[2]))
        elif op in ("G", "g") and operands:
            g = float(operands[0])
            color = (g, g, g)
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
        elif op in ("f", "F", "f*", "B", "B*", "b", "b*", "n", "h"):
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
