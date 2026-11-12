# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Tell an image scan from a vector PDF before digitizing.

:func:`detect_input_kind` looks at the file (for a PDF: the paths, selectable text and placed
images of one page, with pypdf) and returns its kind and the method to use:

* ``scan``: an image (TIFF, PNG, JPEG, or a PDF page that is one large picture). Use the image
  workflow: clean-up, ink mask and curve tracing.
* ``vector``: a PDF page whose curves are vector paths with selectable text. Read the paths
  exactly with :func:`pylogdigitizer.pdf_vector.digitize_vector`; no tracing.
* ``mixed``: a scanned picture with vector text (or header lines) printed over it. Take the
  header and scales from the PDF text (:func:`pylogdigitizer.pdf_vector.mixed_header`) and
  trace the curves from the image with the image workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

METHODS = {
    "scan": ("trace", "image workflow: clean-up, ink mask and curve tracing"),
    "vector": ("vector", "vector workflow: read the curve paths exactly, no tracing"),
    "mixed": ("mixed", "header and scales from the PDF text, curves traced from the image"),
}


@dataclass
class InputKind:
    """Result of :func:`detect_input_kind`. ``evidence`` holds the numbers behind the call:
    ``n_paths`` (stroked paths), ``n_curve_paths`` (paths with 20 or more points), ``text_chars``
    (selectable characters), ``image_coverage`` (page fraction covered by the largest placed
    image), ``n_images``, and for a multi-page PDF ``curve_pages`` (1-based pages with curves)."""
    kind: str
    method: str
    method_label: str
    path: str
    page: int = 0
    n_pages: int = 1
    evidence: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(kind=self.kind, method=self.method, method_label=self.method_label, path=self.path,
                    page=self.page, n_pages=self.n_pages, evidence=dict(self.evidence), notes=list(self.notes))

    def __str__(self) -> str:
        ev = self.evidence
        bits = [f"{Path(self.path).name}: {self.kind}"]
        if "image_coverage" in ev:
            bits.append(f"page {self.page + 1}/{self.n_pages}, {ev.get('n_paths', 0)} paths "
                        f"({ev.get('n_curve_paths', 0)} curve-like), {ev.get('text_chars', 0)} text chars, "
                        f"image coverage {ev.get('image_coverage', 0):.0%}")
        elif "size_px" in ev:
            bits.append(f"{ev['size_px'][0]}x{ev['size_px'][1]} px {ev.get('format', '')}")
        s = " · ".join(bits) + f"\n  recommended: {self.method_label}"
        for n in self.notes:
            s += f"\n  note: {n}"
        return s


def _image_placements(page, max_depth: int = 3) -> list[tuple]:
    """Every image drawn on the page (Do of an image XObject, inline images, and images inside
    form XObjects): ``(x0, y0, x1, y1, ctm, xobject)`` with the page bounding box, the 3x3 matrix
    that maps the unit square onto the page and the image XObject (None for inline images)."""
    from pypdf.generic import ContentStream

    out: list[tuple[float, float, float, float]] = []

    def bbox(m, xo=None):
        pts = np.array([m @ np.array([x, y, 1.0]) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1))])
        return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()),
                m.copy(), xo)

    def walk(stream_owner, resources, ctm0, depth):
        try:
            contents = stream_owner.get_contents() if hasattr(stream_owner, "get_contents") else stream_owner
            if contents is None:
                return
            cs = ContentStream(contents, stream_owner if hasattr(stream_owner, "get_contents") else None)
        except Exception:
            return
        xobj = {}
        try:
            xobj = (resources or {}).get("/XObject") or {}
            xobj = xobj.get_object() if hasattr(xobj, "get_object") else xobj
        except Exception:
            xobj = {}
        ctm = ctm0.copy(); stack = []
        for operands, operator in cs.operations:
            op = operator.decode("latin-1") if isinstance(operator, bytes) else str(operator)
            if op == "q":
                stack.append(ctm.copy())
            elif op == "Q":
                if stack:
                    ctm = stack.pop()
            elif op == "cm" and len(operands) >= 6:
                a, b, c, d, e, f = (float(v) for v in operands[:6])
                ctm = ctm @ np.array([[a, c, e], [b, d, f], [0, 0, 1.0]])
            elif op == "INLINE IMAGE":
                out.append(bbox(ctm))
            elif op == "Do" and operands:
                try:
                    xo = xobj[str(operands[0])].get_object()
                except Exception:
                    continue
                sub = str(xo.get("/Subtype", ""))
                if sub == "/Image":
                    out.append(bbox(ctm, xo))
                elif sub == "/Form" and depth < max_depth:
                    m = [float(v) for v in xo.get("/Matrix", [1, 0, 0, 1, 0, 0])]
                    fm = np.array([[m[0], m[2], m[4]], [m[1], m[3], m[5]], [0, 0, 1.0]])
                    walk(xo, xo.get("/Resources") or resources, ctm @ fm, depth + 1)

    try:
        res = page.get("/Resources")
        res = res.get_object() if res is not None else {}
    except Exception:
        res = {}
    walk(page, res, np.eye(3), 0)
    return out


def _pdf_page_evidence(page) -> dict:
    from pylogdigitizer.pdf_vector.extract import extract_strokes, extract_text_runs
    box = page.mediabox
    W, H = float(box.width), float(box.height)
    x00, y00 = float(box.left), float(box.bottom)
    try:
        strokes = extract_strokes(page)
    except Exception:
        strokes = []
    try:
        runs = extract_text_runs(page)
        chars = sum(len(r.text.replace(" ", "")) for r in runs)
    except Exception:
        chars = 0
    if chars == 0:
        try:
            chars = len((page.extract_text() or "").replace(" ", "").replace("\n", ""))
        except Exception:
            pass
    imgs = _image_placements(page)
    cover = []
    for x0, y0, x1, y1, *_ in imgs:
        w = max(0.0, min(x1, x00 + W) - max(x0, x00)); h = max(0.0, min(y1, y00 + H) - max(y0, y00))
        cover.append(w * h / max(W * H, 1e-9))
    return dict(n_paths=len(strokes), n_curve_paths=sum(len(s.pts) >= 20 for s in strokes), text_chars=int(chars),
                image_coverage=round(float(max(cover, default=0.0)), 4), n_images=len(imgs),
                page_size_pt=[round(W, 1), round(H, 1)])


def _classify(ev: dict) -> tuple[str, list[str]]:
    notes = []
    big = ev["image_coverage"] >= 0.4
    curves = ev["n_curve_paths"] >= 3 or (ev["n_curve_paths"] >= 1 and not big)
    text = ev["text_chars"] >= 20
    if curves and (text or not big):
        if not text:
            notes.append("no selectable text: depth and scales cannot be read from the file")
        return "vector", notes
    if big:
        if text or ev["n_paths"] >= 10:
            return "mixed", notes
        return "scan", notes
    if text:
        notes.append("text and lines but no curve paths on this page")
        return "vector", notes
    notes.append("no curve paths, text or large pictures on this page")
    return "scan", notes


def detect_input_kind(path, page: int = 0) -> InputKind:
    """Kind of input file and the recommended method: ``scan``, ``vector`` or ``mixed``.

    ``page`` is the 0-based PDF page to inspect (ignored for images). TIFF, PNG and JPEG files are
    always ``scan``. For a PDF page: vector paths plus selectable text mean ``vector``; one large
    embedded image means ``scan``; a large image with vector text or header lines over it means
    ``mixed``. For a multi-page PDF, ``evidence['curve_pages']`` lists the pages that carry curve
    paths, so a header page can point you to the log page.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf in IMAGE_SUFFIXES or suf != ".pdf":
        ev: dict = {}
        notes = []
        try:
            from PIL import Image
            with Image.open(p) as im:
                ev = dict(format=im.format, size_px=list(im.size), mode=im.mode, n_frames=int(getattr(im, "n_frames", 1)))
        except Exception as ex:
            if suf not in IMAGE_SUFFIXES:
                raise ValueError(f"{p.name}: not a PDF or an image file this package can read ({ex})") from ex
            notes.append(f"could not read the image header: {ex}")
        m, lab = METHODS["scan"]
        return InputKind("scan", m, lab, str(p), 0, 1, ev, notes)
    from pypdf import PdfReader
    reader = PdfReader(str(p))
    n = len(reader.pages)
    if not 0 <= page < n:
        raise ValueError(f"{p.name} has {n} page(s); page {page} (0-based) does not exist")
    ev = _pdf_page_evidence(reader.pages[page])
    kind, notes = _classify(ev)
    if n > 1 and n <= 30:
        cp = []
        for i in range(n):
            try:
                e2 = ev if i == page else _pdf_page_evidence(reader.pages[i])
            except Exception:
                continue
            if e2["n_curve_paths"] >= 3:
                cp.append(i + 1)
        ev["curve_pages"] = cp
        if kind == "vector" and ev["n_curve_paths"] < 3 and cp:
            notes.append(f"the curves are on page(s) {', '.join(map(str, cp))}: pass page={cp[0] - 1}")
    m, lab = METHODS[kind]
    return InputKind(kind, m, lab, str(p), page, n, ev, notes)
