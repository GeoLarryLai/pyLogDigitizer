# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Mixed PDFs: a scanned picture with vector text (header, scales, depth numbers) printed over it.

:func:`mixed_header` reads the curve names, scales, units and depth numbers from the PDF text and
gives their positions in the pixels of the embedded picture, so the image workflow (clean-up, ink
mask and tracing of the picture) can use exact header values instead of OCR.
:func:`apply_mixed_header` puts them on curves that were traced from the picture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def mixed_header(path: str | Path, page: int = 0) -> dict:
    """Header and depth numbers from the vector text of a PDF page, in the embedded picture's pixels.

    Returns ``{image_size: (w, h), image_bbox_pt, headers: [{name, mnemonic, left, right, unit, kind,
    col0, col1, row}], depth_labels: [(row, depth)], notes}``. Columns and rows are pixels of the
    largest picture on the page (row 0 at its top), as loaded by
    :func:`pylogdigitizer.scan_io.pdf_embedded_image`.
    """
    from pypdf import PdfReader
    from pylogdigitizer.input_kind import _image_placements
    from pylogdigitizer.pdf_vector.vector_log import VectorPage, _merge_runs, _frame, _depth, _headers
    from pylogdigitizer.pdf_vector.extract import extract_strokes, extract_text_runs
    pg = PdfReader(str(path)).pages[page]
    imgs = [im for im in _image_placements(pg) if im[5] is not None]
    if not imgs:
        raise ValueError(f"{Path(path).name} page {page + 1}: no embedded picture (not a scan or mixed page)")
    x0, y0, x1, y1, m, xo = max(imgs, key=lambda im: (im[2] - im[0]) * (im[3] - im[1]))
    W, H = int(xo["/Width"]), int(xo["/Height"])
    if abs(m[0, 1]) > 1e-6 or abs(m[1, 0]) > 1e-6:
        raise ValueError("the picture is rotated on the page: header positions cannot be mapped to its pixels")
    box = pg.mediabox
    vp = VectorPage(page_index=page, width=float(box.width), height=float(box.height),
                    strokes=[s for s in extract_strokes(pg)], texts=_merge_runs([r for r in extract_text_runs(pg) if r.angle == 0]))
    _frame(vp)
    if vp.plot_top is not None and not len(vp.borders) >= 3:
        vp.plot_top = vp.plot_bottom = None; vp.borders = []
    _depth(vp); _headers(vp)
    # unit square -> page: x = m[0,0]*u + m[0,2], y = m[1,1]*v + m[1,2]; image row 0 is at v = 1
    col = lambda x: (np.asarray(x, float) - m[0, 2]) / m[0, 0] * W
    row = lambda y: (1.0 - (np.asarray(y, float) - m[1, 2]) / m[1, 1]) * H
    heads = [dict(name=e.name, mnemonic=e.mnemonic, left=e.left, right=e.right, unit=e.unit, kind=e.kind,
                  col0=float(col(e.x0)), col1=float(col(e.x1)), row=float(row(e.y))) for e in vp.headers]
    labels = [(float(row(y)), float(d)) for y, d in vp.depth_labels]
    return dict(image_size=(W, H), image_bbox_pt=(x0, y0, x1, y1), headers=heads, depth_labels=labels,
                notes=list(vp.notes))


def apply_mixed_header(curves: list[dict], tracks: dict, header: dict) -> list[str]:
    """Set name, unit and scale of traced curves from :func:`mixed_header` (the app's Open scan step
    for a mixed PDF). ``curves``: the app's curve dicts (``track``, ``name``, ``s`` session);
    ``tracks``: ``{track: (x_left, x_right)}`` in sheet pixels. Returns one line per change."""
    from pylogdigitizer.header_ocr import match_mnemonic
    done = []
    by_track: dict[str, list[dict]] = {}
    for c in curves:
        by_track.setdefault(c.get("track"), []).append(c)
    used: set[int] = set()
    for tr, cs in by_track.items():
        if tr not in tracks:
            continue
        xl, xr = tracks[tr]
        cand = [k for k, h in enumerate(header["headers"]) if k not in used
                and min(h["col1"], xr) - max(h["col0"], xl) > 0.5 * (xr - xl)]
        for c in cs:
            if not cand:
                break
            want = match_mnemonic(c.get("name", ""))[0]
            k = next((k for k in cand if want and match_mnemonic(header["headers"][k]["name"])[0] == want), None)
            if k is None:
                if len(cs) == 1 or len(cand) == 1:
                    k = cand[0]
                else:
                    continue
            h = header["headers"][k]; cand.remove(k); used.add(k)
            s = c["s"]
            s.set_scale(h["left"], h["right"], h["kind"], backup=s.scale.get("backup"), wrap_side=s.scale.get("wrap_side") or "right")
            c["name"] = s.curve = h["mnemonic"] or c["name"]; c["unit"] = h["unit"]
            c["header"] = h["name"]; c["review"] = [x for x in c.get("review", []) if not x.startswith(("name", "scale"))]
            c["header_source"] = "pdf text"
            done.append(f"{tr}: {h['name']} {h['left']:g} to {h['right']:g} {h['unit']} ({h['kind']})")
    return done
