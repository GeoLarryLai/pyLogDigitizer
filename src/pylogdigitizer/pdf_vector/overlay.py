"""Overlay QA for vector PDF digitization: the page's own vector content redrawn (light), the grid
lines and track frames the parser detected, and every digitized curve's source path coloured on
top. Matplotlib only (no PDF rasterizer needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pylogdigitizer.pdf_vector.extract import extract_page
from pylogdigitizer.pdf_vector.assemble import classify_strokes


def overlay_pdf_page(pdf_path: str | Path, log, page_index: int, out_path: str | Path | None = None,
                     y_range: tuple[float, float] | None = None, x_range: tuple[float, float] | None = None,
                     dpi: int = 110):
    """Draw page ``page_index`` of ``pdf_path`` with ``log`` (a DigitizedLog) overlaid.

    Light grey: all page strokes in their own colour (faded). Gold: detected grid lines. Green
    boxes: track frames. Thick coloured lines: each trace's source path on this page (legend shows
    mnemonics). ``y_range`` / ``x_range`` crop in PDF points. Returns the matplotlib Figure."""
    import matplotlib; matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pypdf import PdfReader
    pv = extract_page(PdfReader(str(pdf_path)), page_index)
    gh, gv, _ = classify_strokes(pv.strokes)
    x0, x1 = x_range or (0, pv.width); y0, y1 = y_range or (0, pv.height)
    w_in = max(4.0, (x1 - x0) / 72 * 1.2); h_in = max(4.0, (y1 - y0) / 72 * 1.2)
    f = min(1.0, 60 / max(w_in, h_in)); fig, ax = plt.subplots(figsize=(w_in * f, h_in * f), dpi=dpi)
    for s in pv.strokes:
        c = np.clip(np.asarray(s.color[:3] if s.color is not None else (0, 0, 0), float), 0, 1)
        ax.plot(s.pts[:, 0], s.pts[:, 1], color=tuple(0.35 * c + 0.65), lw=0.4)
    for s in gh + gv:
        ax.plot(s.pts[:, 0], s.pts[:, 1], color=(0.95, 0.75, 0.0), lw=0.5, alpha=0.8)
    L = next((l for l in getattr(log, "layouts", []) if l.page_index == page_index), None)
    if L is not None:
        for r in L.by_kind("track"):
            ax.add_patch(Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, ec=(0.1, 0.6, 0.1), lw=1.4))
    cmap = plt.cm.tab10; n = 0
    for i, t in enumerate(log.traces):
        paths = [p for pg, p in (t.info.get("page_paths") or []) if pg == page_index]
        if not paths:
            continue
        col = cmap(i % 10)
        for k, p in enumerate(paths):
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.6, label=t.mnemonic if k == 0 else None)
        n += 1
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_title(f"{Path(pdf_path).name} p{page_index}: {n} curves (coloured), grid (gold), tracks (green)", fontsize=8)
    if n:
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path)); plt.close(fig)
    return fig
