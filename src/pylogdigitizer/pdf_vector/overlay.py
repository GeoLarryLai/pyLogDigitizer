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


# ----------------------------------------------------------------------------- on the real page
GRID_RGB, FRAME_RGB = (0, 190, 230), (70, 70, 70)
TAB10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
         (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]   # RGB


def _grid_paths(strokes):
    """All straight grid paths of the page in a neutral colour (black or grey, solid): the parser's
    grid classes keep black only, but many prints draw minor / vertical grid in grey."""
    gh, gv = [], []
    for s in strokes:
        if len(s.pts) < 2 or s.dash:
            continue
        c = np.asarray(s.color[:3] if s.color is not None else (0, 0, 0), float)
        if np.ptp(c) > 0.08 or c.mean() > 0.85:          # coloured or white: not grid
            continue
        dx, dy = float(np.ptp(s.pts[:, 0])), float(np.ptp(s.pts[:, 1]))
        if dy < 1.5 and dx > 30:
            gh.append(s)
        elif dx < 1.5 and dy > 30:
            gv.append(s)
    return gh, gv


def _line_offsets(dark: np.ndarray, P: np.ndarray, axis: str, win: int = 20, search: int = 4) -> np.ndarray:
    """|offset| (px) between a drawn polyline and the centre of the printed line under it, per window."""
    P = np.asarray(P, float)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if seg[-1] < 2:
        return np.zeros(0)
    t = np.arange(0, seg[-1], 1.0); D = np.c_[np.interp(t, seg, P[:, 0]), np.interp(t, seg, P[:, 1])]
    H, W = dark.shape; d = np.arange(-search, search + 1); out = []
    for s in range(0, len(D) - win // 2, win):
        Q = D[s:s + win]
        if axis == "v":
            yi = np.clip(np.round(Q[:, 1]).astype(int), 0, H - 1); n = Q[:, 0]
            prof = dark[yi[:, None], np.clip(np.round(n).astype(int)[:, None] + d, 0, W - 1)].sum(0)
        else:
            xi = np.clip(np.round(Q[:, 0]).astype(int), 0, W - 1); n = Q[:, 1]
            prof = dark[np.clip(np.round(n).astype(int)[:, None] + d, 0, H - 1), xi[:, None]].sum(0)
        k = int(np.argmax(prof))
        if prof[k] < 0.6 * len(Q):
            continue
        thr = 0.5 * prof[k]; a = k; b = k
        while a > 0 and prof[a - 1] >= thr:
            a -= 1
        while b < len(prof) - 1 and prof[b + 1] >= thr:
            b += 1
        w = prof[a:b + 1].astype(float); c = (np.arange(a, b + 1) * w).sum() / w.sum()
        out.append(abs(c - search + float(np.mean(np.round(n) - n))))
    return np.asarray(out)


def overlay_pdf_render(pdf_path, log, page_index: int, image: np.ndarray, dpi: float, out_path=None,
                       y_range: tuple[float, float] | None = None, title: str | None = None):
    """Overlay on a REAL rendering of the page (``image``: RGB uint8 rendered at ``dpi`` by any
    renderer, e.g. ``pdftoppm -r dpi``; rendering is not a runtime dependency). Every track's
    vector grid (orange), track frames (green) and each digitized curve's own source path (tab10,
    legend) are drawn in page pixel coordinates. ``y_range`` (PDF points) crops a zoom. Returns
    (RGB image, grid residual dict: |px| between each vector grid line and the rendered line)."""
    import cv2
    from pypdf import PdfReader
    pv = extract_page(PdfReader(str(pdf_path)), page_index)
    gh, gv = _grid_paths(pv.strokes)
    f = dpi / 72.0; Hpx, Wpx = image.shape[:2]
    sy = Hpx / (pv.height * f); sx = Wpx / (pv.width * f)          # renderer rounding of the page size
    to_px = lambda P: np.c_[np.asarray(P)[:, 0] * f * sx - 0.5, (pv.height - np.asarray(P)[:, 1]) * f * sy - 0.5]   # pixel centres
    dark = image.mean(2) < 200 if image.ndim == 3 else image < 200
    img = (image.astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pl = lambda P, col, w: cv2.polylines(img, [np.round(P * 16).astype(np.int32)], False, col, w, cv2.LINE_AA, 4)
    L = next((l for l in getattr(log, "layouts", []) if l.page_index == page_index), None)
    tracks = L.by_kind("track") if L is not None else []
    res = {}
    st = lambda v: None if not len(v) else dict(median=round(float(np.median(v)), 2),
                                                 p95=round(float(np.percentile(v, 95)), 2), max=round(float(np.max(v)), 2), n=len(v))
    bands = sorted({(round(r.x0, 1), round(r.x1, 1)) for r in tracks})
    ev = {b: [] for b in bands}; eh = {b: [] for b in bands}
    for ax_, s in [("v", q) for q in gv] + [("h", q) for q in gh]:   # every grid path, all tracks/panels
        P = to_px(s.pts); e = _line_offsets(dark, P, ax_); pl(P, GRID_RGB, 1)
        for b in bands:
            if ax_ == "v" and b[0] - 1 <= float(np.median(s.pts[:, 0])) <= b[1] + 1:
                ev[b].extend(e)
            elif ax_ == "h" and s.pts[:, 0].min() <= b[1] and s.pts[:, 0].max() >= b[0]:
                eh[b].extend(e)
    for k, b in enumerate(bands):
        lab = next((r.label for r in tracks if (round(r.x0, 1), round(r.x1, 1)) == b and r.label), f"track{k + 1}")
        res[f"x {b[0]:.0f}-{b[1]:.0f} pt ({lab})"] = dict(v=st(ev[b]), h=st(eh[b]))
    for r in tracks:
        R = to_px(np.array([[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1], [r.x0, r.y0]]))
        pl(R, FRAME_RGB, 2)
    names = []
    for i, t in enumerate(log.traces):
        paths = [p for pg, p in (t.info.get("page_paths") or []) if pg == page_index]
        if paths:
            col = TAB10[i % 10]; names.append((t.mnemonic, col))
            for p in paths:
                pl(to_px(p), col, 2)
    if y_range is not None:
        y0, y1 = sorted(((pv.height - y_range[1]) * f * sy, (pv.height - y_range[0]) * f * sy))
        img = img[max(0, int(y0)):int(y1)]
    img = _legend_band(img, title or f"{Path(pdf_path).name} page {page_index + 1}", names)
    if out_path:
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img, res


def _legend_band(img, title, names):
    import cv2
    s = max(0.6, img.shape[1] / 2000); hh = int(100 * s)
    band = np.full((hh, img.shape[1], 3), 255, np.uint8)
    cv2.putText(band, title, (10, int(35 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 0, 0), max(1, int(2 * s)), cv2.LINE_AA)
    items = names + [("grid (PDF vector paths)", GRID_RGB), ("track frames (parsed)", FRAME_RGB)]
    x = 10
    for txt, col in items:
        cv2.line(band, (x, int(72 * s)), (x + int(36 * s), int(72 * s)), col, max(2, int(4 * s)))
        cv2.putText(band, txt, (x + int(42 * s), int(80 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0), max(1, int(1.5 * s)), cv2.LINE_AA)
        x += int((56 + 11 * len(txt)) * s)
    return np.concatenate([band, img], 0)
