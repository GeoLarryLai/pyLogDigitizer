# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""End-to-end digital-print PDF → dense traces → LAS (shared resample / write_las).

Handles vertical oilfield prints and horizontal USGS GeoLog Locator exports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from pylogdigitizer.las_export import resample, write_las
from pylogdigitizer.pdf_vector.extract import extract_page
from pylogdigitizer.pdf_vector.calibrate import (
    parse_well_header,
    parse_curve_headers,
    legend_styles_from_strokes,
    CurveHeader,
)
from pylogdigitizer.pdf_vector.assemble import traces_from_page, stitch_traces, CurveTrace
from pylogdigitizer.pdf_vector.regions import detect_page_layout, PageLayout


@dataclass
class DigitizedLog:
    traces: list[CurveTrace]
    well: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    depth: np.ndarray | None = None
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    step: float = 0.5
    headers: list[CurveHeader] = field(default_factory=list)
    layouts: list[PageLayout] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def to_las(self, path: str | Path, step: float | None = None, null: float = -999.25) -> Path:
        step = self.step if step is None else step
        if self.depth is None or not self.curves:
            self.resample(step=step)
        return write_las(path, self.depth, self.curves, units=self.units, well=self.well, null=null)

    def resample(self, step: float = 0.5) -> None:
        """Resample every trace onto a common depth grid (any step > 0 ft, default 0.5). No smoothing.

        Gaps larger than ``max_gap`` (default 4 steps in :func:`las_export.resample`) stay NaN,
        so sparse page coverage is not bridged by long interpolations.
        """
        self.step = step
        if not self.traces:
            self.depth = np.array([])
            self.curves = {}
            return
        series = {}
        for t in self.traces:
            g, v = resample(t.depth, t.value, step=step)
            series[t.mnemonic] = (g, v)
            self.units[t.mnemonic] = t.unit or self.units.get(t.mnemonic, "")
        d0 = min(g[0] for g, _ in series.values())
        d1 = max(g[-1] for g, _ in series.values())
        grid = np.arange(np.ceil(d0 / step) * step, d1 + step / 2, step)
        curves = {}
        for m, (g, v) in series.items():
            out = np.full(grid.shape, np.nan)
            # Place resampled samples on the common grid by exact depth match (both on ``step``).
            idx = np.searchsorted(grid, g)
            ok = (idx >= 0) & (idx < len(grid)) & np.isclose(grid[np.clip(idx, 0, len(grid) - 1)], g)
            out[idx[ok]] = v[ok]
            curves[m] = out
        self.curves = curves
        self.depth = grid


def _page_curve_stroke_score(reader: PdfReader, page_index: int) -> int:
    """How curve-like is this page? Deprioritize VDL/waveform pages (many short flats)."""
    from pylogdigitizer.pdf_vector.extract import extract_page
    import numpy as np
    pv = extract_page(reader, page_index)
    tall = 0
    short_flat = 0
    for s in pv.strokes:
        if len(s.pts) < 2:
            continue
        dx = float(np.ptp(s.pts[:, 0]))
        dy = float(np.ptp(s.pts[:, 1]))
        if dy > 40 and len(s.pts) >= 20:
            tall += 1
        elif dy < 8 and dx < 40:
            short_flat += 1
    # Waveform pages: thousands of short flats, few tall curves.
    if short_flat > 500 and tall < 5:
        return 0
    return tall


def _log_pages(reader: PdfReader) -> list[int]:
    """Pages that look like log plots (vertical oilfield or horizontal USGS).

    Prefers pages with depth ticks *and* tall curve strokes; skips VDL/waveform
    strips that only have depth labels.
    """
    import re
    candidates = []
    for i, page in enumerate(reader.pages):
        texts = []
        def visitor(text, cm, tm, fd, fs):  # noqa: ARG001
            if text and str(text).strip():
                texts.append(str(text).strip())
        page.extract_text(visitor_text=visitor)
        deep = sum(1 for t in texts if re.fullmatch(r"\d{3,5}", t) and 50 <= int(t) <= 50000)
        shallow = sum(1 for t in texts if re.fullmatch(r"\d{1,5}", t) and 0 <= int(t) <= 50000)
        if deep >= 4 or shallow >= 8:
            candidates.append(i)
    if not candidates:
        return []
    # Keep pages with real curves; skip expensive scoring when few candidates.
    if len(candidates) <= 12:
        scored = [(_page_curve_stroke_score(reader, i), i) for i in candidates]
    else:
        # Score a stride of pages, keep those plus neighbours of hits.
        stride = max(1, len(candidates) // 12)
        probe = candidates[::stride]
        scored = [(_page_curve_stroke_score(reader, i), i) for i in probe]
        hit = {i for s, i in scored if s >= 3}
        # Expand hits by ±1 in the candidate list.
        idx = {c: n for n, c in enumerate(candidates)}
        keep = set()
        for i in hit:
            n = idx[i]
            for j in range(max(0, n - 1), min(len(candidates), n + 2)):
                keep.add(candidates[j])
        if keep:
            return [c for c in candidates if c in keep]
    good = [i for s, i in scored if s >= 3]
    return good if good else candidates


def _collect_headers(reader: PdfReader, page_indices: list[int]) -> list[CurveHeader]:
    """Gather curve headers; apply legend styles from rich legend pages.

    Also scans body pages that carry scale labels (some digital prints only
    print scales on the first log panel, not a trailing legend sheet).
    """
    from dataclasses import replace as _dc_replace
    from pylogdigitizer.pdf_vector.calibrate import parse_scale_label

    scale_pages = []
    for i, page in enumerate(reader.pages):
        texts = []
        def visitor(text, cm, tm, fd, fs, _acc=texts):  # noqa: ARG001
            if text and str(text).strip():
                _acc.append(str(text).strip())
        try:
            page.extract_text(visitor_text=visitor)
        except Exception:
            continue
        n_sc = sum(1 for t in texts if parse_scale_label(t))
        if n_sc >= 2:
            scale_pages.append(i)

    order = list(dict.fromkeys(
        list(range(max(0, len(reader.pages) - 5), len(reader.pages)))
        + scale_pages
        + sorted(page_indices, reverse=True)
        + list(range(min(5, len(reader.pages))))
        + list(page_indices)
    ))
    seen: set[str] = set()
    out: list[CurveHeader] = []
    merged_styles: dict[str, dict] = {}
    for i in order:
        pv = extract_page(reader, i)
        page_headers = parse_curve_headers(pv.texts)
        for h in page_headers:
            if h.mnemonic in seen:
                continue
            seen.add(h.mnemonic)
            out.append(h)
        probe = [_dc_replace(h) for h in page_headers]
        got = legend_styles_from_strokes(probe, pv.strokes)
        # Keep styles from every rich legend/header bar; first assignment wins.
        if len(got) < 3:
            continue
        for mnem, style in got.items():
            if mnem not in merged_styles:
                merged_styles[mnem] = style
    by = {h.mnemonic: h for h in out}
    for mnem, style in merged_styles.items():
        h = by.get(mnem)
        if h is None:
            continue
        h.color = style["color"]
        h.width = style["width"]
        h.dash = style["dash"]
    return out


def digitize_pdf(
    path: str | Path,
    pages: list[int] | None = None,
    step: float = 0.5,
    mnemonic_colors: dict[str, tuple[float, float, float]] | None = None,
    mnemonic_styles: dict[str, dict] | None = None,
    well_extra: dict[str, str] | None = None,
    headers: list[CurveHeader] | None = None,
) -> DigitizedLog:
    """Digitize vector log pages of a PDF into dense traces and a regular LAS-ready grid.

    Parameters
    ----------
    path : PDF path
    pages : 0-based indices; ``None`` auto-selects pages with ≥4 depth labels
    step : LAS depth step in ft — ``0.5`` (default) or any step > 0
    mnemonic_colors : optional explicit mnemonic→RGB map (fallback; prefer styles)
    mnemonic_styles : optional mnemonic→{color,width,dash} override
    well_extra : extra / override WELL header fields
    headers : optional pre-parsed curve headers (else harvested from legend pages)
    """
    path = Path(path)
    reader = PdfReader(str(path))
    if pages is None:
        pages = _log_pages(reader)
        if not pages:
            pages = list(range(len(reader.pages)))

    well = {}
    for i in [0] + pages[:1]:
        well.update(parse_well_header(extract_page(reader, i).texts))
    if well_extra:
        well.update(well_extra)

    doc_headers = headers if headers is not None else _collect_headers(reader, pages)

    all_traces: list[CurveTrace] = []
    layouts: list[PageLayout] = []
    for i in pages:
        pv = extract_page(reader, i)
        page_traces = traces_from_page(
            pv, headers=doc_headers,
            mnemonic_colors=mnemonic_colors,
            mnemonic_styles=mnemonic_styles,
        )
        all_traces.extend(page_traces)
        layouts.append(detect_page_layout(pv, traces=page_traces))
    stitched = stitch_traces(all_traces)
    # Fill units from headers when missing.
    h_units = {h.mnemonic: h.unit for h in doc_headers}
    for t in stitched:
        if not t.unit:
            t.unit = h_units.get(t.mnemonic, "")
    log = DigitizedLog(
        traces=stitched, well=well, headers=doc_headers, layouts=layouts,
        info={"pages": pages, "n_raw_traces": len(all_traces), "n_headers": len(doc_headers),
              "axes": [L.axis for L in layouts]},
    )
    log.resample(step=step)
    return log
