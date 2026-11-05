# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""End-to-end vector-PDF → dense traces → LAS (shared resample / write_las)."""
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
    legend_colors_from_strokes,
    CurveHeader,
)
from pylogdigitizer.pdf_vector.assemble import traces_from_page, stitch_traces, CurveTrace


@dataclass
class DigitizedLog:
    traces: list[CurveTrace]
    well: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    depth: np.ndarray | None = None
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    step: float = 0.5
    headers: list[CurveHeader] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def to_las(self, path: str | Path, step: float | None = None, null: float = -999.25) -> Path:
        step = self.step if step is None else step
        if self.depth is None or not self.curves:
            self.resample(step=step)
        return write_las(path, self.depth, self.curves, units=self.units, well=self.well, null=null)

    def resample(self, step: float = 0.5) -> None:
        """Resample every trace onto a common depth grid (0.5 or 0.25 ft). No smoothing.

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


def _log_pages(reader: PdfReader) -> list[int]:
    import re
    out = []
    for i, page in enumerate(reader.pages):
        texts = []
        def visitor(text, cm, tm, fd, fs):  # noqa: ARG001
            if text and str(text).strip():
                texts.append(str(text).strip())
        page.extract_text(visitor_text=visitor)
        n = sum(1 for t in texts if re.fullmatch(r"\d{3,5}", t) and 50 <= int(t) <= 50000)
        if n >= 4:
            out.append(i)
    return out


def _collect_headers(reader: PdfReader, page_indices: list[int]) -> list[CurveHeader]:
    """Gather curve headers from candidate pages (legend + log pages); first wins per mnemonic."""
    seen: set[str] = set()
    out: list[CurveHeader] = []
    # Prefer last pages (Schlumberger puts the scale legend in the tail) then first pages.
    order = list(dict.fromkeys(sorted(page_indices, reverse=True) + list(range(min(5, len(reader.pages))))
                               + list(range(max(0, len(reader.pages) - 5), len(reader.pages)))))
    for i in order:
        pv = extract_page(reader, i)
        for h in parse_curve_headers(pv.texts):
            if h.mnemonic in seen:
                continue
            seen.add(h.mnemonic)
            out.append(h)
        # Attach legend colours when swatches exist on this page.
        cols = legend_colors_from_strokes([h for h in out if h.color is None], pv.strokes)
        for h in out:
            if h.color is None and h.mnemonic in cols:
                h.color = cols[h.mnemonic]
    return out


def digitize_pdf(
    path: str | Path,
    pages: list[int] | None = None,
    step: float = 0.5,
    mnemonic_colors: dict[str, tuple[float, float, float]] | None = None,
    well_extra: dict[str, str] | None = None,
    headers: list[CurveHeader] | None = None,
) -> DigitizedLog:
    """Digitize vector log pages of a PDF into dense traces and a regular LAS-ready grid.

    Parameters
    ----------
    path : PDF path
    pages : 0-based indices; ``None`` auto-selects pages with ≥4 depth labels
    step : LAS depth step — ``0.5`` (default) or ``0.25`` ft
    mnemonic_colors : optional explicit mnemonic→RGB map
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
    colors = {h.mnemonic: h.color for h in doc_headers if h.color is not None}
    if mnemonic_colors:
        colors.update(mnemonic_colors)

    all_traces: list[CurveTrace] = []
    for i in pages:
        pv = extract_page(reader, i)
        all_traces.extend(traces_from_page(pv, headers=doc_headers, mnemonic_colors=colors or None))
    stitched = stitch_traces(all_traces)
    # Fill units from headers when missing.
    h_units = {h.mnemonic: h.unit for h in doc_headers}
    for t in stitched:
        if not t.unit:
            t.unit = h_units.get(t.mnemonic, "")
    log = DigitizedLog(
        traces=stitched, well=well, headers=doc_headers,
        info={"pages": pages, "n_raw_traces": len(all_traces), "n_headers": len(doc_headers)},
    )
    log.resample(step=step)
    return log
