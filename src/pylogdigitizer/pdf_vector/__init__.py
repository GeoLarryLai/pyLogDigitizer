# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""pdf_vector: digitize vector (or high-res) well-log PDFs into depth/value series and LAS.

Uses pypdf (BSD) for path + text extraction — no AGPL PDF engines. Raster pages can be
handed to the existing scan workflow via :func:`rasterize_page` (optional pypdfium2).

Shares :class:`~pylogdigitizer.depth_calibration.DepthMap`,
:class:`~pylogdigitizer.scale_calibration.ScaleMap`, :func:`~pylogdigitizer.las_export.resample`,
and :func:`~pylogdigitizer.las_export.write_las` with the scan pipeline.
"""
from __future__ import annotations

from pylogdigitizer.pdf_vector.extract import PageVectors, extract_page, extract_document
from pylogdigitizer.pdf_vector.calibrate import (
    depth_map_from_labels,
    parse_scale_label,
    parse_curve_headers,
    parse_well_header,
    tracks_from_grids,
    legend_styles_from_strokes,
    legend_colors_from_strokes,
)
from pylogdigitizer.pdf_vector.assemble import (
    CurveTrace,
    classify_strokes,
    traces_from_page,
    stitch_traces,
    apply_clip_nulls,
)
from pylogdigitizer.pdf_vector.digitize import digitize_pdf, DigitizedLog

__all__ = [
    "PageVectors",
    "extract_page",
    "extract_document",
    "depth_map_from_labels",
    "parse_scale_label",
    "parse_curve_headers",
    "parse_well_header",
    "tracks_from_grids",
    "legend_styles_from_strokes",
    "legend_colors_from_strokes",
    "CurveTrace",
    "classify_strokes",
    "traces_from_page",
    "stitch_traces",
    "apply_clip_nulls",
    "digitize_pdf",
    "DigitizedLog",
]
