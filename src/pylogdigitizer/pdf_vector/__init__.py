# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""pdf_vector — digitize clean *digital-print* well logs (PDF, and later TIFF).

This path is for digitally exported prints that still carry vector strokes (or
high-res colour raster): Schlumberger/FORGE-style vertical PDFs and USGS GeoLog
Locator horizontal PDFs. Ink-on-paper scans stay on the existing scan workflow;
do not feed those here.

Uses pypdf (BSD) for path + text extraction — no AGPL PDF engines. Raster-only
pages can be handed to the scan workflow via :func:`rasterize_page` (optional
pypdfium2). A colour TIFF entry point can reuse the same DepthMap / ScaleMap /
LAS helpers once a high-res digital TIFF is available.

Shares :class:`~pylogdigitizer.depth_calibration.DepthMap`,
:class:`~pylogdigitizer.scale_calibration.ScaleMap`, :func:`~pylogdigitizer.las_export.resample`,
and :func:`~pylogdigitizer.las_export.write_las` with the scan pipeline.
"""
from __future__ import annotations

from pylogdigitizer.pdf_vector.extract import PageVectors, extract_page, extract_document
from pylogdigitizer.pdf_vector.calibrate import (
    depth_map_from_labels,
    detect_depth_axis,
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
    traces_from_page_horizontal,
    stitch_traces,
    apply_clip_nulls,
    OFF_CHART_WRAP_NOTE,
    detect_wrap_jumps,
)
from pylogdigitizer.pdf_vector.regions import Region, PageLayout, detect_page_layout, annotate_layout_figure
from pylogdigitizer.pdf_vector.digitize import digitize_pdf, DigitizedLog

__all__ = [
    "PageVectors",
    "extract_page",
    "extract_document",
    "depth_map_from_labels",
    "detect_depth_axis",
    "parse_scale_label",
    "parse_curve_headers",
    "parse_well_header",
    "tracks_from_grids",
    "legend_styles_from_strokes",
    "legend_colors_from_strokes",
    "CurveTrace",
    "classify_strokes",
    "traces_from_page",
    "traces_from_page_horizontal",
    "stitch_traces",
    "apply_clip_nulls",
    "OFF_CHART_WRAP_NOTE",
    "detect_wrap_jumps",
    "digitize_pdf",
    "DigitizedLog",
    "Region",
    "PageLayout",
    "detect_page_layout",
    "annotate_layout_figure",
]
