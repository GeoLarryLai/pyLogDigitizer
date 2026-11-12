# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Real tests for pdf_vector (tiny committed fixture — no big PDFs)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pypdf = pytest.importorskip("pypdf")

from pylogdigitizer.pdf_vector import (
    digitize_pdf,
    extract_page,
    parse_scale_label,
    parse_curve_headers,
    depth_map_from_labels,
    classify_strokes,
    traces_from_page,
)
from pylogdigitizer.pdf_vector.extract import extract_document
from pylogdigitizer.scale_calibration import ScaleMap
from pylogdigitizer.las_export import resample

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vector_log.pdf"


def test_parse_scale_linear_and_log():
    assert parse_scale_label("0 150 gAPI") == (0.0, 150.0, "gAPI", "linear")
    assert parse_scale_label("400 200 us") == (400.0, 200.0, "us", "linear")
    v0, v1, unit, kind = parse_scale_label("0.2 2000 ohm.m")
    assert (v0, v1, unit, kind) == (0.2, 2000.0, "ohm.m", "log")
    assert parse_scale_label("hello") is None


def test_fixture_extract_and_headers():
    assert FIXTURE.is_file() and FIXTURE.stat().st_size < 50_000
    pages = extract_document(FIXTURE, pages=[0])
    assert len(pages) == 1
    pv = pages[0]
    assert len(pv.strokes) >= 4
    headers = parse_curve_headers(pv.texts)
    assert any(h.mnemonic == "GR" for h in headers)
    dm = depth_map_from_labels(pv.texts)
    assert dm is not None
    assert list(dm.depths) == [100.0, 110.0, 120.0]
    # Depth increases as PDF y decreases.
    assert float(dm(-380)) == pytest.approx(100.0, abs=0.5)
    assert float(dm(-180)) == pytest.approx(120.0, abs=0.5)


def test_fixture_traces_and_las(tmp_path):
    log = digitize_pdf(FIXTURE, pages=[0], step=0.5)
    assert "GR" in log.curves
    assert log.depth[0] == pytest.approx(100.0)
    assert log.depth[-1] == pytest.approx(120.0)
    assert np.isfinite(log.curves["GR"]).sum() > 10
    # Dense connected points — no smoothing: resample is linear between raw samples only.
    tr = log.traces[0]
    assert tr.info["n_pts"] >= 20
    g, v = resample(tr.depth, tr.value, step=0.5)
    assert np.all(np.diff(g) == pytest.approx(0.5))
    out = log.to_las(tmp_path / "tiny.las")
    assert out.is_file()
    text = out.read_text(errors="ignore")
    assert "GR" in text and "DEPT" in text


def test_clip_nulls_at_scale_limit():
    scale = ScaleMap(0, 100, 0, 150, "linear")
    x = np.array([0.0, 50.0, 100.0, 49.0])
    v = scale(x)
    from pylogdigitizer.pdf_vector.assemble import apply_clip_nulls
    out = apply_clip_nulls(v, scale, x=x, tol_frac=0.02)
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert np.isfinite(out[1]) and np.isfinite(out[3])


def test_classify_distinguishes_grid_and_curve():
    from pylogdigitizer.pdf_vector.extract import Stroke
    grid = Stroke(pts=np.array([[0.0, 0.0], [0.0, 100.0]]), color=(0, 0, 0), width=1.0)
    curve = Stroke(
        pts=np.column_stack([np.linspace(10, 40, 30), np.linspace(0, 80, 30)]),
        color=(0.0, 0.75, 0.0), width=3.0,
    )
    gh, gv, curves = classify_strokes([grid, curve])
    assert len(gv) == 1 and len(curves) == 1
    assert curves[0].color == (0.0, 0.75, 0.0)


def test_style_distance_and_unwrap():
    from pylogdigitizer.pdf_vector.calibrate import style_distance
    from pylogdigitizer.pdf_vector.assemble import unwrap_log_wraps
    # Same colour, different width/dash → distinguishable.
    d_same = style_distance((0, 0, 0), 5.0, (), (0, 0, 0), 5.0, ())
    d_width = style_distance((0, 0, 0), 5.0, (), (0, 0, 0), 3.0, ())
    d_dash = style_distance((0, 0, 0), 3.0, (9.0, 19.0), (0, 0, 0), 3.0, ())
    assert d_same < d_width < 1.0
    assert d_dash > d_same

    # Fake wrap: left→mid then jump to left again after right edge.
    scale = ScaleMap(0, 100, 0.2, 2000, "log")
    # Decade = 10000; mid x=50 → geometric mid.
    xs = np.array([20.0, 40.0, 98.0, 1.0, 30.0])
    ys = np.array([500.0, 480.0, 460.0, 440.0, 420.0])
    pts = np.column_stack([xs, ys])
    vals = unwrap_log_wraps(pts, scale, edge_tol=3.0)
    # Edge samples null; after wrap, values should jump by ~decade vs unwrapped base.
    assert np.isnan(vals[2]) and np.isnan(vals[3])
    assert np.isfinite(vals[0]) and np.isfinite(vals[1]) and np.isfinite(vals[4])
    # Point after wrap (index 4) should be > one decade above its unwrapped scale value.
    base4 = float(scale(np.array([30.0]))[0])
    assert vals[4] > base4 * 50  # at least ~one decade up


def test_unwrap_continuity_linear():
    from pylogdigitizer.pdf_vector.assemble import unwrap_continuity
    # Deep on-scale 300, shallow wrapped as 475 (=75+400); span from scale 100-500.
    depth = np.array([100.0, 200.0, 3000.0, 3100.0])
    value = np.array([475.0, 480.0, 300.0, 310.0])  # shallow wrapped
    out = unwrap_continuity(depth, value, 100.0, 500.0, kind="linear")
    assert out[2] == pytest.approx(300.0)
    assert out[0] == pytest.approx(75.0, abs=1.0)
    assert out[1] == pytest.approx(80.0, abs=1.0)


def test_unwrap_continuity_preserves_on_scale():
    from pylogdigitizer.pdf_vector.assemble import unwrap_continuity
    depth = np.array([1000.0, 1100.0, 1200.0])
    value = np.array([140.0, 145.0, 150.0])
    out = unwrap_continuity(depth, value, 100.0, 500.0, kind="linear")
    assert np.allclose(out, value)


def test_usgs_scale_and_axis_helpers():
    """USGS digital-print quirks: spaced units, 48 In → 4..8, horizontal axis."""
    from pylogdigitizer.pdf_vector.calibrate import detect_depth_axis
    from pylogdigitizer.pdf_vector.extract import TextItem

    assert parse_scale_label("50 55 Deg F") == (50.0, 55.0, "Deg F", "linear")
    assert parse_scale_label("48 In") == (4.0, 8.0, "In", "linear")
    assert parse_scale_label("3 300 Ft^2/d")[3] == "linear"
    # Horizontal depth ticks along x.
    ticks = [TextItem(x=100 + 20 * i, y=12.0, text=str(10 * i), size=1.0) for i in range(0, 12)]
    assert detect_depth_axis(ticks) == "x"
    # Vertical depth ticks along y.
    ticks_v = [TextItem(x=20.0, y=500 - 30 * i, text=str(1000 + 50 * i), size=10.0) for i in range(6)]
    assert detect_depth_axis(ticks_v) == "y"


def test_page_layout_regions_api():
    from pylogdigitizer.pdf_vector.regions import detect_page_layout, Region, PageLayout
    from pylogdigitizer.pdf_vector.extract import extract_page
    from pypdf import PdfReader

    assert FIXTURE.is_file()
    reader = PdfReader(str(FIXTURE))
    pv = extract_page(reader, 0)
    layout = detect_page_layout(pv)
    assert isinstance(layout, PageLayout)
    assert layout.axis in {"x", "y"}
    kinds = {r.kind for r in layout.regions}
    # Tiny fixture is vertical with scales at top.
    assert "scale_legend" in kinds or "track" in kinds or "depth_axis" in kinds


def test_off_chart_wrap_detect_and_note():
    from pylogdigitizer.pdf_vector.assemble import detect_wrap_jumps, OFF_CHART_WRAP_NOTE
    from pylogdigitizer.pdf_vector.extract import Stroke
    scale = ScaleMap(0, 100, 0.2, 2000, "log")
    # Right-edge then jump to left (wrap).
    xs = np.array([30.0, 50.0, 97.0, 3.0, 25.0])
    ys = np.array([500.0, 480.0, 460.0, 440.0, 420.0])
    meta = detect_wrap_jumps(np.column_stack([xs, ys]), scale, edge_tol=4.0)
    assert meta["has_off_chart_wrap"] and meta["n_wraps"] >= 1
    assert "opposite" in OFF_CHART_WRAP_NOTE.lower() or "left" in OFF_CHART_WRAP_NOTE.lower()
    assert "decade" in OFF_CHART_WRAP_NOTE.lower()


def test_glued_scale_unit_and_thin_black_curve():
    from pylogdigitizer.pdf_vector.assemble import classify_strokes
    from pylogdigitizer.pdf_vector.extract import Stroke
    assert parse_scale_label("150 200US/CM") == (150.0, 200.0, "US/CM", "linear")
    assert parse_scale_label("60 80 OHM-M")[3] == "linear"
    assert parse_scale_label("0.2 2000 ohm.m")[3] == "log"
    # Thin black tall stroke counts as a curve (USGS-style).
    xs = np.linspace(30, 50, 80)
    ys = np.linspace(100, 700, 80)
    thin = Stroke(pts=np.column_stack([xs, ys]), color=(0.0, 0.0, 0.0), width=0.3)
    _gh, _gv, curves = classify_strokes([thin])
    assert len(curves) == 1


def test_overlay_pdf_page_draws_curve_paths(tmp_path):
    from pylogdigitizer.pdf_vector import digitize_pdf
    from pylogdigitizer.pdf_vector.overlay import overlay_pdf_page
    log = digitize_pdf(FIXTURE, pages=[0])
    assert log.traces and all(t.info.get("page_paths") for t in log.traces)
    for t in log.traces:                            # source path spans the trace's depth extent
        assert sum(len(p) for _, p in t.info["page_paths"]) >= 2
    fig = overlay_pdf_page(FIXTURE, log, 0, tmp_path / "ov.png")
    assert (tmp_path / "ov.png").stat().st_size > 1000
    labels = [l.get_label() for l in fig.axes[0].get_lines() if not l.get_label().startswith("_")]
    assert sorted(labels) == sorted(t.mnemonic for t in log.traces)


def test_overlay_on_rendered_page_grid_residual(tmp_path):
    """Grid vector paths land on the printed lines of a rendering (here: our own exact raster of the
    grid at 72 dpi), all tracks; curve paths are drawn in page pixels."""
    pytest.importorskip("cv2")
    from pypdf import PdfReader
    from pylogdigitizer.pdf_vector import digitize_pdf
    from pylogdigitizer.pdf_vector.extract import extract_page
    from pylogdigitizer.pdf_vector.overlay import overlay_pdf_render, _grid_paths
    log = digitize_pdf(FIXTURE, pages=[0])
    pv = extract_page(PdfReader(str(FIXTURE)), 0)
    dpi = 144; f = dpi / 72; Hp, Wp = int(round(pv.height * f)), int(round(pv.width * f))
    img = np.full((Hp, Wp, 3), 255, np.uint8)
    gh, gv = _grid_paths(pv.strokes)
    assert gh and gv
    for s in gh + gv:                                    # "render": 2-px lines centred on the path
        P = np.c_[s.pts[:, 0] * f - 0.5, (pv.height - s.pts[:, 1]) * f - 0.5]
        x0, x1 = sorted(P[:, 0][[0, -1]]); y0, y1 = sorted(P[:, 1][[0, -1]])
        img[int(round(y0 - 0.5)):int(round(y1 + 0.5)) + 1, int(round(x0 - 0.5)):int(round(x1 + 0.5)) + 1] = 0
    out, res = overlay_pdf_render(FIXTURE, log, 0, img, dpi, tmp_path / "r.png")
    assert (tmp_path / "r.png").exists() and out.shape[1] == Wp
    vals = [v["h"]["median"] for v in res.values() if v["h"]]
    assert vals and max(vals) <= 0.5


def test_legendless_header_does_not_steal_coloured_curve_pieces():
    """A green GR split into pieces by on-curve labels; a header without a legend
    style (STIT) used to take the leftover green pieces. Taken styles are now reserved."""
    from pylogdigitizer.pdf_vector.assemble import _best_chain_for_header, _style_key
    from pylogdigitizer.pdf_vector.calibrate import CurveHeader, TrackFrame
    from pylogdigitizer.pdf_vector.extract import Stroke
    y = np.linspace(100, 300, 60)
    piece = Stroke(pts=np.c_[50 + 10 * np.sin(y / 9), y], color=(0.0, 0.75, 0.0), width=3.0)
    stit = CurveHeader("STIT", 0, 50, "ft", "linear", x=170.0)
    fr = TrackFrame(0.0, 180.0)
    assert _best_chain_for_header(stit, fr, [piece], set()) is piece                  # old behaviour
    assert _best_chain_for_header(stit, fr, [piece], set(), taken_styles={_style_key(piece)}) is None
