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
