# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Colour-scan path: palette suggestion, per-curve ink, header line parser, pairs, wrap fills."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylogdigitizer.scan_color import (
    COLOR_TRACE_PRESET, classify_colors, find_log_body, is_color_rgb, legend_color,
    require_palette, suggest_palette,
)
from pylogdigitizer.curve_ink import black_curve_only, make_base_ink, make_curve_ink, thin_vertical_lines
from pylogdigitizer.curve_pairs import shared_scale_pairs, splice_pairs
from pylogdigitizer.header_ocr import apply_curve_fixes, merge_backup_rows, printed_name, propose_curves_from_lines
from pylogdigitizer.las_export import common_grid
from pylogdigitizer.scale_wraparound import default_backup, fill_backups
from pylogdigitizer.scan_io import is_color_scan, load_rgb, pdf_embedded_image


def _synth_rgb(h=240, w=360):
    """White paper, black bar, solid green / red / blue blocks (easy for k-means)."""
    rgb = np.full((h, w, 3), 248, np.uint8)
    rgb[:, :12] = (0, 0, 0)
    rgb[40:200, 40:110] = (36, 158, 52)
    rgb[40:200, 140:210] = (196, 42, 42)
    rgb[40:200, 240:310] = (42, 78, 204)
    return rgb


def test_color_trace_preset_does_not_change_defaults():
    from pylogdigitizer.curve_tracing import trace_track
    import inspect
    sig = inspect.signature(trace_track)
    assert sig.parameters["min_area"].default == 400
    assert sig.parameters["straight_len"].default == 121
    assert sig.parameters["center_weight"].default == 0.0
    assert COLOR_TRACE_PRESET["min_area"] == 0
    assert COLOR_TRACE_PRESET["straight_len"] == 0
    assert COLOR_TRACE_PRESET["min_width"] == 5


def test_require_palette_and_is_color_rgb():
    with pytest.raises(ValueError, match="white"):
        require_palette({"black": (0, 0, 0), "red": (200, 0, 0)})
    assert require_palette({"white": (255, 255, 255), "black": (0, 0, 0)}) == ["white", "black"]
    assert not is_color_rgb(np.full((80, 80, 3), 240, np.uint8))
    rgb = _synth_rgb()
    assert is_color_rgb(rgb)


def test_suggest_and_classify_palette():
    rgb = _synth_rgb()
    pal = suggest_palette(rgb, max_colors=8, min_share=0.001)
    assert "white" in pal and "black" in pal
    chroma = [n for n in pal if n not in ("white", "black")]
    assert len(chroma) >= 2
    labels, names = classify_colors(rgb, pal)
    assert labels.shape == rgb.shape[:2]
    green = next(n for n in names if n.startswith("green"))
    red = next(n for n in names if n.startswith("red"))
    gi, ri = names.index(green), names.index(red)
    assert (labels[80:160, 50:100] == gi).mean() > 0.8
    assert (labels[80:160, 150:200] == ri).mean() > 0.8
    r0, r1 = find_log_body(labels, names)
    assert r1 - r0 > 50


def test_legend_color():
    names = ["white", "black", "green", "red"]
    lab = np.zeros((80, 120), np.uint8)
    lab[30:38, 20:100] = 2
    assert legend_color(lab, names, 10, 25, (10, 110), below=20, above=2) == "green"


def test_curve_ink_isolates_one_colour(tmp_path):
    rgb = _synth_rgb()
    pal = suggest_palette(rgb)
    labels, names = classify_colors(rgb, pal)
    base = make_base_ink(labels, names, (0, rgb.shape[0]))
    assert base.shape == labels.shape
    assert base[:, :12].mean() > 0.9
    green = next(n for n in names if n.startswith("green"))
    ink = make_curve_ink(base, labels, names, green, (20, 130), (0, rgb.shape[0]),
                         gap_close_px=3, thicken_px=1, margin=2, thin_grid_px=0)
    assert ink[80:160, 50:100].mean() > 0.8
    red = next(n for n in names if n.startswith("red"))
    ink_r = make_curve_ink(base, labels, names, red, (130, 230), (0, rgb.shape[0]),
                           gap_close_px=3, thicken_px=1, margin=2, thin_grid_px=0)
    assert ink_r[80:160, 50:100].mean() < 0.2
    from pylogdigitizer.curve_ink import save_ink_tif
    p = save_ink_tif(ink, tmp_path / "g.tif")
    assert p.exists() and p.stat().st_size > 0


def test_thin_vertical_lines_and_black_curve():
    block = np.zeros((80, 60), bool)
    block[:, 20:28] = True
    thin = thin_vertical_lines(block, keep_px=2, frac=0.5)
    assert thin[:, 20:28].sum() < block[:, 20:28].sum()
    assert thin.any()
    names = ["white", "black"]
    lab = np.zeros((50, 40), np.uint8)
    lab[:, 10] = 1
    lab[5:45, 20] = 1
    only = black_curve_only(lab, names)
    assert only.shape == lab.shape


def test_printed_names_kept():
    assert printed_name("RES(16N)") == "RES16N"
    assert printed_name("RES(64N)") == "RES64N"
    assert printed_name("LATERAL") == "LATERAL"
    assert printed_name("GAMMA") == "GR"
    assert printed_name("PORNEU") == "NPHI"


def _w(text, x0, y0, x1=None, y1=None):
    return dict(text=text, x0=x0, y0=y0, x1=x1 or x0 + 50, y1=y1 or y0 + 14)


def test_propose_curves_from_lines_keeps_distinct_resistivities():
    tracks = {"track2": dict(x=(100, 520), grid="log", words=[
        _w("RES(16N)", 120, 20), _w("0.2", 120, 42), _w("OHMM", 210, 42), _w("20", 360, 42),
        _w("RES(64N)", 120, 70), _w("0.2", 120, 92), _w("OHMM", 210, 92), _w("20", 360, 92),
        _w("LATERAL", 120, 120), _w("0.2", 120, 142), _w("OHMM", 210, 142), _w("20", 360, 142),
    ])}
    rows = propose_curves_from_lines(tracks)
    names = [r["name"] for r in rows]
    assert names == ["RES16N", "RES64N", "LATERAL"]
    assert all(r["left"] == 0.2 and r["right"] == 20 and r["unit"] == "OHMM" for r in rows)
    assert all(r["track"] == "track2" and r["scale_type"] == "log" for r in rows)


def test_apply_curve_fixes_and_merge_backup():
    rows = [dict(name="GR", label="GAMMA", track="track1", color="green", left=0, right=150,
                 scale_type="linear", backup=None, unit="GAPI", check="")]
    out, unknown = apply_curve_fixes(rows, {"GR": {"right": 160}})
    assert out[0]["right"] == 160 and out[0]["check"] == "fixed by caller" and unknown == []
    two = [
        dict(name="GR", track="track1", color="green", left=0, right=150, scale_type="linear",
             backup=None, unit="GAPI"),
        dict(name="GRBK", track="track1", color="green", left=150, right=300, scale_type="linear",
             backup=None, unit="GAPI"),
    ]
    merged = merge_backup_rows(two)
    assert len(merged) == 1 and merged[0]["backup"] == (150, 300) and merged[0]["name"] == "GR"


def test_shared_scale_pairs_and_splice_opt_in():
    rows = [
        dict(name="GR", track="track1", color="red", scale_type="linear", unit="GAPI", left=0, right=150),
        dict(name="GRHI", track="track1", color="red", scale_type="linear", unit="GAPI", left=0, right=2000),
        dict(name="SP", track="track1", color="black", scale_type="linear", unit="MV", left=-160, right=40),
    ]
    pairs = shared_scale_pairs(rows)
    assert len(pairs) == 1 and {pairs[0][0]["name"], pairs[0][1]["name"]} == {"GR", "GRHI"}
    vn = np.array([40.0, 80.0, np.nan])
    vw = np.array([180.0, 90.0, 220.0])
    out_rows, curves = splice_pairs(rows, {"GR": vn, "GRHI": vw})
    spliced = [r for r in out_rows if r.get("spliced")]
    assert len(spliced) == 1
    assert np.allclose(curves[spliced[0]["name"]], [180.0, 80.0, 220.0], equal_nan=True)


def test_default_backup_and_fill_backups_flagged():
    assert default_backup(dict(left=0, right=150, scale_type="linear")) == (150, 300)
    assert default_backup(dict(left=0.2, right=20, scale_type="log")) == pytest.approx((20, 2000))
    names = ["white", "black", "green", "yellow"]
    lab = np.zeros((180, 240), np.uint8)
    lab[30:150, 0:90] = 3
    lab[30:150, 90:98] = 2
    fills, reviews = fill_backups(lab, names, ["yellow"], (0, 239), (0, 180))
    assert "green" in fills
    rs, xs = fills["green"]
    assert np.isfinite(xs).sum() > 20
    assert "green" in reviews and "review" in reviews["green"].lower()
    assert "shading-backup" in reviews["green"]


def test_common_grid_no_smoothing():
    d1 = np.array([100.0, 100.5, 101.0]); v1 = np.array([1.0, 2.0, 3.0])
    d2 = np.array([100.5, 101.0, 101.5]); v2 = np.array([4.0, 5.0, 6.0])
    grid, curves = common_grid({"A": (d1, v1), "B": (d2, v2)}, step=0.5)
    assert grid[0] == pytest.approx(100.0) and grid[-1] == pytest.approx(101.5)
    assert curves["A"][0] == pytest.approx(1.0) and np.isnan(curves["A"][-1])
    assert np.isnan(curves["B"][0]) and curves["B"][-1] == pytest.approx(6.0)
    g, _ = common_grid({"A": (d1, v1)}, step=0.1)          # any step > 0 ft
    assert np.allclose(np.diff(g), 0.1)
    for bad in (0, -0.5, float("nan"), "abc"):
        with pytest.raises(ValueError, match="greater than 0"):
            common_grid({"A": (d1, v1)}, step=bad)


def test_load_rgb_and_is_color_scan(tmp_path):
    from PIL import Image
    rgb = _synth_rgb()
    p = tmp_path / "c.tif"
    Image.fromarray(rgb).save(p)
    got = load_rgb(p)
    assert got.shape == rgb.shape and got.dtype == np.uint8
    assert is_color_scan(p)
    bw = tmp_path / "bw.tif"
    Image.fromarray(np.full((40, 40), 255, np.uint8), mode="L").convert("1").save(bw)
    assert not is_color_scan(bw)
    jpg = tmp_path / "c.jpg"
    Image.fromarray(rgb).save(jpg, quality=95)
    assert is_color_scan(jpg)
    pal = tmp_path / "p.tif"
    Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=16).save(pal)
    assert is_color_scan(pal)


def test_pdf_embedded_image_rejects_blank(tmp_path):
    from pypdf import PdfWriter
    pdf = tmp_path / "blank.pdf"
    w = PdfWriter(); w.add_blank_page(width=200, height=200)
    w.write(pdf)
    with pytest.raises(ValueError, match="embedded image"):
        pdf_embedded_image(pdf)


def test_prepare_color_scan_manual_palette(tmp_path):
    from PIL import Image
    from pylogdigitizer.workflows import prepare_color_scan
    rgb = _synth_rgb()
    p = tmp_path / "c.tif"
    Image.fromarray(rgb).save(p)
    pal = {"white": (255, 255, 255), "black": (0, 0, 0), "green": (36, 158, 52),
           "red": (196, 42, 42), "blue": (42, 78, 204)}
    prep = prepare_color_scan(p, palette=pal)
    assert prep["names"][:2] == ["white", "black"]
    assert prep["labels"].shape == rgb.shape[:2]
    assert prep["base"].any()


class _DM:
    def __call__(self, row):
        return 4000.0 + float(row) / 10.0
    def inverse(self, depth):
        return (float(depth) - 4000.0) * 10.0


def _plot_fixture():
    h, w = 220, 160
    rgb = np.full((h, w, 3), 245, np.uint8)
    rgb[:, 8:12] = 20
    rgb[20:200, 16:50, 1] = 170
    rgb[20:200, 90:140, 0] = 200
    pal = {"white": (255, 255, 255), "black": (0, 0, 0), "green": (0, 170, 0), "red": (200, 0, 0)}
    curves = [
        dict(name="GR", color="green", track="track1", left=0, right=150, scale_type="linear"),
        dict(name="RILD", color="red", track="track2", left=0.2, right=200, scale_type="log"),
    ]
    d = np.linspace(4002, 4018, 40)
    traces = {"GR": (d, 24 + 6 * np.sin(d / 3)), "RILD": (d, 108 + 8 * np.cos(d / 4))}
    tracks = {"track1": (16, 50), "track2": (90, 140)}
    ink = {"GR": np.full((h, w), 255, np.uint8), "RILD": np.full((h, w), 255, np.uint8)}
    for i, row in enumerate(np.linspace(20, 200, 40)):
        ink["GR"][int(row), int(round(24 + 6 * np.sin(d[i] / 3)))] = 0
        ink["RILD"][int(row), int(round(108 + 8 * np.cos(d[i] / 4)))] = 0
    notes = [dict(track="track1", label="GR 0–150 (OCR)"),
             dict(track="track2", label="RILD 0.2–200 (OCR)")]
    return rgb, pal, curves, traces, tracks, ink, notes


def test_plot_color_autodetect_is_wide():
    from pylogdigitizer.scan_color import plot_color_autodetect
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb, pal, curves, traces, tracks, _, notes = _plot_fixture()
    fig = plot_color_autodetect(rgb, _DM(), (4002, 4018), pal, curves, traces, tracks,
                                ocr_notes=notes, title="test")
    assert fig.get_figwidth() > fig.get_figheight()
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_color_trace_pairs_share_limits():
    from pylogdigitizer.scan_color import plot_color_trace_pairs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, pal, curves, traces, tracks, ink, _ = _plot_fixture()
    fig = plot_color_trace_pairs(ink, traces, curves, _DM(), (4002, 4018), tracks, pal,
                                 title="test")
    assert len(fig.axes) == 4
    for i in range(0, 4, 2):
        assert fig.axes[i].get_xlim() == fig.axes[i + 1].get_xlim()
        assert fig.axes[i].get_ylim() == fig.axes[i + 1].get_ylim()
    plt.close(fig)


STONE = Path(__file__).resolve().parents[1] / "example_data" / "STONE_5-19_15-055-22611" / "triple_combo_color_crop.tif"


@pytest.mark.skipif(not STONE.exists(), reason="example data not present")
def test_default_header_parser_one_row_per_curve_on_stone():
    """Default parser: each printed curve row is its own proposal with its own scale (the three
    resistivities and the two porosities are not lumped into one row)."""
    from pylogdigitizer.depth_calibration import ocr_available
    if not ocr_available():
        pytest.skip("RapidOCR not installed")
    from pylogdigitizer.header_ocr import propose_header
    from pylogdigitizer.scan_io import load_rgb
    from pylogdigitizer.scan_color import suggest_palette, classify_colors
    from pylogdigitizer.workflows import analyze_layout
    rgb = load_rgb(STONE)
    labels, names = classify_colors(rgb, suggest_palette(rgb))
    H = propose_header(str(STONE), layout=analyze_layout(STONE), labels=labels, label_names=names)
    got = {(c["track"], c["mnemonic"]["value"]): c for c in H["curves"]}
    for m in ("ILD", "ILM", "SFL"):
        c = got[("track2", m)]
        assert (c["scale_left"]["value"], c["scale_right"]["value"], c["scale_type"]["value"]) == (0.2, 200.0, "log")
    for m in ("NPHI", "DPHI"):
        c = got[("track3", m)]
        assert (c["scale_left"]["value"], c["scale_right"]["value"]) == (30.0, -10.0)
    assert ("track2", "RES") not in got
    assert (got[("track1", "GR")]["scale_left"]["value"], got[("track1", "GR")]["scale_right"]["value"]) == (0.0, 150.0)
