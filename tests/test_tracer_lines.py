"""Thick depth-line bands, centreline cost, stroke-style identity and legend line samples."""
import numpy as np

from pylogdigitizer import curve_tracing as ct
from pylogdigitizer import separation as sep
from pylogdigitizer import header_ocr as ho


def _panel(h=900, w=300):
    p = np.zeros((h, w), bool)
    for r in range(0, h, 30):          # thin grid rows
        p[r:r + 2] = True
    for r in (300, 600):               # thick depth lines (every 10 grid rows)
        p[r:r + 9] = True
    return p


def test_bands_and_anchors():
    p = _panel()
    bands = sep.hline_bands(p)
    assert any(e - a >= 8 for a, e in bands)
    anc = sep.thick_band_anchors(bands)
    assert len(anc) == 2 and abs(anc[0] - 304) <= 2 and abs(anc[1] - 604) <= 2


def test_trace_bridges_thick_band_without_lateral_following():
    p = _panel()
    y = np.arange(p.shape[0])
    x = (150 + 60 * np.sin(y / 90)).astype(int)
    for yy, xx in zip(y, x):
        p[yy, xx - 1:xx + 2] = True
    T = ct.trace_track(p, fy=1, fx=1)
    err = np.abs(T["x"] - x)
    assert np.median(err) < 3
    for a in (300, 600):                # no excursion along the band
        assert err[a:a + 9].max() < 12


def test_run_offcentre():
    ink = np.zeros((1, 15), bool); ink[0, 3:12] = True
    off = ct.run_offcentre(ink)[0]
    assert off[7] == 0 and off[3] == 1 and off[11] == 1 and off[0] == 1


def test_center_weight_moves_trace_to_stroke_centre():
    p = np.zeros((600, 300), bool)
    y = np.arange(600)
    c = (150 + 50 * np.sin(y / 70)).astype(int)
    for yy, cc in zip(y, c):
        p[yy, cc - 10:cc + 11] = True   # 21-px wide wavy stroke
    T0 = ct.trace_track(p, fy=1, fx=1, center_weight=0.0)
    T1 = ct.trace_track(p, fy=1, fx=1, center_weight=0.3)
    e0, e1 = np.abs(T0["x"] - c), np.abs(T1["x"] - c)
    assert np.median(e1) <= 3
    assert np.mean(e1) <= np.mean(e0)


def test_restyle_segments_swaps_exchanged_segment():
    n = 300
    A = dict(path=np.full(n, 50.0)); B = dict(path=np.full(n, 150.0))
    A["path"][200:] = 150; B["path"][200:] = 50       # DP exchanged them at row 200
    A["path"][195:205] = B["path"][195:205] = 100       # contact
    wa = np.where(np.arange(n) < 200, 6.0, 2.0); wb = np.where(np.arange(n) < 200, 2.0, 6.0)
    sw = ct.restyle_segments(A, B, wa, wb, near=4)
    assert sw[210:].all() and not sw[:190].any()


def test_path_style_thick_vs_dashed():
    p = np.zeros((600, 200), bool)
    p[:, 48:55] = True                                  # thick solid at x=51
    for r in range(0, 600, 20):
        p[r:r + 10, 150:152] = True                     # thin dashed at x=150
    rows = np.arange(600) + 0.5
    f = lambda x: dict(rows=rows, x=np.full(600, x), fy=1, row0=0,
                       off_left=np.zeros(600, bool), off_right=np.zeros(600, bool))
    s1, s2 = ct.path_style(p, f(51.0)), ct.path_style(p, f(150.0))
    assert s1["width"] >= 6 and s1["ink"] > 0.95 and s1["dash_period"] == 0
    assert s2["width"] <= 3 and 0.3 < s2["ink"] < 0.7 and abs(s2["dash_period"] - 20) <= 2
    assert ct.style_distance(s1, s2) > ct.style_distance(s1, s1) + 0.5


def test_legend_styles_finds_solid_and_dashed_samples():
    ink = np.zeros((1500, 400), bool)
    ink[:, 0:2] = ink[:, 398:400] = True                # track borders
    for r in range(800, 1500, 25):
        ink[r:r + 2] = True                             # grid starts at 800
    for c in range(40, 400, 40):
        ink[800:, c:c + 2] = True                       # vertical grid lines
    ink[600:605, 100:250] = True                        # heavy solid legend sample
    for c in range(100, 250, 12):
        ink[700:702, c:c + 6] = True                    # thin dashed legend sample
    L = ho.legend_styles(ink, 0, 400)
    assert len(L) == 2
    assert L[0]["width"] >= 4 and not L[0]["dashed"]
    assert L[1]["width"] <= 2 and L[1]["dashed"]


def test_scale_join_needs_backup_span_1x_or_2x():
    p = dict(left=10.0, right=110.0); q = dict(left=110.0, right=310.0)
    assert ho._join(p, q) == ((10.0, 110.0), (110.0, 310.0), False)
    assert ho._join(dict(left=110.0, right=10.0), dict(left=10.0, right=3101.0)) is None
    assert ho._join(dict(left=0.0, right=16.0), dict(left=160.0, right=321.0))[0] == (0.0, 160.0)
