"""Independent seeds for multi-curve tracks and the explicit backup-wrap transition."""
import numpy as np

from pylogdigitizer import appearance_trace as at


def _draw(H, W, xs, w, rows=None):
    import cv2
    im = np.zeros((H, W), np.uint8)
    y = np.arange(H) if rows is None else rows
    pts = np.c_[xs, y].astype(np.int32)
    cv2.polylines(im, [pts], False, 1, int(w))
    return im.astype(bool)


def test_find_seeds_two_curves_widest_first():
    H, W = 300, 200
    ink = _draw(H, W, np.full(H, 50.0), 6) | _draw(H, W, np.full(H, 150.0), 2)
    s, seeds = at.find_seeds(ink, 3, 2)
    assert s == 0 and abs(seeds[0][0] - 50) <= 2 and abs(seeds[1][0] - 150) <= 2 and seeds[0][1] > seeds[1][1]
    assert at.find_seeds(ink, 3, 3) is None


def test_seeded_multi_trace_follows_both_crossing_curves():
    H, W, fy = 900, 220, 3
    y = np.arange(H)
    xa = 60 + 50 * np.sin(y / 120) + y * 0.08        # thick, crosses the thin one
    xb = 160 - 50 * np.sin(y / 120) - y * 0.08
    ink = _draw(H, W, xa, 6) | _draw(H, W, xb, 2)
    T = at.seeded_multi_trace(ink, fy, 2)
    assert len(T) == 2
    rows = np.arange(0, H, fy) + 1
    far = np.abs(xa[rows] - xb[rows]) > 15
    ea = np.abs(T[0]["x"] - xa[rows])[far]; eb = np.abs(T[1]["x"] - xb[rows])[far]
    assert np.median(ea) <= 3.5 and np.median(eb) <= 2
    assert np.mean(eb < 4) > 0.9 and np.mean(ea < 5) > 0.9          # no identity swap after crossings


def test_match_to_orders_by_overlap():
    c = [dict(x=np.full(10, 10.0)), dict(x=np.full(10, 90.0))]
    s = [dict(x=np.full(10, 88.0)), dict(x=np.full(10, 12.0))]
    m = at.match_to(c, s)
    assert m[0]["x"][0] == 12.0 and m[1]["x"][0] == 88.0


def test_wrap_backup_follows_backup_curve_from_opposite_edge():
    H, W, fy = 600, 200, 3
    y = np.arange(H).astype(float)
    x = 120 + 0 * y
    # spike: main curve runs to the right edge for rows 240-360, backup re-enters from the left
    up = (y >= 200) & (y < 240); x[up] = 120 + (y[up] - 200) * 2.1
    x[(y >= 240) & (y < 360)] = 205
    dn = (y >= 360) & (y < 400); x[dn] = 204 - (y[dn] - 360) * 2.1
    ink = _draw(H, W, np.clip(x, 0, 199), 4)
    rb = np.arange(250, 350); xb_true = 10 + 30 * np.sin((rb - 250) / 100 * np.pi)
    ink |= _draw(H, W, xb_true, 3, rows=rb)
    ink[250:350, 0:4] = True                                     # backup enters at the left edge
    n = H // fy
    tx = np.clip(x[np.arange(n) * fy + 1], 0, 196)               # tracer flattened at the edge
    wb = at.wrap_backup(ink, tx, fy, W, "right")
    assert len(wb) == 1
    a, b, xb = wb[0]
    assert a * fy <= 260 and b * fy >= 340
    mid = (np.arange(a, b) * fy + 1 >= 270) & (np.arange(a, b) * fy + 1 < 330)
    truth = np.interp(np.arange(a, b)[mid] * fy + 1, rb, xb_true)
    assert np.median(np.abs(xb[mid] - truth)) <= 4
    assert at.wrap_backup(_draw(H, W, np.full(H, 100.0), 4), np.full(n, 100.0), fy, W) == []


def test_find_seeds_rejects_border_ink():
    H, W = 300, 400
    ink = _draw(H, W, np.full(H, 3.0), 4) | _draw(H, W, np.full(H, 120.0), 3) | _draw(H, W, np.full(H, 260.0), 5)
    s, seeds = at.find_seeds(ink, 3, 2)
    xs = sorted(round(x) for x, _ in seeds)
    assert all(abs(a - b) <= 2 for a, b in zip(xs, [120, 260]))


def test_role_by_style():
    seeded = [dict(gap_frac=0.05, width=np.full(5, 6.0)), dict(gap_frac=0.4, width=np.full(5, 2.0))]
    assert at.role_by_style(seeded, "dashed") == 1 and at.role_by_style(seeded, "solid") == 0
    assert at.role_by_style(seeded, "thick") == 0 and at.role_by_style(seeded, "thin") == 1


def test_seeded_trace_does_not_ride_the_other_curve_after_a_gap():
    # thin curve B disappears for a stretch next to thick curve A, then reappears farther away:
    # B must not collapse onto A for the whole gap and must be found again afterwards
    H, W, fy = 1200, 260, 3
    y = np.arange(H)
    xa = np.full(H, 80.0); xb = 140 + 20 * np.sin(y / 60)
    ink = _draw(H, W, xa, 6)
    rows_b = np.r_[np.arange(0, 400), np.arange(700, H)]
    ink |= _draw(H, W, xb[:400], 2, rows=np.arange(400)) | _draw(H, W, xb[700:] + 60, 2, rows=np.arange(700, H))
    T = at.seeded_multi_trace(ink, fy, 2)
    rows = np.arange(0, H, fy) + 1
    late = rows >= 760
    tb = np.interp(rows[late], np.arange(700, H), xb[700:] + 60)
    assert np.median(np.abs(T[1]["x"][late] - tb)) <= 3
    gap = (rows > 450) & (rows < 650)
    assert np.mean(np.abs(T[1]["x"][gap] - 80) < 6) < 0.8


def test_clip_to_null_edges_but_not_wraps():
    from pylogdigitizer.workflows import clip_to_null
    x = np.r_[np.full(10, 50.0), np.full(10, 99.0), np.full(10, 50.0), np.full(10, 0.0), np.full(10, 50.0), [99.0, 99.0]]
    v = x.copy()
    out = clip_to_null(v, x, 100, wraps=[(30, 40)])
    assert np.isnan(out[10:20]).all() and np.isfinite(out[:10]).all()
    assert np.isfinite(out[30:40]).all()                  # wrap span kept (backup carries it)
    assert np.isfinite(out[-2:]).all()                    # short touch kept


def test_las_resample_linear_keeps_nulls():
    from pylogdigitizer.las_export import resample
    d = np.arange(100.0, 110.0, 1 / 3); v = 2 * d; v[9:15] = np.nan
    g, o = resample(d, v, 0.25)
    ok = np.isfinite(o)
    assert np.allclose(o[ok], 2 * g[ok])              # pure linear, no smoothing
    assert np.isnan(o[(g > 103.1) & (g < 104.9)]).all()
    g5, _ = resample(d, v, 0.5); assert np.allclose(np.diff(g5), 0.5)
    import pytest
    with pytest.raises(ValueError):
        resample(d, v, 0.1)


def test_class_inks_trace_each_curve_on_its_own_class():
    # two curves cross repeatedly; the identity segmenter's class masks keep them apart even where
    # a plain seeded trace would have no appearance cue (same width here)
    H, W, fy = 900, 220, 3
    y = np.arange(H)
    xa = 110 + 70 * np.sin(y / 90); xb = 110 - 70 * np.sin(y / 90)
    ia = _draw(H, W, xa, 4); ib = _draw(H, W, xb, 4) & ~ia
    T = at.seeded_multi_trace(ia | ib, fy, 2, class_inks=[ib, ia])
    rows = np.arange(0, H, fy) + 1
    far = np.abs(xa[rows] - xb[rows]) > 15
    assert np.median(np.abs(T[0]["x"] - xb[rows])[far]) <= 3      # first = the requested class (B)
    assert np.mean(np.abs(T[1]["x"] - xa[rows])[far] < 7) > 0.9   # right curve (a swap would be ~100 px off)


def test_role_by_position_hint():
    seeded = [dict(x=np.full(5, 300.0)), dict(x=np.full(5, 40.0))]
    assert at.role_by_style(seeded, "left") == 1 and at.role_by_style(seeded, "right") == 0


def test_line_hint_rejects_unknown():
    import pytest
    from pylogdigitizer import workflows as wf
    with pytest.raises(ValueError, match="line_hint"):
        wf.check_line_hint("NPHI")
    assert wf.check_line_hint("thin") == "thin"
