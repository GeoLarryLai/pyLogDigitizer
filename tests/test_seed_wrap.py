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
