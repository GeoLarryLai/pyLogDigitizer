"""Joint identity re-trace on curve ink (appearance_trace.apply_identity) and stroke_band."""
import numpy as np

from pylogdigitizer import appearance_trace as at


def _draw(H, W, xs, w):
    import cv2
    im = np.zeros((H, W), np.uint8)
    pts = np.c_[xs, np.arange(H)].astype(np.int32)
    cv2.polylines(im, [pts], False, 1, int(w))
    return im.astype(bool)


def _T(x, fy, W, pad=2):
    n = len(x)
    return dict(x=np.asarray(x, float), off_left=np.zeros(n, bool), off_right=np.zeros(n, bool), fy=fy, fx=1,
                pad=pad, path_padded=np.asarray(x, int) + pad, cost=np.ones((n, W + 2 * pad)), conf=np.ones(n))


def test_stroke_band_covers_slanted_stroke():
    b = at.stroke_band(np.array([10.0, 20.0, 30.0]), np.array([4.0, 4.0, 4.0]), 3, (9, 50))
    assert b[0, 10:21].all() and b[4, 20:31].all() and not b[:, 40:].any()


def test_identity_keeps_two_crossing_curves_apart():
    H, W, fy = 600, 200, 3
    y = np.arange(H)
    xa = 100 + 60 * np.sin(y / 90)            # thick curve
    xb = 100 - 60 * np.sin(y / 90) + 5        # thin curve, crosses A twice
    ink = _draw(H, W, xa, 6) | _draw(H, W, xb, 2)
    rows = np.arange(0, H, fy) + fy // 2
    # crude priors: B's prior is pulled half-way towards A (a raster tracer that drifted)
    pa = xa[rows] + 3; pb = 0.5 * xb[rows] + 0.5 * xa[rows]
    A, B = _T(pa, fy, W), _T(pb, fy, W)
    A["candidates"] = [A, B]
    at.apply_identity(A, ink)
    far = np.abs(xa[rows] - xb[rows]) > 15      # away from the crossings
    assert np.median(np.abs(A["x"] - xa[rows])[far]) <= 3.5   # inside the 6-px stroke
    assert np.median(np.abs(B["x"] - xb[rows])[far]) <= 2
    assert np.mean(np.abs(B["x"] - xa[rows])[far] < 4) < 0.05     # B never rides A's stroke


def test_identity_keeps_wrap_samples():
    H, W, fy = 300, 120, 3
    ink = _draw(H, W, np.full(H, 60.0), 4)
    n = H // fy
    T = _T(np.full(n, 58.0), fy, W)
    T["off_right"][40:50] = True; T["x"][40:50] = W - 1
    at.apply_identity(T, ink)
    assert np.all(T["x"][40:50] == W - 1) and np.median(np.abs(T["x"][:40] - 60)) <= 2.5   # on the 4-px stroke
