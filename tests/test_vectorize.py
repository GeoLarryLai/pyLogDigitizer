import numpy as np
from pylogdigitizer import vectorize as vz
from pylogdigitizer import curve_tracing as ct


def _scene(h=600, w=200):
    ink = np.zeros((h, w), bool)
    for x in range(20, w, 40):            # thin vertical grid columns
        ink[:, x] = True
    for y in range(50, h, 100):           # horizontal grid rows, one thick depth line
        ink[y:y + (4 if y == 250 else 1), :] = True
    rows = np.arange(h)
    xa = (100 + 32 * np.sin(rows / 90.0)).astype(int)     # thick curve
    xb = (100 + 32 * np.cos(rows / 120.0)).astype(int)     # thin curve, crosses the thick one
    for y in rows:
        ink[y, xa[y] - 2:xa[y] + 3] = True
        ink[y, xb[y]:xb[y] + 2] = True
    return ink, xa, xb


def test_skeleton_is_thin():
    ink = np.zeros((40, 40), bool); ink[5:35, 15:22] = True
    sk = vz.skeletonize(ink)
    assert sk.any() and sk.sum(axis=1).max() <= 2


def test_grid_and_curve_classes():
    ink, xa, xb = _scene()
    R = vz.vector_separate(ink, n_curves=2)
    cls, V = R["cls"], R["V"]
    sk = V["skel"]; seg = V["seg"]
    on_col = seg[:, 60][seg[:, 60] > 0]               # grid column x=60
    assert (cls[on_col] == vz.GRID).mean() > 0.8
    cur = R["curve_ink"]
    assert not cur[:, 140].all() and cur[300, :].sum() < 20     # grid column / grid row gone
    rows = np.arange(ink.shape[0])
    assert cur[rows, xa].mean() > 0.95                  # thick curve kept, incl. under grid lines
    assert sk.any()


def test_trace_connected_on_curve_vectors():
    ink, xa, xb = _scene()
    R = vz.vector_separate(ink, n_curves=1)

    def hit(panel):
        T = ct.trace_track(panel, 0, px_per_ft=10, min_area=50, offscale_cost=0.98)
        x = np.asarray(T["x"]); r = np.asarray(T["rows"]).astype(int)
        return np.mean(np.minimum(np.abs(x - xa[r]), np.abs(x - xb[r])) <= 3)
    v, raw = hit(R["curve_ink"]), hit(ink)
    assert v > 0.95 and v >= raw         # on curve vectors the path stays on a curve, not a grid column
