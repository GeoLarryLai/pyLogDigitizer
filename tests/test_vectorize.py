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


def _dashed_scene(h=800, w=240):
    ink = np.zeros((h, w), bool)
    for x in range(20, w, 40):
        ink[:, x] = True
    for y in range(50, h, 100):
        ink[y, :] = True
    rows = np.arange(h)
    xs = (120 + 50 * np.sin(rows / 110.0)).astype(int)      # thick solid curve
    xd = (120 + 50 * np.cos(rows / 140.0)).astype(int)      # thin dashed curve (10 on / 6 off)
    for y in rows:
        ink[y, xs[y] - 2:xs[y] + 3] = True
        if y % 16 < 10:
            ink[y, xd[y]:xd[y] + 2] = True
    return ink, xs, xd


def test_links_make_long_chains_and_dashed_style():
    ink, xs, xd = _dashed_scene()
    R = vz.vector_separate_v2(ink, n_curves=2)
    ch = sorted(R["chains"], key=lambda c: -c["span"])
    assert ch[0]["span"] > 150                       # linked across grid crossings (grid rows every 100)
    for c in ch[:6]:                                 # chains do not mix the two curves
        yy = np.clip(c["y"], 0, ink.shape[0] - 1)
        assert max(np.mean(np.abs(c["x"] - xs[yy]) <= 3), np.mean(np.abs(c["x"] - xd[yy]) <= 3)) > 0.9
    rows = np.arange(ink.shape[0])

    def owner(xc):
        hit = {}
        for c in R["chains"]:
            m = np.abs(c["x"] - xc[np.clip(c["y"], 0, len(xc) - 1)]) <= 2
            if m.sum() > 50:
                hit[c["id"]] = c
        return list(hit.values())
    dashed = owner(xd); solid = owner(xs)
    assert any(c["dashed"] for c in dashed)
    assert not any(c["dashed"] for c in solid if c["span"] > 300)


def test_rescue_steep_stretch_on_grid_column():
    h, w = 400, 120
    ink = np.zeros((h, w), bool)
    for x in (20, 60, 100):
        ink[:, x] = True
    for y in range(0, h):        # curve: slant in, run 120 rows along column x=60 (offset 1 px), slant out
        if y < 140:
            x = 30 + int((y / 140) * 31)
        elif y < 260:
            x = 61
        else:
            x = 61 + int(((y - 260) / 140) * 30)
        ink[y, x:x + 3] = True
    R = vz.vector_separate_v2(ink, n_curves=1)
    assert R["curve_ink"][150:250, 60:64].any(axis=1).mean() > 0.9


def test_long_curve_not_text():
    from pylogdigitizer import vectorize as v
    ink, xs, xd = _dashed_scene()
    R = v.vector_separate_v2(ink, n_curves=2)
    rows = np.arange(ink.shape[0])
    on = np.zeros(len(rows), bool)
    for c in R["chains"]:
        yy = np.clip(c["y"], 0, len(rows) - 1)
        on[yy[np.abs(c["x"] - xd[yy]) <= 2]] = True
    assert on.mean() > 0.8                           # dashed curve kept as curve chains, not text


def test_chain_path_keeps_spike_tip():
    ink = np.zeros((300, 200), bool)
    for y in range(300):                               # wiggly baseline curve (not a straight line)
        x0 = int(50 + 10 * np.sin(y / 20.0)); ink[y, x0:x0 + 3] = True
    ink[150, 50:180] = True                            # one-row spike to x=180
    ink[149:152, 175:180] = True
    R = vz.vector_separate_v2(ink, n_curves=1)
    x, cov, _ = vz.chain_path(R["chains"], R["groups"], ink.shape, 1)
    assert cov.mean() > 0.95 and np.nanmax(x) > 150


def test_broken_grid_row_stays_grid():
    ink = np.zeros((300, 300), bool)
    for y in range(300):
        x0 = int(150 + 10 * np.sin(y / 20.0)); ink[y, x0:x0 + 3] = True
    for x in (30, 90, 210, 270):
        ink[:, x] = True
    ink[100, :] = True
    ink[100, 60:66] = False                          # small gaps in a grid row
    ink[100, 236:242] = False
    R = vz.vector_separate_v2(ink, n_curves=1)
    row = R["curve_ink"][100]
    assert row[:120].sum() <= 3 and row[190:].sum() <= 3
