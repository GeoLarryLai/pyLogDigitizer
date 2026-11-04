import numpy as np
from pylogdigitizer import separation as sep
from pylogdigitizer import curve_tracing as ct


def _panel(H=900, W=300, seed=0):
    ink = np.zeros((H, W), bool)
    for x in range(0, W, 30):
        ink[:, x:x + 2] = True                   # thin vertical grid
    for y in range(0, H, 25):
        ink[y:y + 2, :] = True                   # thin horizontal grid
    for y in range(100, H, 250):
        ink[y:y + 9, :] = True                   # heavy depth lines
    ink[:, 200:206] = True                       # thick reference line
    r = np.arange(H)
    x = (120 + 60 * np.sin(r / 60)).astype(int)
    for dx in range(-2, 3):
        ink[r, np.clip(x + dx, 0, W - 1)] = True  # 5-px curve
    return ink, x


def test_remove_grid_erases_lines_keeps_curve():
    ink, x = _panel()
    out = sep.remove_grid(ink)
    r = np.arange(len(x))
    on_curve = out[r, x]
    assert on_curve.mean() > 0.99                # curve continuous, crossings bridged
    curve = np.zeros_like(ink)
    for dx in range(-3, 4):
        curve[r, np.clip(x + dx, 0, ink.shape[1] - 1)] = True
    left = out & ~curve
    assert left.sum() < 0.01 * (ink & ~curve).sum()   # grid, bands and reference line gone


def test_trace_on_grid_removed_follows_curve():
    ink, x = _panel()
    T = ct.trace_track(ink, 0, px_per_ft=10, grid_removed=True)
    err = np.abs(T["x"] - np.interp(T["rows"], np.arange(len(x)), x))
    assert np.median(err) < 3 and np.mean(err > 10) < 0.02


def test_remove_grid_erases_dashed_columns():
    ink, x = _panel()
    rng = np.random.default_rng(1)
    for x0 in (60, 240):
        ink[:, x0:x0 + 2] = False
        for y in range(0, ink.shape[0], 12):
            ink[y:y + 7, x0:x0 + 2] = True          # dashed column (7 on / 5 off)
    out = sep.remove_grid(ink)
    for x0 in (60, 240):
        r = np.arange(len(x)); far = np.abs(x - x0) > 8
        assert out[r[far], x0].mean() < 0.01


def test_remove_grid_seg_mode_with_given_prob():
    ink, x = _panel()
    r = np.arange(len(x))
    pc = np.zeros(ink.shape, np.float32); pg = np.full(ink.shape, 0.9, np.float32)
    for dx in range(-2, 3):
        pc[r, np.clip(x + dx, 0, ink.shape[1] - 1)] = 0.95   # oracle P(curve) on the curve
    ink[300:303, 10:25] = True                               # short partial-width dash
    out = sep.remove_grid(ink, prob=(pc, pg))
    assert out[r, x].mean() > 0.99
    assert not out[300:303, 10:25].any()


def test_trace_multi_pick_margin_keeps_first_pass():
    ink, x = _panel()
    r = np.arange(len(x))
    x2 = np.clip(x + 70, 0, ink.shape[1] - 1)
    ink[r, x2] = True                                   # thin companion curve
    T0 = ct.trace_multi(ink, 0, n_curves=2, px_per_ft=10, grid_removed=True)
    T1 = ct.trace_multi(ink, 0, n_curves=2, px_per_ft=10, grid_removed=True, pick_margin=1e9)
    assert T1["picked"] == 0
    sc = [t["score"] for t in T0["candidates"]]
    assert T0["picked"] == int(np.argmax(sc))
