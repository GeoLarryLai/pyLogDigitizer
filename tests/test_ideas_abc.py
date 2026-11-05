import pathlib
import numpy as np
import pytest

from pylogdigitizer import grid_layer as gl, appearance_trace as at
from pylogdigitizer.ml_training_data import synth_scan as ss


def _grid_panel(H=900, W=300, curve=True):
    ink = np.zeros((H, W), bool)
    for x in range(0, W, 30):
        ink[:, x:x + 2] = True
    ink[:, 150:155] = True                       # heavy line
    for y in range(10, H, 30):
        ink[y:y + 2] = True
    xc = None
    if curve:
        yy = np.arange(H); xc = (100 + 60 * np.sin(yy / 90)).astype(int)
        for r in range(H):
            ink[r, xc[r] - 2:xc[r] + 2] = True
    return ink, xc


def test_run_lengths():
    a = np.array([[1, 1, 0, 1], [0, 1, 1, 1]], bool)
    assert gl.run_lengths(a, 1).tolist() == [[2, 2, 0, 1], [0, 3, 3, 3]]
    assert gl.run_lengths(a, 0).tolist() == [[1, 2, 0, 2], [0, 2, 1, 2]]


def test_fit_grid_lattice_and_heavy_line():
    ink, _ = _grid_panel(curve=False)
    G = gl.fit_grid(ink, block=300)
    cs = sorted(v["center"] for v in G["vlines"])
    assert len(cs) == 10 and G["v_kind"] in ("linear", "irregular")
    assert abs(G["h_period"] - 30) < 1
    heavy = [G["vlines"][i]["center"] for i in G["heavy_v"]]
    assert heavy and all(abs(h - 152) < 4 for h in heavy)


def test_residual_keeps_curve_drops_grid():
    ink, xc = _grid_panel()
    R = gl.grid_residual(ink, block=300)
    res = R["residual"]
    rows = np.arange(len(xc))
    v, _ = _grid_panel(curve=False); v[:, 98:102] = True             # vertical curve crossing every grid row
    assert gl.grid_residual(v, block=300)["residual"][:, 99].all()
    # known gap: slanted strokes crossing a grid row / riding a grid column lose some pixels
    assert res[rows, xc].mean() > 0.75
    far = np.ones_like(ink); far[np.abs(np.arange(ink.shape[1])[None, :] - xc[:, None]) < 4] = False
    assert (res & far).sum() < 0.01 * (ink & far).sum()   # grid explained away


def test_appearance_tracer_follows_prior_curve_not_thicker_line():
    H, W = 900, 160
    ink = np.zeros((H, W), bool); ink[:, 18:28] = True            # thick, continuous neighbour
    yy = np.arange(H); xc = (90 + 40 * np.sin(yy / 60)).astype(int)
    for r in range(H):
        if (r // 15) % 4 != 3:                                       # thin and broken (dashed)
            ink[r, xc[r] - 1:xc[r] + 2] = True
    prior = xc[1::3].astype(float) + 15                              # coarse, offset raster prior
    r = at.trace_appearance(ink, 3, prior)
    assert np.median(np.abs(r["x"] - xc[1::3])) <= 2
    assert np.mean(np.abs(r["x"] - xc[1::3]) < 6) > 0.9
    assert r["gap_frac"].max() > 0                                    # dash gaps were learnt


@pytest.mark.xfail(strict=True, reason="known gap: nothing rewards the out-and-back of a flat spike yet; "
                   "the tracer stays on the baseline (seen on WEATHERFORD ~3172/3197 ft too)")
def test_appearance_tracer_takes_flat_spike():
    H, W = 600, 300
    ink = np.zeros((H, W), bool); ink[:, 60:62] = True
    ink[300:302, 60:240] = True; ink[302:304, 60:240] = True        # near-flat spike out and back
    prior = np.full(H // 3, 61.0)
    r = at.trace_appearance(ink, 3, prior, p_wt=0.0)
    assert r["x"].max() >= 230 and r["sideways"].any()


def test_sweep_matches_bruteforce():
    rng = np.random.default_rng(1)
    M = rng.random(40); runs = [(3, 15), (20, 21), (25, 40)]
    for right in (True, False):
        v, a = at._sweep(M, runs, 0.05, right)
        for s, e in runs:
            for x in range(s, e):
                js = range(s, x) if right else range(x + 1, e)
                best = min((M[j] + 0.05 * abs(x - j) for j in js), default=np.inf)
                assert np.isclose(v[x], best) or (np.isinf(best) and np.isinf(v[x]))


def test_zigzag_score():
    assert ss.zigzag_score(np.tile([0, 1], 50)) > 0.9
    assert ss.zigzag_score(np.sin(np.arange(200) / 20)) < 0.1


def test_render_tile_labels():
    curves = [dict(name="a:GR", depth=np.arange(0, 400, 0.5), value=50 + 30 * np.sin(np.arange(800) / 30), zigzag=0.0)]
    r = ss.render_tile(np.random.default_rng(3), curves, H=256, W=256, binary=True)
    img, lab = r["image"], r["label"]
    assert img.shape == lab.shape == (256, 256)
    assert set(np.unique(img)) <= {0, 255}
    assert (lab == ss.GRID).any() and (lab == ss.CURVE_A).any()
    cur = lab == ss.CURVE_A
    assert np.isfinite(r["direction"][cur]).mean() > 0.95 and np.isnan(r["direction"][lab == ss.PAPER]).all()
    assert (img[cur] == 0).mean() > 0.6         # labelled curve pixels are mostly inked


def test_load_curves_drops_zigzag(tmp_path):
    lasio = pytest.importorskip("lasio")
    las = lasio.LASFile()
    d = np.arange(1000, 1500, 0.5)
    las.append_curve("DEPT", d); las.append_curve("GR", 60 + 20 * np.sin(d / 7)); las.append_curve("ZZ", np.tile([0.0, 5.0], len(d) // 2))
    p = tmp_path / "x.las"; las.write(str(p))
    C = ss.load_curves([p])
    names = [c["name"].split(":")[-1] for c in C]
    assert "GR" in names and "ZZ" not in names
