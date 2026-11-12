"""Black-ink curves told apart by line style (synthetic tracks from synth_scan; no example data)."""
import numpy as np
import pytest

from pylogdigitizer import curve_tracing as ct, line_styles as ls
from pylogdigitizer.ml_training_data.synth_scan import render_style_track

H, W = 3000, 700
R = np.arange(H)
XS = [150 + 90 * np.sin(R / 170.0), 330 + 100 * np.sin(R / 230.0 + 1), 500 + 60 * np.sin(R / 140.0 + 2),
      620 + 40 * np.sin(R / 300.0)]


def _suggest(styles, widths):
    trk = render_style_track(XS[:len(styles)], styles, widths, W)
    return trk, ls.suggest_line_styles(trk, {"track1": (0, W)}, None, margin=6)


def test_three_styles_proposed_and_traced_separately():
    trk, S = _suggest(["solid", "dashed", "dotted"], [6, 5, 3])
    got = S["styles"]["track1"]
    assert [s["pattern"] for s in got] == ["solid", "dashed", "dotted"]
    assert {s["name"] for s in got} == {"solid thick", "dashed thick", "dotted thin"}
    assert S["names"][:2] == ["white", "black"] and all(s["name"] in S["names"] for s in got)
    base = ls.make_style_base(trk, S)
    assert not (base & S["curve_ink"]).any()                  # no curve ink left in the base
    for s, truth in zip(got, XS):
        ink = ls.make_style_ink(base, S, s["name"], (0, W))
        T = ct.trace_track(ink[:, 6:W - 6], 0, px_per_ft=10, **ls.LINE_TRACE_PRESET)
        x = np.interp(R, T["rows"], T["x"]) + 6
        others = [o for o in XS[:3] if o is not truth]
        err = np.median(np.abs(x - truth))
        assert err < 8, (s["name"], err)                       # follows its own curve ...
        assert all(np.median(np.abs(x - o)) > 60 for o in others)   # ... not another one


def test_dash_dot_and_two_solid_weights():
    _, S = _suggest(["solid", "dashed", "dotted", "dash-dot"], [6, 5, 3, 3])
    assert {s["pattern"] for s in S["styles"]["track1"]} == {"solid", "dashed", "dotted", "dash-dot"}
    _, S = _suggest(["solid", "solid"], [7, 2])
    assert sorted(s["name"] for s in S["styles"]["track1"]) == ["solid thick", "solid thin"]
    _, S = _suggest(["solid"], [5])
    assert [s["name"] for s in S["styles"]["track1"]] == ["solid"]     # one weight: no weight word


def test_fill_and_specks_are_not_curves():
    trk = render_style_track(XS[:1], ["solid"], [6], W)
    yy, xx = np.mgrid[400:900:10, 350:600:10]                  # stippled shading (2-D lattice of dots)
    for y, x in zip(yy.ravel(), xx.ravel()):
        trk[y:y + 4, x:x + 4] = True
    rng = np.random.default_rng(0)
    for y, x in zip(rng.integers(0, H, 60), rng.integers(300, 690, 60)):   # isolated specks
        trk[y:y + 3, x:x + 3] = True
    S = ls.suggest_line_styles(trk, {"track1": (0, W)}, None, margin=6)
    assert [s["pattern"] for s in S["styles"]["track1"]] == ["solid"]


def test_legend_match_and_names():
    styles = [dict(pattern="solid", width_px=6.0), dict(pattern="dashed", width_px=5.0), dict(pattern="dotted", width_px=2.5)]
    legend = [dict(dashed=True, ink=0.6, dash_period=30.0, width=5), dict(dashed=False, width=6),
              dict(dashed=True, ink=0.4, dash_period=10.0, width=2), dict(dashed=True, ink=0.5, dash_period=40.0, width=4)]
    assert ls.match_legend(styles, legend) == [1, 0, 2, None]
    assert ls.style_name("dotted", "thin") == "dotted thin" and ls.style_name("solid") == "solid"
    with pytest.raises(ValueError):
        ls.style_name("wavy")


def test_dotted_line_hint():
    from pylogdigitizer import appearance_trace as at, workflows as wf
    assert wf.check_line_hint("dotted") == "dotted"
    seeded = [dict(gap_frac=0.05, width=np.full(10, 6.0), x=np.zeros(10)),
              dict(gap_frac=0.45, width=np.full(10, 5.0), x=np.zeros(10)),
              dict(gap_frac=0.55, width=np.full(10, 2.0), x=np.zeros(10))]
    assert at.role_by_style(seeded, "dotted") == 2 and at.role_by_style(seeded, "solid") == 0


def _touching_pair():
    """Solid curve with a dashed curve 6-20 px to its right and stippled shading between them
    (where the curves come close, the dots touch both and merge them into one piece, as on
    USA 'A' 16), plus a second dashed curve far right that never crosses."""
    sol = 200 + 90 * np.sin(R / 170.0)
    off = 6 + 14 * (0.5 + 0.5 * np.sin(R / 110.0))
    dsh = sol + off
    far = 560 + 30 * np.sin(R / 260.0)
    trk = render_style_track([sol, dsh, far], ["solid", "dashed", "dashed"], [6, 5, 5], W)
    for y in range(0, H, 8):
        for x in np.arange(sol[y] + 4, dsh[y] - 2, 8):
            trk[y:y + 5, int(x):int(x) + 5] = True
    return trk, sol, dsh, far, off


def test_touching_curves_split_along_solid_trace():
    trk, sol, dsh, far, off = _touching_pair()
    S = ls.suggest_line_styles(trk, {"track1": (0, W)}, None, margin=6)
    names = [s["name"] for s in S["styles"]["track1"]]
    solid = next(n for n in names if n.startswith("solid")); dashed = next(n for n in names if n.startswith("dashed"))
    i_d = S["names"].index(dashed)
    near = trk & (np.abs(np.arange(W)[None, :] - dsh[:, None]) <= 3) & (off[:, None] > 12)
    before = ((S["labels"] == i_d) & near).sum() / near.sum()
    base = ls.make_style_base(trk, S)
    T = ct.trace_track(ls.make_style_ink(base, S, solid, (0, W))[:, 6:W - 6], 0, px_per_ft=10, **ls.LINE_TRACE_PRESET)
    assert np.median(np.abs(np.interp(R, T["rows"], T["x"]) + 6 - sol)) < 8
    S2 = ls.reassign_off_trace(S, solid, dashed, (0, W), T["rows"], T["x"] + 6, margin=6)
    after = ((S2["labels"] == i_d) & near).sum() / near.sum()
    assert before < 0.6 and after > before + 0.2, (before, after)   # merged dashes back in their class
    assert S2["labels"] is not S["labels"]
    # two dashed curves that never cross: "left" is the one beside the solid, "right" the far one
    err = {}
    for tag, sug in (("before", S), ("after", S2)):
        ink = ls.make_style_ink(base, sug, dashed, (0, W))[:, 6:W - 6]
        for pick, truth in (("left", dsh), ("right", far)):
            Tm = ct.trace_multi(ink, 0, n_curves=2, pick=pick, px_per_ft=10, **ls.LINE_TRACE_PRESET)
            err[tag, pick] = np.median(np.abs(np.interp(R, Tm["rows"], Tm["x"]) + 6 - truth))
    # the left curve is the one the split is for (NPHI beside DPHI on USA 'A' 16): much better after
    assert err["after", "left"] < 10 and err["after", "left"] < err["before", "left"] / 3, err
    # the far curve is untouched by the split; with the near curve broken at a bend (its dashes
    # merged into the shading there), the sorted pair can hop, so it is checked before the split
    assert err["before", "right"] < 6, err


def test_route_hidden_along_trace_flags_only_long_empty_stretches():
    """A dashed curve hidden under a solid one has no ink of its own there: those stretches (and only
    those) are flagged as hidden, and the solid trace is drawn there only to keep the trace connected."""
    H, W = 400, 300
    names = ["bg", "grid", "solid thick", "dashed thick"]
    labels = np.zeros((H, W), np.int8)
    rows = np.arange(H, dtype=float)
    xs = 100 + 20 * np.sin(rows / 40)
    for r in range(0, 150, 12):           # dashes beside the solid curve, rows 0-150
        labels[r:r + 6, int(xs[r]) + 25:int(xs[r]) + 29] = 3
    for r in range(260, H, 12):           # and again from row 260
        labels[r:r + 6, int(xs[r]) + 25:int(xs[r]) + 29] = 3
    labels[220:226, 250:254] = 3          # a stray dash far away does not count
    sug = dict(labels=labels, names=names)
    ink = np.zeros((H, W), bool)
    out, hidden, iv = ls.route_hidden_along_trace(ink, sug, "dashed thick", rows, xs)
    assert not ink.any()                  # input untouched
    assert hidden[175:235].all() and not hidden[:140].any() and not hidden[275:].any()
    r = 200
    assert out[r, int(round(xs[r]))] and not out[50, int(round(xs[50]))]
    assert len(iv) == 1 and iv[0][0] <= 175 and iv[0][1] >= 235 and iv[0][0] >= 140 and iv[0][1] <= 275
    with pytest.raises(ValueError):
        ls.route_hidden_along_trace(ink, sug, "dotted thin", rows, xs)
