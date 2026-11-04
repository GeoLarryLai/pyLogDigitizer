import importlib.util, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_stair_reference_flagged_and_natural_not():
    rq = _load("ref_quality")
    rng = np.random.default_rng(0)
    natural = 60 + np.cumsum(rng.normal(0, 2, 2000))
    stairs = np.repeat(rng.uniform(20, 120, 100), 20)          # 10-ft constant steps at 0.5 ft
    ramps = np.concatenate([np.linspace(a, b, 12) for a, b in zip(stairs[::20][:-1], stairs[::20][1:])])
    assert not rq.quality(natural, (0, 150))["suspect"]
    assert rq.quality(stairs, (0, 150))["suspect"]
    assert rq.quality(ramps, (0, 150))["suspect"]


def test_header_scale_confidence_never_silent():
    hs = _load("header_scale_check")
    f = lambda v, r=False: {"value": v, "needs_review": r}
    good = dict(track="track1", mnemonic=f("GR"), scale_left=f(0.0), scale_right=f(160.0), scale_type=f("linear"), backup=f([160.0, 320.0]))
    assert hs.confidence(good, True) == ("high", "")
    bad = dict(good, backup=f([100.0, 200.0]))
    assert hs.confidence(bad, True)[0] == "low"
    assert hs.confidence(dict(good, scale_left=f(0.0, True)), True)[0] == "low"
    assert hs.confidence(None, False)[0] == "low"
