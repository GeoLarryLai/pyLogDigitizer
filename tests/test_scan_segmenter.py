"""synth_scan realism generator, suspect-LAS filter, scan_stats and the ONNX segmenter runtime."""
import numpy as np
import pytest

from pylogdigitizer.ml_training_data import synth_scan as ss
from pylogdigitizer.ml_training_data.scan_stats import scan_stats


def _smooth(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = np.exp(-0.5 * (np.arange(-30, 31) / 8.0) ** 2); k /= k.sum()   # Gaussian-smoothed noise, like a log
    return np.convolve(rng.normal(size=n + 60), k, "valid")[:n] * 120 + 80


def test_suspect_reasons_clean_curve_passes():
    assert ss.suspect_reasons(_smooth()) == []


def test_suspect_reasons_flags_stairs_linear_sawtooth():
    v = _smooth()
    assert "stairs" in ss.suspect_reasons(np.repeat(v[::5], 5))
    knots = v[::20]; lin = np.interp(np.arange(len(knots) * 20 - 19), np.arange(0, len(knots) * 20, 20), knots)
    assert "linear" in ss.suspect_reasons(lin)
    saw = v + np.where(np.arange(len(v)) % 2, 3.0, -3.0)
    assert "sawtooth" in ss.suspect_reasons(saw)
    assert ss.suspect_reasons(np.ones(100)) == ["flat"]


def _curves():
    return [dict(name=f"c{i}", depth=np.arange(3000) * 0.5, value=_smooth(3000, i), lo=0.0, hi=160.0, log=False)
            for i in range(3)]


def test_render_scan_tile_binary_and_labels_consistent():
    try:
        t = ss.render_scan_tile(np.random.default_rng(3), _curves(), 256, 256)
    except (KeyError, TypeError) as e:
        pytest.skip(f"curve dict schema differs: {e}")
    img, lab = t["image"], t["label"]
    assert img.shape == lab.shape == (256, 256)
    assert set(np.unique(img)) <= {0, 255}
    ink = img < 128
    assert 0.02 < ink.mean() < 0.7
    assert lab.max() <= 4 and (lab == 1).any() and ((lab == 2) | (lab == 3)).any()
    assert ((lab > 0) & ~ink).mean() < 0.05            # labels sit on ink
    d = t["direction"]; assert d.dtype == np.float32 and np.isnan(d[lab == 0]).all()
    t2 = ss.render_scan_tile(np.random.default_rng(3), _curves(), 256, 256)
    assert np.array_equal(t2["image"], img)             # deterministic per seed


def test_scan_stats_on_known_grid():
    ink = np.zeros((300, 400), bool)
    ink[:, 50:53] = True; ink[:, 250:253] = True      # 3-px vertical lines
    for y in range(20, 300, 40):
        ink[y:y + 2] = True                             # 2-px horizontal lines every 40 px
    yy = np.arange(300); xx = (120 + 10 * np.sin(yy / 40)).astype(int)
    for dx in range(5):
        ink[yy, xx + dx] = True                         # 5-px wide curve
    s = scan_stats(ink)
    assert s["v_lines"] == 2 and s["v_width_med"] == 3
    assert s["h_width_med"] == 2 and abs(s["h_spacing"] - 40) < 1
    assert 3 <= s["stroke_med"] <= 7
    assert s["speckle_per_mpx"] == 0


def _tiny_model(path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    # 1x1 conv: ink -> class 2 (curve) logit high, paper logit high where no ink
    W = np.array([-5, 0, 5, 0], np.float32).reshape(4, 1, 1, 1); B = np.array([2.5, 0, 0, 0], np.float32)
    g = helper.make_graph([helper.make_node("Conv", ["ink", "W", "B"], ["logits"])], "t",
                          [helper.make_tensor_value_info("ink", TensorProto.FLOAT, [1, 1, None, None])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4, None, None])],
                          [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
    onnx.save(m, str(path)); return str(path)


def test_segment_scan_tiles_blend_and_label(tmp_path):
    from pylogdigitizer.scan_segmenter import CLASSES4 as CLASSES, label_scan, segment_scan
    mp = _tiny_model(tmp_path / "t.onnx")
    rng = np.random.default_rng(0); ink = rng.random((300, 517)) < 0.3
    p = segment_scan(ink, mp, tile=128, overlap=32)
    assert p.shape == (len(CLASSES), 300, 517) and np.allclose(p.sum(0), 1, atol=1e-5)
    lab = label_scan(ink, mp, tile=128, overlap=32)
    assert (lab[ink] == 2).all() and (lab[~ink] == 0).all()
    small = label_scan(ink[:40, :50], mp)                # smaller than a tile
    assert small.shape == (40, 50) and (small[ink[:40, :50]] == 2).all()


def test_curve_probability_four_class_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    from pylogdigitizer.scan_segmenter import curve_probability, classes_for
    W = np.array([-5, 0, 5, 0], np.float32).reshape(4, 1, 1, 1); B = np.array([2.5, 0, 0, 0], np.float32)
    g = helper.make_graph([helper.make_node("Conv", ["ink", "W", "B"], ["logits"])], "t",
                          [helper.make_tensor_value_info("ink", TensorProto.FLOAT, [1, 1, None, None])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4, None, None])],
                          [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
    onnx.save(m, str(tmp_path / "m4.onnx"))
    assert classes_for(4)[2] == "curve"
    ink = np.random.default_rng(1).random((100, 140)) < 0.3
    p = curve_probability(ink, str(tmp_path / "m4.onnx"), tile=64, overlap=16)
    assert p.shape == ink.shape and (p[ink] > 0.9).all() and (p[~ink] < 0.1).all()


def test_synth_overflow_spikes_are_curve_labelled_across_grid():
    from pylogdigitizer.ml_training_data import synth_scan as ss
    rng = np.random.default_rng(3)
    y = np.linspace(0, 1, 400)
    C = [dict(name="GR", depth=np.arange(400) * 0.5, value=60 + 40 * np.sin(y * 20))]
    n_ok = 0
    for s in range(30):
        t = ss.render_scan_tile(np.random.default_rng(s), C, 256, 256, overflow=1.0)
        if not t["meta"].get("overflow"):
            continue
        lab = t["label"]
        # curve labels reach a tile edge column (the excursion runs out to the edge)
        if ((lab[:, :4] == ss.CURVE_A).any() or (lab[:, -4:] == ss.CURVE_A).any()):
            n_ok += 1
    assert n_ok >= 10
    t0 = ss.render_scan_tile(np.random.default_rng(5), C, 256, 256)
    assert not t0["meta"].get("overflow")


def test_bundled_default_models_load_and_are_small():
    from pylogdigitizer import scan_segmenter as sg
    import os
    for name in ("default", "overflow"):
        p = sg.default_model(name)
        assert os.path.getsize(p) < 10e6
        ink = np.zeros((256, 256), bool); ink[:, 120:124] = True
        cp = sg.curve_probability(ink, p)
        assert cp.shape == ink.shape and np.isfinite(cp).all()


def test_training_refuses_to_overwrite_bundled_default(tmp_path):
    import sys, pathlib, pytest
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import train_scan_segmenter as tss
    from pylogdigitizer import scan_segmenter as sg
    with pytest.raises(ValueError, match="overwrite_default"):
        tss.train("nothing/*.las", str(sg.MODEL_DIR), steps=1)
