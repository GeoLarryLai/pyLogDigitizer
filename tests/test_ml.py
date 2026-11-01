# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Tests for the ML cost-map modules, training pipeline, and DP tracer integration."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from pylogdigitizer import (
    curve_tracing as ct,
    ml_batch_digitize as mbd,
    ml_models,
    ml_training,
    ml_training_data as mtd,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_panel_and_ref(h_dp: int = 256, w_dp: int = 120, fy: int = 3, fx: int = 2):
    """Create a synthetic track panel with a wiggling target curve and a straight distractor line."""
    H, W = h_dp * fy, w_dp * fx
    panel = np.zeros((H, W), dtype=bool)
    ys = np.arange(H, dtype=np.float32)
    # Target wiggling curve
    x_curve_px = np.round((0.45 + 0.25 * np.sin(ys / 35.0)) * (W - 1)).astype(int)
    for r in range(H):
        c0 = max(0, x_curve_px[r] - 2)
        c1 = min(W, x_curve_px[r] + 3)
        panel[r, c0:c1] = True
    # Stationary vertical distractor (e.g. caliper line)
    c_dist = int(0.82 * (W - 1))
    panel[:, max(0, c_dist - 2) : min(W, c_dist + 3)] = True

    cost, _ = ct.build_cost(panel, min_width=2, fy=fy, fx=fx, min_area=20, straight_len=121, return_labels=True)
    x_ref_dp = x_curve_px[fy // 2 :: fy][: cost.shape[0]].astype(np.float32) / fx
    return panel, cost, x_ref_dp


def test_select_device_and_unet_shapes(tmp_path: Path):
    dev_cpu = ml_models.select_device("cpu")
    assert dev_cpu.type == "cpu"
    dev_auto = ml_models.select_device()
    assert dev_auto.type in ("cpu", "mps", "cuda")

    model = ml_models.build_curve_segmenter(in_channels=8, n_classes=1, base_channels=16, width_std=96)
    x = torch.randn(2, 8, 128, 148)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 128, 148)

    ckpt = tmp_path / "test_unet.pt"
    ml_models.save_model(model, ckpt, metadata={"test": True})
    assert ckpt.exists()
    loaded, meta = ml_models.load_model(ckpt, device="cpu")
    assert meta.get("test") is True
    with torch.no_grad():
        out2 = loaded(x)
    assert np.allclose(out.numpy(), out2.numpy(), atol=1e-5)


def test_dp_features_and_probability_prediction():
    panel, cost, _ = _synthetic_panel_and_ref(h_dp=160, w_dp=100)
    ic, ir = ml_models.clean_dp_ink(panel, cost, fy=3, fx=2)
    assert ic.shape == cost.shape
    assert ir.shape == cost.shape

    feat = ml_models.extract_dp_features(cost, ink_cls=ic, ink_raw=ir)
    assert feat.shape == (8, cost.shape[0], cost.shape[1])
    assert feat.dtype == np.float32

    model = ml_models.build_curve_segmenter(in_channels=8, n_classes=1, base_channels=8, width_std=64)
    prob = ml_models.predict_track_probabilities(panel, cost, model=model, fy=3, fx=2, tile_h=128, stride_h=64)
    assert prob.shape == cost.shape
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_strict_well_split_and_tile_generation():
    split = mtd.split_wells()
    assert len(split["train"]) == 4
    assert len(split["holdout"]) == 4
    assert set(split["train"]).isdisjoint(set(split["holdout"]))

    panel, cost, x_ref = _synthetic_panel_and_ref(h_dp=320, w_dp=110)
    ic, ir = ml_models.clean_dp_ink(panel, cost, fy=3, fx=2)
    tx_ora = mtd.snap_reference_to_ink(cost, ic, ir, x_ref)
    assert tx_ora.shape == (cost.shape[0],)
    assert np.mean(np.abs(tx_ora - x_ref)) < 3.0

    rec_train = {"well": "W_TRAIN", "cost": cost, "ink_cls": ic, "ink_raw": ir, "x_al": x_ref, "tx_ora": tx_ora}
    rec_hold = {"well": "W_HOLD", "cost": cost, "ink_cls": ic, "ink_raw": ir, "x_al": x_ref, "tx_ora": tx_ora}
    ds = mtd.build_training_dataset([rec_train, rec_hold], allowed_wells=["W_TRAIN"], tile_rows=128, stride_rows=64, width_std=96)
    assert len(ds["X"]) > 0
    assert ds["X"].shape[1:] == (8, 128, 96)
    assert ds["Y"].shape[1:] == (1, 128, 96)
    assert ds["W"].shape[1:] == (1, 128, 96)
    assert set(ds["wells"]) == {"W_TRAIN"}


def test_training_and_leakage_guard(tmp_path: Path):
    panel, cost, x_ref = _synthetic_panel_and_ref(h_dp=280, w_dp=96)
    ic, ir = ml_models.clean_dp_ink(panel, cost, fy=3, fx=2)
    recs = [
        {"well": "W1", "cost": cost, "ink_cls": ic, "ink_raw": ir, "x_al": x_ref},
        {"well": "W2", "cost": cost, "ink_cls": ic, "ink_raw": ir, "x_al": x_ref},
    ]
    with pytest.raises(ValueError, match="leakage"):
        ml_training.train("test_prof", ["W1"], ["W1", "W2"], tmp_path / "m", well_records=recs, epochs=1)

    res = ml_training.train(
        "test_prof",
        ["W1"],
        ["W2"],
        tmp_path / "m",
        well_records=recs,
        epochs=2,
        batch_size=4,
        tile_rows=128,
        stride_rows=64,
        device="cpu",
    )
    assert Path(res["model_path"]).exists()
    assert res["n_train_tiles"] > 0
    assert res["n_val_tiles"] > 0

    ft = ml_training.fine_tune(
        res["model_path"],
        labeled_logs=recs[:1],
        out_dir=tmp_path / "m_ft",
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    assert Path(ft["model_path"]).exists()


def test_trace_track_cost_modes_and_shipped_checkpoint():
    ckpt = ROOT / "models" / "kansas_gr_unet.pt"
    if not ckpt.exists():
        pytest.skip("trained checkpoint models/kansas_gr_unet.pt not present (local only)")
    _, meta = ml_models.load_model(ckpt, device="cpu")
    assert set(meta["train_wells"]).isdisjoint(set(meta["val_wells"]))

    panel, _, x_ref = _synthetic_panel_and_ref(h_dp=200, w_dp=100)
    t_conv = ct.trace_track(panel, cost_mode="conventional")
    t_blend = ct.trace_track(panel, cost_mode="blend", ml_model=ckpt)
    t_ml = ct.trace_track(panel, cost_mode="ml", ml_model=ckpt)

    assert t_conv["cost_mode"] == "conventional"
    assert t_conv["ml_prob"] is None
    assert t_blend["cost_mode"] == "blend"
    assert t_blend["ml_prob"] is not None
    assert t_ml["cost_mode"] == "ml"
    assert len(t_ml["x"]) == len(t_conv["x"])

    fake_batch = [
        {
            "well": "TEST_W",
            "scan": "scan.tif",
            "curve": "GR",
            "depth": np.linspace(1000.0, 1200.0, 200),
            "conf": np.concatenate([np.full(80, 0.95), np.full(40, 0.45), np.full(80, 0.92)]),
        }
    ]
    q = mbd.review_queue(fake_batch, threshold=0.78, min_len=10, min_span_ft=10.0)
    assert len(q) == 1
    assert q[0]["well"] == "TEST_W"
    assert q[0]["mean_conf"] < 0.78
