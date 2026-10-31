# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""ml_models: Compact 2D U-Net cost-map segmenter for well-log curve tracing.

Predicts per-pixel curve probability P(target curve | ink & local geometry) on the
downsampled DP grid (H_dp, W_dp), which is converted into a cost map or blended
with the conventional morphological cost map before running the Rust/Numba DP tracer.
Supports Apple Silicon MPS (`mps`), CUDA (`cuda`), and CPU (`cpu`).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CurveUNet",
    "select_device",
    "build_curve_segmenter",
    "extract_dp_features",
    "extract_track_features",
    "predict_track_probabilities",
    "save_model",
    "load_model",
    "get_cached_model",
]

_MODEL_CACHE: dict[tuple[str, str], tuple[CurveUNet, dict[str, Any], torch.device]] = {}


def select_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device (`mps` > `cuda` > `cpu`)."""
    if prefer is not None:
        return torch.device(prefer)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 4 if out_ch % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CurveUNet(nn.Module):
    """Compact 2D U-Net with horizontal global context for track curve segmentation."""

    def __init__(
        self,
        in_channels: int = 8,
        n_classes: int = 1,
        base_channels: int = 16,
        width_std: int = 96,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.width_std = width_std

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 4
        self.e1 = _ConvBlock(in_channels, c1)
        self.e2 = _ConvBlock(c1, c2)
        self.e3 = _ConvBlock(c2, c3)
        self.bot = _ConvBlock(c3, c4)

        self.d3 = _ConvBlock(c4 + c3, c2)
        self.d2 = _ConvBlock(c2 + c2, c1)
        self.d1 = _ConvBlock(c1 + c1, c1)
        self.head = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        if orig_w != self.width_std:
            x_in = F.interpolate(x, size=(orig_h, self.width_std), mode="bilinear", align_corners=False)
        else:
            x_in = x

        x1 = self.e1(x_in)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        b = self.bot(F.max_pool2d(x3, 2))

        b = b + b.mean(dim=-1, keepdim=True)
        u3 = F.interpolate(b, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        y3 = self.d3(torch.cat([u3, x3], dim=1))

        u2 = F.interpolate(y3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.d2(torch.cat([u2, x2], dim=1))

        u1 = F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.d1(torch.cat([u1, x1], dim=1))

        logits = self.head(y1)
        if orig_w != self.width_std:
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return logits


def build_curve_segmenter(
    n_classes: int = 1,
    in_channels: int = 8,
    base_channels: int = 16,
    width_std: int = 96,
) -> CurveUNet:
    """Construct a CurveUNet instance."""
    return CurveUNet(
        in_channels=in_channels,
        n_classes=n_classes,
        base_channels=base_channels,
        width_std=width_std,
    )


def clean_dp_ink(
    panel: np.ndarray,
    cost: np.ndarray | None = None,
    fy: int = 3,
    fx: int = 2,
    panel_gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build grid-cleaned DP-resolution ink masks (ink_cls, ink_raw) from a track panel."""
    from pylogdigitizer import track_grid_detection as tgd
    from pylogdigitizer.scan_cleanup import line_masks

    if cost is None:
        cost = panel
        ic = (
            cv2.morphologyEx(
                (cost == 0.0).astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((9, 3), np.uint8),
            )
            > 0
        ).astype(np.float32)
        if panel_gray is None:
            return ic, (cost < 0.85).astype(np.float32)
        panel = panel_gray < 128 if panel_gray.dtype == np.uint8 else panel_gray.astype(bool)

    if panel.dtype != bool:
        panel = panel < 128 if panel.max() > 1 else panel > 0
    h2, w2 = cost.shape
    hm, _ = line_masks(panel, max(61, panel.shape[1] // 4))
    clean = panel & ~hm
    vl = tgd.vertical_lines(panel[: min(len(panel), 20000)], 0.5)
    for l in vl:
        clean[:, max(0, l["x0"] - 2) : min(panel.shape[1], l["x1"] + 3)] = False
    ir = (
        clean[: h2 * fy, : w2 * fx]
        .reshape(h2, fy, w2, fx)
        .any(axis=(1, 3))
        .astype(np.float32)
    )
    ic = (
        cv2.morphologyEx(
            (cost == 0.0).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((9, 3), np.uint8),
        )
        > 0
    ).astype(np.float32)
    return ic, ir


def extract_dp_features(
    cost: np.ndarray,
    ink_cls: np.ndarray | None = None,
    ink_raw: np.ndarray | None = None,
) -> np.ndarray:
    """Compute 8-channel geometric/morphological feature tensor (8, H, W) on the DP grid.

    Channels:
      0: ir_close   - gap-closed ink mask (excludes grid lines)
      1: ir_v       - short vertical ink density (15x3 window)
      2: v61        - medium vertical persistence (61x3 window; high on stationary caliper/grid)
      3: v201       - long vertical persistence (201x5 window)
      4: wiggle     - horizontal wiggle richness (high on active curves like GR)
      5: dx_act     - local horizontal derivative activity (|d(ir_close)/dx| smoothed)
      6: rel_x      - normalized track column position in [0, 1]
      7: rel_span   - relative position between leftmost and rightmost active ink in row
    """
    H, W = cost.shape
    if ink_cls is None:
        ic = (
            cv2.morphologyEx(
                (cost == 0.0).astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((9, 3), np.uint8),
            )
            > 0
        ).astype(np.float32)
    else:
        ic = ink_cls.astype(np.float32)

    if ink_raw is None:
        ir = ic.copy()
    else:
        ir = ink_raw.astype(np.float32)

    ir_close = np.maximum(
        ic,
        (cv2.morphologyEx(ir.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8)) > 0).astype(np.float32),
    )
    ir_v = cv2.boxFilter(ir_close, -1, (3, 15))
    v61 = cv2.boxFilter(ir_close, -1, (3, 61))
    v201 = cv2.boxFilter(ir_close, -1, (5, 201))

    w25 = cv2.boxFilter(ir_close, -1, (25, 61))
    wiggle = np.clip(w25 - v61 * 0.25, 0.0, 1.0)

    dx = np.abs(np.diff(ir_close, axis=1, prepend=ir_close[:, :1]))
    dx_act = np.clip(cv2.boxFilter(dx, -1, (19, 41)) * 4.0, 0.0, 1.0)

    rel_x = np.broadcast_to(
        np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :], (H, W)
    ).copy()

    active = ir_v > 0.15
    x_idx = np.arange(W, dtype=np.float32)[None, :]
    x_min = np.where(active, x_idx, W).min(axis=1, keepdims=True)
    x_max = np.where(active, x_idx, 0).max(axis=1, keepdims=True)
    x_min_s = cv2.boxFilter(x_min.astype(np.float32), -1, (1, 101))
    x_max_s = cv2.boxFilter(x_max.astype(np.float32), -1, (1, 101))
    span = np.maximum(x_max_s - x_min_s, 10.0)
    rel_span_pos = np.clip((x_idx - x_min_s) / span, 0.0, 1.0).astype(np.float32)

    return np.stack(
        [ir_close, ir_v, v61, v201, wiggle, dx_act, rel_x, rel_span_pos],
        axis=0,
    ).astype(np.float32)


def resize_features_width(feat: np.ndarray, width_std: int = 96) -> np.ndarray:
    """Resize an (C, H, W) feature tensor horizontally to `width_std` columns."""
    C, H, W = feat.shape
    if W == width_std:
        return feat.astype(np.float32)
    return np.stack(
        [cv2.resize(feat[c], (width_std, H), interpolation=cv2.INTER_LINEAR) for c in range(C)],
        axis=0,
    ).astype(np.float32)


def extract_track_features(
    panel: np.ndarray,
    cost: np.ndarray,
    fy: int = 3,
    fx: int = 2,
) -> np.ndarray:
    """Extract 8-channel DP feature map from full-resolution `panel` and DP `cost`."""
    ic, ir = clean_dp_ink(panel, cost, fy=fy, fx=fx)
    return extract_dp_features(cost, ink_cls=ic, ink_raw=ir)


def predict_track_probabilities(
    panel: np.ndarray | None,
    cost: np.ndarray,
    model: CurveUNet | str | Path,
    fy: int = 3,
    fx: int = 2,
    tile_h: int = 512,
    stride_h: int = 256,
    batch_size: int = 16,
    device: str | torch.device | None = None,
    feat: np.ndarray | None = None,
) -> np.ndarray:
    """Run tiled batched inference over a track and return probability map of shape (H_dp, W_dp)."""
    if isinstance(model, (str, Path)):
        dev_str = str(select_device(str(device) if device is not None else None))
        model_obj, _, dev = get_cached_model(model, device=dev_str)
    else:
        dev = select_device(str(device) if device is not None else None) if not isinstance(device, torch.device) else device
        model_obj = model.to(dev)
        model_obj.eval()

    if feat is None:
        if panel is not None:
            feat = extract_track_features(panel, cost, fy=fy, fx=fx)
        else:
            feat = extract_dp_features(cost)

    _, H, W = feat.shape
    if H <= tile_h:
        pad_h = (8 - H % 8) % 8
        f_pad = np.pad(feat, ((0, 0), (0, pad_h), (0, 0)), mode="edge")
        with torch.no_grad():
            inp = torch.from_numpy(f_pad[None]).to(dev)
            prob = torch.sigmoid(model_obj(inp))[0, 0, :H, :W].cpu().numpy()
        return prob.astype(np.float32)

    starts = list(range(0, max(1, H - tile_h + 1), stride_h))
    if starts[-1] + tile_h < H:
        starts.append(max(0, H - tile_h))

    tiles = np.stack([feat[:, s : s + tile_h, :] for s in starts], axis=0)
    win = np.hanning(tile_h).astype(np.float32)
    win = np.maximum(win, 0.05)[:, None]

    prob_acc = np.zeros((H, W), dtype=np.float32)
    weight_acc = np.zeros((H, 1), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch = torch.from_numpy(tiles[i : i + batch_size]).to(dev)
            preds = torch.sigmoid(model_obj(batch))[:, 0].cpu().numpy()
            for j, p in enumerate(preds):
                s = starts[i + j]
                prob_acc[s : s + tile_h] += p * win
                weight_acc[s : s + tile_h] += win

    return (prob_acc / np.maximum(weight_acc, 1e-6)).astype(np.float32)


def save_model(
    model: CurveUNet,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save CurveUNet weights and architecture config to a checkpoint file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "in_channels": model.in_channels,
            "n_classes": model.n_classes,
            "base_channels": model.base_channels,
            "width_std": model.width_std,
        },
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, p)
    _MODEL_CACHE.clear()
    return p


def load_model(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[CurveUNet, dict[str, Any]]:
    """Load CurveUNet weights and metadata from a checkpoint file."""
    dev = select_device(str(device) if device is not None else None) if not isinstance(device, torch.device) else device
    ckpt = torch.load(Path(path), map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})
    model = CurveUNet(
        in_channels=cfg.get("in_channels", 8),
        n_classes=cfg.get("n_classes", 1),
        base_channels=cfg.get("base_channels", 16),
        width_std=cfg.get("width_std", 96),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    return model, ckpt.get("metadata", {})


def get_cached_model(
    path: str | Path,
    device: str | None = None,
) -> tuple[CurveUNet, dict[str, Any], torch.device]:
    """Load and cache CurveUNet in memory by resolved path and device."""
    dev = select_device(device)
    key = (str(Path(path).resolve()), str(dev))
    if key not in _MODEL_CACHE:
        model, meta = load_model(path, device=dev)
        _MODEL_CACHE[key] = (model, meta, dev)
    return _MODEL_CACHE[key]
