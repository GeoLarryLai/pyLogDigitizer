# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""ml_training: Train and fine-tune CurveUNet models per dataset profile."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import time
import numpy as np

from .. import ml_models, ml_training_data


def _default_records_npz() -> Path:
    """Cached DP training records: ``outputs/tiles/kansas_gr_dp_records.npz`` if present, else
    ``$PYLOGDIGITIZER_CACHE`` or the cache dir (``$XDG_CACHE_HOME`` or ``~/.cache``)/pylogdigitizer."""
    local = Path("outputs/tiles/kansas_gr_dp_records.npz")
    if local.exists():
        return local
    base = os.environ.get("PYLOGDIGITIZER_CACHE") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "pylogdigitizer"
    return Path(base) / "kansas_gr_dp_records.npz"

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _ink_bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    wbce = (bce * weights).sum() / weights.sum().clamp_min(1.0)

    probs = torch.sigmoid(logits)
    ink_gate = (weights >= 1.0).to(logits.dtype)
    p_ink = probs * ink_gate
    t_ink = targets * ink_gate
    inter = (p_ink * t_ink).sum(dim=(1, 2, 3))
    denom = (p_ink + t_ink).sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return wbce + 0.45 * dice_loss


def _compute_ink_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pos = probs[targets > 0.5]
        neg = probs[(weights >= 2.0) & (targets <= 0.5)]
        if len(pos) > 0 and len(neg) > 0:
            ps = pos[::max(1, len(pos) // 2000)]
            ns = neg[::max(1, len(neg) // 2000)]
            return float((ps[:, None] > ns[None, :]).to(targets.dtype).mean().item())
        p_bin = (probs >= 0.3).to(targets.dtype)
        t_bin = (targets > 0.5).to(targets.dtype)
        inter = float((p_bin * t_bin).sum().item())
        denom = float((p_bin + t_bin).sum().item())
        return (2.0 * inter + 1e-5) / (denom + 1e-5)


def load_records_from_npz(npz_path: str | Path, well_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Load precomputed DP-resolution well records from an .npz cache."""
    z = np.load(npz_path)
    if well_ids is None:
        seen = []
        for k in z.files:
            if "__" in k:
                w = k.split("__", 1)[0]
                if w not in seen:
                    seen.append(w)
        well_ids = seen
    records = []
    for w in well_ids:
        if f"{w}__cost" not in z:
            continue
        rec: dict[str, Any] = {
            "well": w,
            "cost": z[f"{w}__cost"],
            "ink_cls": z[f"{w}__ink_cls"] if f"{w}__ink_cls" in z else None,
            "ink_raw": z[f"{w}__ink_raw"] if f"{w}__ink_raw" in z else None,
            "tx_ora": z[f"{w}__tx_ora"] if f"{w}__tx_ora" in z else None,
            "x_al": z[f"{w}__x_al"] if f"{w}__x_al" in z else None,
        }
        for opt_k in ("lab", "fy", "fx", "panel_w", "px_per_ft", "d_dp", "scale", "ref_depth", "ref_val", "shift_ft", "tx_conv"):
            key = f"{w}__{opt_k}"
            if key in z:
                rec[opt_k] = z[key]
        records.append(rec)
    return records


def train_unet_from_tiles(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    W_train: np.ndarray,
    X_val: np.ndarray | None = None,
    Y_val: np.ndarray | None = None,
    W_val: np.ndarray | None = None,
    epochs: int = 12,
    batch_size: int = 24,
    lr: float = 3e-3,
    pos_weight: float = 1.0,
    base_channels: int = 16,
    seed: int = 42,
    device: str | None = None,
    base_model: ml_models.CurveUNet | None = None,
) -> tuple[ml_models.CurveUNet, dict[str, Any]]:
    """Train or fine-tune a CurveUNet on strip tiles using ink-weighted BCE + Dice loss."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for train_unet_from_tiles")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = ml_models.select_device(device)

    in_ch = int(X_train.shape[1]) if len(X_train) > 0 else 8
    w_std = int(X_train.shape[3]) if len(X_train) > 0 else 96
    model = (
        base_model.to(dev)
        if base_model is not None
        else ml_models.CurveUNet(in_channels=in_ch, n_classes=1, base_channels=base_channels, width_std=w_std).to(dev)
    )

    if len(X_train) == 0:
        return model, {"epochs": [], "device": str(dev), "elapsed_s": 0.0}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    n_train = len(X_train)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        tr_losses = []
        tr_dices = []
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(dev)
            yb = torch.from_numpy(Y_train[idx]).to(dev)
            wb = torch.from_numpy(W_train[idx]).to(dev)
            opt.zero_grad()
            logits = model(xb)
            loss = _ink_bce_dice_loss(logits, yb, wb, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))
            tr_dices.append(_compute_ink_dice(logits, yb, wb))
        sched.step()

        val_loss = float(np.mean(tr_losses))
        val_dice = float(np.mean(tr_dices))
        if X_val is not None and Y_val is not None and W_val is not None and len(X_val) > 0:
            model.eval()
            v_losses = []
            v_dices = []
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    xb = torch.from_numpy(X_val[i : i + batch_size]).to(dev)
                    yb = torch.from_numpy(Y_val[i : i + batch_size]).to(dev)
                    wb = torch.from_numpy(W_val[i : i + batch_size]).to(dev)
                    logits = model(xb)
                    v_losses.append(float(_ink_bce_dice_loss(logits, yb, wb, pos_weight=pos_weight).item()))
                    v_dices.append(_compute_ink_dice(logits, yb, wb))
            val_loss = float(np.mean(v_losses))
            val_dice = float(np.mean(v_dices))

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": ep,
                "train_loss": round(float(np.mean(tr_losses)), 5),
                "train_ink_dice": round(float(np.mean(tr_dices)), 5),
                "val_loss": round(val_loss, 5),
                "val_ink_dice": round(val_dice, 5),
            }
        )

    model.load_state_dict(best_state)
    model.eval()
    elapsed_s = round(time.perf_counter() - t0, 3)
    return model, {
        "device": str(dev),
        "elapsed_s": elapsed_s,
        "best_val_loss": round(best_val_loss, 5),
        "best_val_ink_dice": round(max((h["val_ink_dice"] for h in history), default=0.0), 5),
        "epochs": history,
    }


def train(
    profile: str,
    train_ids: list[str],
    val_ids: list[str],
    out_dir: str | Path,
    well_records: list[dict[str, Any]] | None = None,
    cache_npz: str | Path | None = None,
    corrections_dir: str | Path | None = None,
    epochs: int = 12,
    batch_size: int = 24,
    lr: float = 3e-3,
    seed: int = 42,
    device: str | None = None,
    tile_rows: int = 256,
    stride_rows: int = 128,
) -> dict[str, Any]:
    """Train CurveUNet on ``train_ids`` and validate on ``val_ids`` with strict zero well overlap."""
    train_set = {str(w) for w in train_ids}
    val_set = {str(w) for w in val_ids}
    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"Data leakage error: wells present in both train_ids and val_ids: {sorted(overlap)}")

    if well_records is None:
        default_npz = Path(cache_npz) if cache_npz else _default_records_npz()
        if default_npz.exists():
            well_records = load_records_from_npz(default_npz, list(train_ids) + list(val_ids))
        else:
            needed = set(train_ids) | set(val_ids)
            well_records = [
                ml_training_data.build_well_record_from_scan(spec)
                for spec in ml_training_data.WELL_SPECS
                if spec["well"] in needed
            ]

    ds_train = ml_training_data.build_training_dataset(
        well_records,
        allowed_wells=train_ids,
        corrections_dir=corrections_dir,
        tile_rows=tile_rows,
        stride_rows=stride_rows,
    )
    ds_val = ml_training_data.build_training_dataset(
        well_records,
        allowed_wells=val_ids,
        corrections_dir=None,
        tile_rows=tile_rows,
        stride_rows=tile_rows,
    )

    model, train_info = train_unet_from_tiles(
        ds_train["X"],
        ds_train["Y"],
        ds_train["W"],
        X_val=ds_val["X"],
        Y_val=ds_val["Y"],
        W_val=ds_val["W"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "unet.pt"
    meta = {
        "profile": profile,
        "train_wells": list(train_ids),
        "val_wells": list(val_ids),
        "n_train_tiles": int(len(ds_train["X"])),
        "n_val_tiles": int(len(ds_val["X"])),
        "train_info": train_info,
    }
    ml_models.save_model(model, ckpt_path, metadata=meta)

    hist_path = out_path / "training_history.json"
    hist_path.write_text(json.dumps(meta, indent=2))
    return {
        "profile": profile,
        "model_path": str(ckpt_path),
        "history_path": str(hist_path),
        "train_wells": list(train_ids),
        "val_wells": list(val_ids),
        "n_train_tiles": int(len(ds_train["X"])),
        "n_val_tiles": int(len(ds_val["X"])),
        **train_info,
    }


def fine_tune(
    base_model: str | Path,
    labeled_logs: list[str | Path | dict[str, Any]] | None = None,
    out_dir: str | Path = "outputs/models_ft",
    corrections_dir: str | Path | None = None,
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str | None = None,
    corrected_logs: list[str | Path | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fine-tune an existing CurveUNet checkpoint on user corrections or newly labeled logs."""
    if labeled_logs is None:
        labeled_logs = corrected_logs or []
    model, _ = ml_models.load_model(base_model, device=device)
    records = [r for r in labeled_logs if isinstance(r, dict)]
    allowed = []
    for r in labeled_logs:
        if isinstance(r, dict):
            allowed.append(str(r.get("well", "")))
        else:
            p = Path(r)
            if p.exists() and (p / "metadata.json").exists():
                if corrections_dir is None:
                    corrections_dir = p.parent
                allowed.append(p.name.split("__")[0])
            else:
                allowed.append(str(r))
    allowed = list(dict.fromkeys(allowed)) or None
    ds = ml_training_data.build_training_dataset(
        records,
        allowed_wells=allowed,
        corrections_dir=corrections_dir,
    )
    model, info = train_unet_from_tiles(
        ds["X"],
        ds["Y"],
        ds["W"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        base_model=model,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "unet.pt"
    ml_models.save_model(model, ckpt_path, metadata={"fine_tuned_from": str(base_model), **info})
    return {
        "model_path": str(ckpt_path),
        "n_tiles": int(len(ds["X"])),
        **info,
    }
