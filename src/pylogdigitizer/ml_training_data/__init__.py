# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""ml_training_data: build ML training samples.

Two sources:
1. Paired LAS rendered into scan pixel space (``las_to_pixel_labels``) - automatic labels.
2. User corrections saved by the web editor / ``assisted_tracing`` (``save_correction``) -
   the folder format below, read back with ``load_correction`` / ``list_corrections``.

Correction folder format (one folder per curve section, under e.g. ``training_data/corrections/``)::

    <well>__<scan>__<curve>__<YYYYmmdd-HHMMSS>/
        image.png          rectified track panel, uint8 (0 = ink, 255 = paper), rows x width
        mask.png           curve-pixel label, uint8 (255 = curve, 0 = background), same shape;
                           only on-scale samples are drawn (wrapped/backup samples are not)
        trace.csv          per DP sample: sheet_row, panel_row, x_panel, depth_ft, value, conf, off_scale
        anchors.json       user anchors [[sheet_row, x_panel], ...] (re-trace constraints)
        calibration.json   depth anchors [[sheet_row, depth_ft], ...], px_per_ft, scale {left,right,type,backup,wrap_side}
        metadata.json      format_version, well, scan, curve, track, panel_rows, width, source, created, backend

Row coordinates are rows of the working (deskewed) sheet image; ``panel_row = sheet_row - panel_rows[0]``.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
import csv, datetime, json, re
import numpy as np

FORMAT_VERSION = 1
FILES = ("image.png", "mask.png", "trace.csv", "anchors.json", "calibration.json", "metadata.json")
TRACE_FIELDS = ("sheet_row", "panel_row", "x_panel", "depth_ft", "value", "conf", "off_scale")


def las_to_pixel_labels(scan_shape: tuple[int, int], depth_map: Callable, scale_map: Callable, depth: np.ndarray,
                        values: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Render a reference curve into a label mask. ``depth_map``: depth -> row (e.g. DepthMap.inverse),
    ``scale_map``: value -> column (e.g. ScaleMap.inverse). Consecutive samples are joined."""
    m = np.zeros(scan_shape, np.uint8)
    r = np.asarray(depth_map(np.asarray(depth, float)), float); x = np.asarray(scale_map(np.asarray(values, float)), float)
    ok = np.isfinite(r) & np.isfinite(x) & (x >= 0) & (x < scan_shape[1])
    _polylines(m, r, x, ok, thickness)
    return m > 0


def _polylines(m: np.ndarray, r, x, ok, thickness: int):
    import cv2
    idx = np.flatnonzero(ok)
    if not len(idx):
        return
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for run in runs:
        pts = np.c_[np.round(x[run]), np.round(r[run])].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m, [pts], False, 255, thickness)


def make_tiles(image: np.ndarray, mask: np.ndarray, size: int = 512, stride: int | None = None,
               min_label_frac: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut (image, mask) into size x size tiles (row-wise; the panel width is padded to ``size``).
    Tiles whose label fraction is below ``min_label_frac`` are skipped."""
    stride = stride or size
    h, w = image.shape[:2]
    pw = max(size, w)
    img = np.full((h, pw), 255, image.dtype) if image.ndim == 2 else np.full((h, pw) + image.shape[2:], 255, image.dtype)
    img[:, :w] = image; msk = np.zeros((h, pw), mask.dtype); msk[:, :w] = mask
    out = []
    for y in range(0, max(1, h - size + 1), stride):
        for x in range(0, pw - size + 1, stride):
            t, m = img[y:y + size, x:x + size], msk[y:y + size, x:x + size]
            if t.shape[0] < size or (m > 0).mean() < min_label_frac:
                continue
            out.append((t, m))
    return out


def split_by_well(well_ids: list[str], fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0) -> dict[str, list[str]]:
    """Deterministic train/val/test split by well (no well in two splits)."""
    w = sorted(set(well_ids)); rng = np.random.default_rng(seed); rng.shuffle(w)
    n = len(w); a = int(round(fractions[0] * n)); b = a + int(round(fractions[1] * n))
    return {"train": w[:a], "val": w[a:b], "test": w[b:]}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def save_correction(root: str | Path, image: np.ndarray, trace: dict[str, np.ndarray], anchors: list,
                    calibration: dict, metadata: dict, thickness: int = 3) -> Path:
    """Write one correction sample (see module docstring). ``image``: bool ink panel or uint8
    gray; ``trace``: arrays named as ``TRACE_FIELDS``. Returns the sample folder."""
    from PIL import Image
    root = Path(root); stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "__".join(_slug(metadata.get(k, "")) for k in ("well", "scan", "curve")) + "__" + stamp
    d = root / name; k = 1
    while d.exists():
        d = root / f"{name}_{k}"; k += 1
    d.mkdir(parents=True)
    img = np.where(image, 0, 255).astype(np.uint8) if image.dtype == bool else np.asarray(image, np.uint8)
    Image.fromarray(img).save(d / "image.png")
    m = np.zeros(img.shape[:2], np.uint8)
    pr = np.asarray(trace["panel_row"], float); xp = np.asarray(trace["x_panel"], float)
    ok = ~np.asarray(trace["off_scale"], bool) & np.isfinite(xp)
    _polylines(m, pr, xp, ok, thickness)
    Image.fromarray(m).save(d / "mask.png")
    n = len(pr)
    with open(d / "trace.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(TRACE_FIELDS)
        cols = [np.asarray(trace[k]) for k in TRACE_FIELDS]
        for i in range(n):
            wr.writerow([int(c[i]) if k == "off_scale" else round(float(c[i]), 4) for k, c in zip(TRACE_FIELDS, cols)])
    (d / "anchors.json").write_text(json.dumps([[float(a), float(b)] for a, b in anchors], indent=1))
    (d / "calibration.json").write_text(json.dumps(calibration, indent=1, default=float))
    meta = dict(metadata, format_version=FORMAT_VERSION, created=datetime.datetime.now().isoformat(timespec="seconds"),
                shape=list(img.shape[:2]))
    (d / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    return d


def load_correction(path: str | Path) -> dict[str, Any]:
    """Read a correction folder -> dict(image uint8, mask bool, trace dict of arrays, anchors,
    calibration, metadata). Raises FileNotFoundError naming the missing file."""
    from PIL import Image
    d = Path(path)
    for f in FILES:
        if not (d / f).exists():
            raise FileNotFoundError(f"{d}: missing {f} (expected files: {', '.join(FILES)})")
    with open(d / "trace.csv") as f:
        rows = list(csv.DictReader(f))
    trace = {k: np.array([float(r[k]) for r in rows]) for k in TRACE_FIELDS}
    trace["off_scale"] = trace["off_scale"].astype(bool)
    return dict(path=d, image=np.asarray(Image.open(d / "image.png")), mask=np.asarray(Image.open(d / "mask.png")) > 0,
                trace=trace, anchors=json.loads((d / "anchors.json").read_text()),
                calibration=json.loads((d / "calibration.json").read_text()),
                metadata=json.loads((d / "metadata.json").read_text()))


def list_corrections(root: str | Path) -> list[Path]:
    """All correction folders under ``root`` (those with a metadata.json), sorted."""
    root = Path(root)
    return sorted(p.parent for p in root.glob("*/metadata.json")) if root.exists() else []


def corrections_to_tiles(root: str | Path, size: int = 512, min_label_frac: float = 0.001):
    """Yield (well, image_tile, mask_tile) for every saved correction (input for ml_training)."""
    for d in list_corrections(root):
        s = load_correction(d)
        for t, m in make_tiles(s["image"], s["mask"], size, min_label_frac=min_label_frac):
            yield s["metadata"].get("well", ""), t, m


DEFAULT_HOLDOUT_WELLS: tuple[str, ...] = (
    "USA_A_16",
    "WEATHERFORD_B_1",
    "BACASTOW_1",
    "GOHEEN_UNIT_1",
)
DEFAULT_TRAIN_WELLS: tuple[str, ...] = (
    "Hall_D_4",
    "MAKEMSON_A_1",
    "Chitwood_12",
    "Minzer_1",
)

WELL_SPECS: list[dict[str, Any]] = [
    {"well": "USA_A_16", "folder": "USA_A_16_15-055-20536", "scan": "4342.tif", "las": "1042553273.las", "curve": "GR", "track": "track1", "scale": (0.0, 150.0), "backup": (150.0, 300.0), "use_hatch": True, "first_depth": 1500.0},
    {"well": "WEATHERFORD_B_1", "folder": "WEATHERFORD_B_1_15-129-20464", "scan": "194290.tif", "las": "1001178972.las", "curve": "GR", "track": "track1", "scale": (0.0, 150.0), "backup": (150.0, 300.0), "use_hatch": True},
    {"well": "BACASTOW_1", "folder": "BACASTOW_1_15-035-23415", "scan": "169724.tif", "las": "1043694739.las", "curve": "GR", "track": "track1", "scale": (0.0, 150.0), "use_hatch": False},
    {"well": "GOHEEN_UNIT_1", "folder": "GOHEEN_UNIT_1_15-191-30175", "scan": "171203.tif", "las": "1042553292.las", "curve": "GR", "track": "track1", "scale": (0.0, 160.0), "backup": (160.0, 320.0), "use_hatch": True},
    {"well": "Hall_D_4", "folder": "Hall_D_4_15-051-05370", "scan": "193004.tif", "las": "1043914416.las", "curve": "GR", "track": "track1", "scale": (10.0, 110.0), "backup": (110.0, 210.0), "use_hatch": False},
    {"well": "MAKEMSON_A_1", "folder": "MAKEMSON_A_1_15-081-20018", "scan": "225992.tif", "las": "1044080517.las", "curve": "GR", "track": "track1", "scale": (0.0, 120.0), "use_hatch": False},
    {"well": "Chitwood_12", "folder": "Chitwood_12_15-151-10469", "scan": "260245.tif", "las": "1044194859.las", "curve": "GR:2", "track": "track1", "scale": (0.0, 160.0), "backup": (160.0, 320.0), "use_hatch": False},
    {"well": "Minzer_1", "folder": "Minzer_1_15-077-30177", "scan": "171665.tif", "las": "1044076084.las", "curve": "GR", "track": "track1", "scale": (0.0, 100.0), "use_hatch": False},
]


def split_wells(
    well_ids: list[str] | tuple[str, ...] | None = None,
    holdout_wells: list[str] | tuple[str, ...] = DEFAULT_HOLDOUT_WELLS,
) -> dict[str, list[str]]:
    """Strict well-level train/holdout split with zero well overlap."""
    if well_ids is None:
        well_ids = list(DEFAULT_TRAIN_WELLS) + list(DEFAULT_HOLDOUT_WELLS)
    hold_set = {str(w) for w in holdout_wells}
    seen: list[str] = []
    for w in well_ids:
        ws = str(w)
        if ws not in seen:
            seen.append(ws)
    train = [w for w in seen if w not in hold_set]
    holdout = [w for w in seen if w in hold_set]
    assert set(train).isdisjoint(set(holdout)), "Train and holdout well sets must be disjoint"
    return {"train": train, "holdout": holdout}


def snap_reference_to_ink(
    cost: np.ndarray,
    ink_cls: np.ndarray,
    ink_raw: np.ndarray,
    x_ref: np.ndarray,
) -> np.ndarray:
    """Snap LAS-projected column positions onto scan curve ink using a reference-guided DP trace."""
    from .. import curve_tracing as ct

    H, W = cost.shape
    xs = np.arange(W, dtype=np.float32)[None, :]
    dist = np.where(np.isfinite(x_ref[:, None]), np.abs(xs - x_ref[:, None]), 0.0)
    c_ora = (
        np.where(
            cost == 0.0,
            0.0,
            np.where(ink_cls > 0, 0.15, np.where(ink_raw > 0, 0.35, 0.85)),
        )
        + 0.015 * dist
    )
    return ct.trace(c_ora.astype(np.float32), max_step=max(5, W // 2), smooth=0.02).astype(np.int32)


def build_dp_strip_tiles(
    cost: np.ndarray,
    ink_cls: np.ndarray | None = None,
    ink_raw: np.ndarray | None = None,
    tx_target: np.ndarray | None = None,
    tx_conv: np.ndarray | None = None,
    valid_rows: np.ndarray | None = None,
    tile_rows: int = 256,
    stride_rows: int = 128,
    width_std: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract multi-channel strip tiles (X, Y, W_mask) at DP resolution for CurveUNet training."""
    import cv2
    from .. import ml_models

    H, W = cost.shape
    if ink_cls is None or ink_raw is None:
        ink_cls, ink_raw = ml_models.clean_dp_ink(cost)
    feat = ml_models.extract_dp_features(cost, ink_cls=ink_cls, ink_raw=ink_raw)
    feat_std = ml_models.resize_features_width(feat, width_std=width_std)

    if tx_target is None:
        raise ValueError("tx_target is required to build training labels")
    if valid_rows is None:
        valid_rows = np.ones(H, dtype=bool)
    else:
        valid_rows = np.asarray(valid_rows, dtype=bool)

    xs = np.arange(W, dtype=np.float32)[None, :]
    ir_close = feat[0] > 0.5
    v61 = feat[2]
    near_target = (np.abs(xs - tx_target[:, None]) <= 3.0) & valid_rows[:, None]
    any_ink = ir_close | (ink_cls > 0) | (ink_raw > 0) | (np.abs(xs - tx_target[:, None]) <= 1.5)
    y_full = (near_target & any_ink).astype(np.float32)
    y_std = cv2.resize(y_full, (width_std, H), interpolation=cv2.INTER_LINEAR)
    y_std = (y_std >= 0.25).astype(np.float32)

    # Balanced supervision: strongly weight target vs distractor curve on disagreement rows
    if tx_conv is not None:
        disagree = (np.abs(tx_conv - tx_target) > max(5.0, 0.04 * W)) & valid_rows
        near_conv = (np.abs(xs - tx_conv[:, None]) <= 3.5) & disagree[:, None] & ~near_target
        pos_disagree = near_target & disagree[:, None]
    else:
        near_conv = np.zeros((H, W), dtype=bool)
        pos_disagree = np.zeros((H, W), dtype=bool)

    stat_neg = (v61 > 0.32) & (np.abs(xs - tx_target[:, None]) > 6.0) & ~near_target
    w_full = np.full((H, W), 0.15, dtype=np.float32)
    w_full[near_target] = 1.5
    w_full[stat_neg] = 6.0
    w_full[near_conv] = 20.0
    w_full[pos_disagree] = 20.0
    w_full = w_full * valid_rows[:, None].astype(np.float32)
    w_std = cv2.resize(w_full, (width_std, H), interpolation=cv2.INTER_LINEAR)

    X_list, Y_list, W_list = [], [], []
    for r0 in range(0, max(1, H - tile_rows + 1), stride_rows):
        r1 = r0 + tile_rows
        if r1 > H:
            continue
        if valid_rows[r0:r1].mean() < 0.5:
            continue
        yt = y_std[r0:r1]
        if yt.sum() < 20:
            continue
        X_list.append(feat_std[:, r0:r1, :])
        Y_list.append(yt[None, :, :])
        W_list.append(w_std[r0:r1][None, :, :])

    if not X_list:
        C = feat_std.shape[0]
        return (
            np.zeros((0, C, tile_rows, width_std), dtype=np.float32),
            np.zeros((0, 1, tile_rows, width_std), dtype=np.float32),
            np.zeros((0, 1, tile_rows, width_std), dtype=np.float32),
        )
    return (
        np.stack(X_list, axis=0).astype(np.float32),
        np.stack(Y_list, axis=0).astype(np.float32),
        np.stack(W_list, axis=0).astype(np.float32),
    )


def build_training_dataset(
    well_records: list[dict[str, Any]],
    allowed_wells: list[str] | tuple[str, ...] | None = None,
    corrections_dir: str | Path | None = None,
    tile_rows: int = 256,
    stride_rows: int = 128,
    width_std: int = 96,
) -> dict[str, Any]:
    """Build combined strip-tile dataset from paired LAS well records and optional editor corrections.
    Strictly filters both well_records and corrections to ``allowed_wells`` when provided."""
    allowed_set = {str(w) for w in allowed_wells} if allowed_wells is not None else None
    X_all, Y_all, W_all, tile_wells = [], [], [], []

    for rec in well_records:
        w = str(rec["well"])
        if allowed_set is not None and w not in allowed_set:
            continue
        cost = rec["cost"]
        ic = rec.get("ink_cls")
        ir = rec.get("ink_raw")
        tx_target = rec.get("tx_ora")
        if tx_target is None and "x_al" in rec:
            if ic is None or ir is None:
                from .. import ml_models
                ic, ir = ml_models.clean_dp_ink(cost)
            tx_target = snap_reference_to_ink(cost, ic, ir, rec["x_al"])
        valid = np.isfinite(rec["x_al"]) if "x_al" in rec else np.ones(cost.shape[0], dtype=bool)
        Xt, Yt, Wt = build_dp_strip_tiles(
            cost,
            ink_cls=ic,
            ink_raw=ir,
            tx_target=tx_target,
            tx_conv=rec.get("tx_conv"),
            valid_rows=valid,
            tile_rows=tile_rows,
            stride_rows=stride_rows,
            width_std=width_std,
        )
        if len(Xt) > 0:
            X_all.append(Xt)
            Y_all.append(Yt)
            W_all.append(Wt)
            tile_wells.extend([w] * len(Xt))

    if corrections_dir is not None and Path(corrections_dir).exists():
        from .. import curve_tracing as ct, ml_models

        latest_by_key = {}
        for cp in list_corrections(corrections_dir):
            parts = cp.name.split("__")
            key = tuple(parts[:3]) if len(parts) >= 3 else (cp.name,)
            latest_by_key[key] = cp
        for corr_path in latest_by_key.values():
            corr = load_correction(corr_path)
            w = str(corr["metadata"].get("well", ""))
            if allowed_set is not None and w not in allowed_set:
                continue
            img = corr["image"]
            pan_bool = (img < 128) if img.dtype == np.uint8 else img.astype(bool)
            fy, fx = 3, 2
            cost_dp, _ = ct.build_cost(pan_bool, min_width=3, fy=fy, fx=fx, min_area=400, straight_len=121, return_labels=True)
            H_dp, W_dp = cost_dp.shape
            ic_dp, ir_dp = ml_models.clean_dp_ink(pan_bool, cost_dp, fy=fy, fx=fx)
            xp = np.asarray(corr["trace"]["x_panel"], dtype=np.float32)
            pr = np.asarray(corr["trace"]["panel_row"], dtype=np.float32)
            off = np.asarray(corr["trace"]["off_scale"], dtype=bool)
            dp_rows = np.arange(H_dp, dtype=np.float32) * fy + fy / 2.0
            tx_corr = np.round(np.interp(dp_rows, pr, xp) / fx).astype(np.int32)
            tx_corr = np.clip(tx_corr, 0, W_dp - 1)
            valid_corr = np.interp(dp_rows, pr, (~off).astype(np.float32)) > 0.5
            Xt, Yt, Wt = build_dp_strip_tiles(
                cost_dp,
                ink_cls=ic_dp,
                ink_raw=ir_dp,
                tx_target=tx_corr,
                valid_rows=valid_corr,
                tile_rows=tile_rows,
                stride_rows=stride_rows,
                width_std=width_std,
            )
            if len(Xt) > 0:
                X_all.append(Xt)
                Y_all.append(Yt)
                W_all.append(Wt)
                tile_wells.extend([w] * len(Xt))

    if not X_all:
        return {
            "X": np.zeros((0, 8, tile_rows, width_std), dtype=np.float32),
            "Y": np.zeros((0, 1, tile_rows, width_std), dtype=np.float32),
            "W": np.zeros((0, 1, tile_rows, width_std), dtype=np.float32),
            "wells": [],
        }
    return {
        "X": np.concatenate(X_all, axis=0),
        "Y": np.concatenate(Y_all, axis=0),
        "W": np.concatenate(W_all, axis=0),
        "wells": tile_wells,
    }


def build_well_record_from_scan(
    spec: dict[str, Any],
    data_root: str | Path = "example_data",
    fy: int = 3,
    fx: int = 2,
) -> dict[str, Any]:
    """Extract DP-resolution cost, grid-cleaned ink masks, and aligned LAS reference trajectory for one well."""
    from .. import las_export, ml_models, qc_compare as qc, workflows as wf

    w = spec["well"]
    folder = spec.get("folder", w)
    scan_p = Path(data_root) / folder / spec["scan"]
    las_p = Path(data_root) / folder / spec["las"]
    kw = dict(
        track=spec.get("track", "track1"),
        scale=tuple(spec.get("scale", (0.0, 150.0))),
        backup=tuple(spec["backup"]) if spec.get("backup") else None,
        use_hatch=bool(spec.get("use_hatch", True)),
        first_depth=spec.get("first_depth"),
        section=spec.get("section"),
        cost_mode="conventional",
        fy=fy,
        fx=fx,
    )
    try:
        res = wf.digitize_curve(scan_p, **kw)
    except wf.SectionChoiceNeeded as e:     # several log sections: train on the longest one
        kw["section"] = max(e.sections, key=lambda c: c["base"] - c["top"])["index"]
        res = wf.digitize_curve(scan_p, **kw)
    pan = res["panel"]
    T = res["trace"]
    cost = T["cost_conv"]
    ic, ir = ml_models.clean_dp_ink(pan, cost, fy=fy, fx=fx)
    tx_conv = np.clip(np.round(np.asarray(T["x"], float) / fx).astype(np.int32), 0, cost.shape[1] - 1)

    df = las_export.read_las(las_p).df()
    curve = spec.get("curve", "GR")
    ref = df[curve].dropna()
    ref = ref[ref > -999]
    comp = qc.compare(ref.index.values.astype(float), ref.values.astype(float), res["depth"], res["value"])
    shift = float(comp["shift_ft"])

    s0, s1 = spec.get("scale", (0.0, 150.0))
    W_dp = cost.shape[1]
    val_at_d = np.interp(res["depth"] + shift, ref.index.values.astype(float), ref.values.astype(float), left=np.nan, right=np.nan)
    x_al = (val_at_d - s0) / (s1 - s0) * (W_dp - 1)
    in_bounds = (val_at_d >= min(s0, s1)) & (val_at_d <= max(s0, s1))
    x_al = np.where(in_bounds, x_al, np.nan).astype(np.float32)
    tx_ora = snap_reference_to_ink(cost, ic, ir, x_al)
    return {
        "well": w,
        "cost": cost,
        "ink_cls": ic,
        "ink_raw": ir,
        "x_al": x_al,
        "tx_ora": tx_ora,
        "tx_conv": tx_conv,
    }
