# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""ml_batch_digitize: Batch digitize scans with a trained U-Net profile, compute per-depth
confidence, validate against paired LAS (when available), and route low-confidence intervals
to a prioritized human review queue.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import json
import time
import numpy as np

from pylogdigitizer import (
    curve_tracing as ct,
    las_export,
    ml_models,
    ml_training_data as mtd,
    qc_compare as qc,
    workflows as wf,
)

__all__ = [
    "digitize_one",
    "digitize_batch",
    "review_queue",
]

UNITS = {"GR": "API", "DT": "US/F", "ILD": "OHMM", "RHOB": "G/C3", "NPHI": "V/V", "SP": "MV", "CALI": "IN"}


def _resolve_scan_spec(item: str | Path | dict[str, Any], data_root: str | Path = "example_data") -> dict[str, Any]:
    if isinstance(item, dict):
        spec = dict(item)
        if "scan_path" not in spec and "scan" in spec:
            p = Path(spec["scan"])
            if not p.exists() and "well" in spec:
                p = Path(data_root) / spec["well"] / spec["scan"]
            spec["scan_path"] = str(p)
        if "las_path" not in spec and "las" in spec and spec["las"]:
            p = Path(spec["las"])
            if not p.exists() and "well" in spec:
                p = Path(data_root) / spec["well"] / spec["las"]
            spec["las_path"] = str(p)
        spec.setdefault("well", Path(spec["scan_path"]).parent.name)
        spec.setdefault("curve", "GR")
        spec.setdefault("track", "track1")
        spec.setdefault("scale", (0.0, 150.0))
        return spec

    p = Path(item)
    well_name = p.parent.name if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg") else p.name
    for s in mtd.WELL_SPECS:
        if s["well"] == well_name or s.get("folder") == well_name or (p.is_file() and s["scan"] == p.name):
            folder = s.get("folder", s["well"])
            scan_p = p if p.is_file() else Path(data_root) / folder / s["scan"]
            las_p = Path(data_root) / folder / s["las"]
            out = dict(s)
            out["scan_path"] = str(scan_p)
            out["las_path"] = str(las_p) if las_p.exists() else None
            out.setdefault("curve", "GR")
            return out

    if p.is_dir():
        tifs = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
        if not tifs:
            raise FileNotFoundError(f"No TIFF scans found in {p}")
        scan_p = tifs[0]
        las_files = sorted(p.glob("*.las"))
    else:
        scan_p = p
        las_files = sorted(scan_p.parent.glob("*.las"))

    return {
        "well": scan_p.parent.name,
        "scan": scan_p.name,
        "scan_path": str(scan_p),
        "las_path": str(las_files[0]) if las_files else None,
        "curve": "GR",
        "track": "track1",
        "scale": (0.0, 150.0),
    }


def _resolve_profile_model(profile: str | Path | None, model_path: str | Path | None = None) -> Path | None:
    if model_path is not None:
        return Path(model_path)
    if profile is None:
        return None
    p = Path(profile)
    if p.is_file():
        return p
    if (p / "unet.pt").is_file():
        return p / "unet.pt"
    repo_root = Path(__file__).resolve().parents[2]
    for cand in (
        Path(f"models/{profile}/unet.pt"),
        Path("models/kansas_gr_unet.pt"),
        repo_root / "models" / str(profile) / "unet.pt",
        repo_root / "models" / "kansas_gr_unet.pt",
    ):
        if cand.is_file():
            return cand
    return None


def digitize_one(
    spec_or_path: str | Path | dict[str, Any],
    out_dir: str | Path,
    model_path: str | Path | None = None,
    cost_mode: str = "blend",
    compare_conventional: bool = True,
    review_threshold: float = 0.78,
    min_review_rows: int = 12,
    data_root: str | Path = "example_data",
) -> dict[str, Any]:
    """Digitize one well scan using the U-Net cost map + DP tracer and compute confidence & QC."""
    t0 = time.time()
    spec = _resolve_scan_spec(spec_or_path, data_root=data_root)
    well = spec["well"]
    curve = spec.get("curve", "GR")
    scan_path = spec["scan_path"]
    las_path = spec.get("las_path")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    kw: dict[str, Any] = dict(
        track=spec.get("track", "track1"),
        scale=tuple(spec.get("scale", (0.0, 150.0))),
    )
    for opt in ("first_depth", "label_interval", "backup", "wrap_side", "use_hatch", "x_range", "rows", "min_width"):
        if opt in spec and spec[opt] is not None:
            kw[opt] = spec[opt]

    res_ml = wf.digitize_curve(
        scan_path,
        cost_mode=cost_mode,
        ml_model=model_path,
        **kw,
    )
    T = res_ml["trace"]
    depth = np.asarray(res_ml["depth"], dtype=float)
    value = np.asarray(res_ml["value"], dtype=float)
    dp_conf = np.asarray(res_ml["conf"], dtype=float)
    ml_conf = np.asarray(T.get("ml_conf", dp_conf), dtype=float)

    # Combined confidence: high when on ink AND backed by strong row-relative U-Net probability
    if "prob_map" in T:
        prob = T["prob_map"]
        c0 = T["cost_conv"] == 0.0
        p_max = np.where(c0, prob, 0.0).max(axis=1)
        p_path = prob[np.arange(len(T["path"])), T["path"]]
        rel_prob = np.where(p_max > 0.02, np.clip(p_path / np.maximum(p_max, 0.05), 0.0, 1.0), 0.5)
        k = np.ones(25) / 25.0
        rel_prob_s = np.clip(np.convolve(rel_prob, k, mode="same"), 0.0, 1.0)
        comb_conf = np.clip(0.55 * dp_conf + 0.45 * rel_prob_s, 0.0, 1.0)
    else:
        comb_conf = dp_conf

    low_spans = ct.low_confidence_intervals(comb_conf, threshold=review_threshold, min_len=min_review_rows)
    intervals = []
    for a, b in low_spans:
        b_idx = min(b - 1, len(depth) - 1)
        d0, d1 = float(depth[a]), float(depth[b_idx])
        r0_s, r1_s = float(T["rows"][a]), float(T["rows"][b_idx])
        intervals.append(
            {
                "well": well,
                "scan": Path(scan_path).name,
                "curve": curve,
                "dp_row_start": int(a),
                "dp_row_end": int(b),
                "sheet_row_start": round(r0_s, 1),
                "sheet_row_end": round(r1_s, 1),
                "depth_top_ft": round(min(d0, d1), 1),
                "depth_base_ft": round(max(d0, d1), 1),
                "span_ft": round(abs(d1 - d0), 1),
                "mean_conf": round(float(comb_conf[a:b].mean()), 3),
                "min_conf": round(float(comb_conf[a:b].min()), 3),
            }
        )

    las_out = out_root / f"{well}_{curve}.las"
    g_d, g_v = las_export.resample(depth, value, 0.5)
    las_export.write_las(
        las_out,
        g_d,
        {curve: g_v},
        units={curve: UNITS.get(curve, "")},
        well={"WELL": well, "SRVC": f"pyLogDigitizer ({cost_mode})"},
    )

    qc_ml = None
    qc_conv = None
    if las_path and Path(las_path).exists():
        df = las_export.read_las(las_path).df()
        if curve in df.columns:
            ref = df[curve].dropna()
            ref = ref[ref > -999]
            c_ml = qc.compare(ref.index.values.astype(float), ref.values.astype(float), depth, value)
            qc_ml = dict(c_ml["aligned"], shift_ft=float(c_ml["shift_ft"]))
            if compare_conventional:
                p_par = T["params"]
                pad = T["pad"]
                c_conv_pad = np.pad(T["cost_conv"], ((0, 0), (1, 1)), constant_values=0.8) if pad else T["cost_conv"]
                p_conv = ct.trace(
                    c_conv_pad,
                    max_step=p_par["max_step"],
                    smooth=p_par["smooth"],
                    seg_weight=p_par["seg_weight"],
                    labels=T["labels"],
                    switch_cost=p_par["switch_cost"],
                    use=p_par["use"],
                )
                T_conv = ct.finish_trace(c_conv_pad, p_conv, T["row0"], T["fy"], T["fx"], pad, p_par["edge_frac"])
                if "_hatch_rows" in T:
                    T_conv["_hatch_rows"] = T["_hatch_rows"]
                    T_conv["_hatch_rows_key"] = T.get("_hatch_rows_key")
                v_conv, _ = wf.trace_to_values(
                    res_ml["panel"],
                    T_conv,
                    res_ml["panel"].shape[1],
                    kw["scale"],
                    kw.get("scale_type", "linear"),
                    kw.get("backup"),
                    kw.get("wrap_side", "right"),
                    kw.get("use_hatch", True),
                )
                c_cv = qc.compare(
                    ref.index.values.astype(float),
                    ref.values.astype(float),
                    depth,
                    np.asarray(v_conv, dtype=float),
                )
                qc_conv = dict(c_cv["aligned"], shift_ft=float(c_cv["shift_ft"]))

    elapsed_s = time.time() - t0
    return {
        "well": well,
        "scan": Path(scan_path).name,
        "scan_path": str(scan_path),
        "curve": curve,
        "cost_mode": cost_mode,
        "las_out": str(las_out),
        "depth_range_ft": (round(float(depth.min()), 1), round(float(depth.max()), 1)),
        "n_samples": int(len(depth)),
        "mean_conf": round(float(comb_conf.mean()), 3),
        "p10_conf": round(float(np.percentile(comb_conf, 10)), 3),
        "n_review_intervals": len(intervals),
        "review_intervals": intervals,
        "qc_ml": qc_ml,
        "qc_conventional": qc_conv,
        "elapsed_s": round(elapsed_s, 2),
        "depth": depth,
        "value": value,
        "conf": comb_conf,
        "ml_conf": ml_conf,
    }


def digitize_batch(
    scans: list[str | Path | dict[str, Any]],
    profile: str | Path | None = "kansas_kgs",
    out_dir: str | Path = "outputs/batch",
    model_path: str | Path | None = None,
    cost_mode: str = "blend",
    compare_conventional: bool = True,
    max_workers: int = 1,
    review_threshold: float = 0.78,
    data_root: str | Path = "example_data",
) -> list[dict[str, Any]]:
    """Batch digitize a list of scans with a trained U-Net profile and export LAS + summary JSON."""
    resolved_model = _resolve_profile_model(profile, model_path=model_path)
    if cost_mode != "conventional" and resolved_model is not None and resolved_model.is_file():
        ml_models.get_cached_model(resolved_model)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    def _run(item):
        return digitize_one(
            item,
            out_dir=out_root,
            model_path=resolved_model,
            cost_mode=cost_mode,
            compare_conventional=compare_conventional,
            review_threshold=review_threshold,
            data_root=data_root,
        )

    if max_workers and max_workers > 1 and len(scans) > 1:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            results = list(ex.map(_run, scans))
    else:
        results = [_run(s) for s in scans]

    summary = [
        {k: v for k, v in r.items() if k not in ("depth", "value", "conf", "ml_conf")}
        for r in results
    ]
    (out_root / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    return results


def review_queue(
    results: list[dict[str, Any]],
    threshold: float = 0.78,
    min_len: int = 12,
    min_span_ft: float = 10.0,
) -> list[dict[str, Any]]:
    """Build a prioritized review queue of low-confidence intervals across batch results."""
    queue: list[dict[str, Any]] = []
    for r in results:
        if "conf" in r and "depth" in r:
            conf = np.asarray(r["conf"], dtype=float)
            depth = np.asarray(r["depth"], dtype=float)
            spans = ct.low_confidence_intervals(conf, threshold=threshold, min_len=min_len)
            for a, b in spans:
                b_idx = min(b - 1, len(depth) - 1)
                d0, d1 = float(depth[a]), float(depth[b_idx])
                span_ft = abs(d1 - d0)
                if span_ft < min_span_ft:
                    continue
                queue.append(
                    {
                        "well": r["well"],
                        "scan": r.get("scan", ""),
                        "curve": r.get("curve", "GR"),
                        "dp_row_start": int(a),
                        "dp_row_end": int(b),
                        "depth_top_ft": round(min(d0, d1), 1),
                        "depth_base_ft": round(max(d0, d1), 1),
                        "span_ft": round(span_ft, 1),
                        "mean_conf": round(float(conf[a:b].mean()), 3),
                        "min_conf": round(float(conf[a:b].min()), 3),
                    }
                )
        else:
            for iv in r.get("review_intervals", []):
                if iv.get("mean_conf", 1.0) < threshold and iv.get("span_ft", 0.0) >= min_span_ft:
                    queue.append(dict(iv))

    queue.sort(key=lambda x: (x["mean_conf"], -x["span_ft"]))
    return queue
