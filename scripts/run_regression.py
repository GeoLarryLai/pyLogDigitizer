#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Regression harness: run the conventional pipeline with DEFAULTS on every paired well/curve in
dataset_profiles/kansas_kgs_wells.yaml and compare with the paired KGS LAS.

    python scripts/run_regression.py                 # all runs
    python scripts/run_regression.py --only GR DT    # curve filter
    python scripts/run_regression.py --wells MOBIL_12-1 --no-ocr

Writes outputs/qc/regression_metrics.csv (one row per run, with status) and a QC overlay per run
in outputs/qc/regression/. Each run is a separate process (crash/timeout/memory isolation).
Status: ok (r >= 0.8), poor (0.5 <= r < 0.8), fail (r < 0.5 or < 200 samples), error (exception;
message says which stage). Log-scale curves are compared as log10. The reference LAS is used
only here, after digitization - never as an input.
"""
from __future__ import annotations
import argparse, csv, multiprocessing as mp, pathlib, sys, time, traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIELDS = ["well", "scan", "curve", "status", "r", "rmse", "bias", "mae", "n", "shift_ft", "depth_top", "depth_base",
          "ref_top", "ref_base", "px_per_ft", "depth_source", "label_mode", "n_labels", "loo_max_ft", "n_wraps",
          "scale_source", "seconds", "message"]


def run_one(job: dict, use_ocr: bool, q):
    import numpy as np
    from pylogdigitizer import workflows as wf, qc_compare as qc
    from pylogdigitizer.las_export import read_las
    row = {k: "" for k in FIELDS}
    row.update(well=job["well"], scan=job["scan"], curve=job["curve"], scale_source=job.get("scale_source", ""))
    t0 = time.time()
    try:
        kw = {k: job[k] for k in ("track", "backup", "wrap_side", "use_hatch", "scale_type", "first_depth", "grid_rectify") if k in job}
        kw.update(job.get("trace", {}) or {})
        df = read_las(job["las_path"]).df()
        ref = df[job["curve"]].dropna()
        ref = ref[ref > -999]
        choice = ""
        try:
            R = wf.digitize_curve(job["scan_path"], scale=tuple(job["scale"]), use_ocr=use_ocr, **kw)
        except wf.SectionChoiceNeeded as e:
            # Several log sections and nothing on the sheet says which: a user must choose. Benchmark
            # stand-in for that choice: the section whose depth interval best overlaps the reference LAS interval
            # (the LAS was made from that print). Reported as "needed a user choice".
            rt, rb = float(ref.index.min()), float(ref.index.max())
            iou = lambda c: max(0.0, min(c["base"], rb) - max(c["top"], rt)) / (max(c["base"], rb) - min(c["top"], rt))
            best = max(e.sections, key=iou)
            choice = f"needed a user choice: section={best['index']} ({best['top']:.0f}-{best['base']:.0f} ft) of " + \
                     ", ".join(f"{c['top']:.0f}-{c['base']:.0f}" for c in e.sections) + ";"
            R = wf.digitize_curve(job["scan_path"], scale=tuple(job["scale"]), use_ocr=use_ocr, section=best["index"], **kw)
        rv, tv = ref.values.astype(float), np.asarray(R["value"], float)
        if job.get("scale_type") == "log":
            rv = np.log10(np.clip(rv, 1e-3, None)); tv = np.log10(np.clip(tv, 1e-3, None))
        c = qc.compare(ref.index.values, rv, R["depth"], tv)
        m, ck = c["aligned"], R["checks"]
        row.update(r=m["r"], rmse=m["rmse"], bias=m["bias"], mae=m["mae"], n=m["n"], shift_ft=c["shift_ft"],
                   depth_top=ck.get("depth_top"), depth_base=ck.get("depth_base"),
                   ref_top=float(ref.index.min()), ref_base=float(ref.index.max()), px_per_ft=ck.get("px_per_ft"),
                   depth_source=ck.get("source"), label_mode=ck.get("label_mode"), n_labels=ck.get("n_labels"),
                   loo_max_ft=ck.get("loo_max_ft"), n_wraps=ck.get("n_wraps"), message=choice + m.get("reason", ""))
        if "ocr_disagrees_with_first_depth" in ck:
            row["message"] += f" OCR first label {ck['ocr_disagrees_with_first_depth'][0]} != config {ck['ocr_disagrees_with_first_depth'][1]}"
        r = m["r"]
        row["status"] = ("fail" if not np.isfinite(r) or m["n"] < 200 or r < 0.5 else "poor" if r < 0.8 else "ok")
        if job.get("save_dir"):     # digitized trace for later analysis (e.g. reference-LAS QC masking)
            sd = pathlib.Path(job["save_dir"]); sd.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sd / f"{job['well']}_{pathlib.Path(job['scan']).stem}_{job['curve'].replace(':', '_')}.npz",
                                depth=np.asarray(R["depth"], float), value=tv, conf=np.asarray(R["conf"], float),
                                section=ck.get("section_used", -1))
        if len(c["grid"]):
            out = ROOT / "outputs" / "qc" / "regression"; out.mkdir(parents=True, exist_ok=True)
            qc.qc_plot(c["grid"], c["ref"], c["test_aligned"],
                       str(out / f"{job['well']}_{pathlib.Path(job['scan']).stem}_{job['curve'].replace(':', '_')}.png"),
                       f"{job['well']} {job['scan']} {job['curve']} ({row['status']})")
    except Exception as e:  # report, never hide
        row.update(status="error", message=f"{type(e).__name__}: {e}"[:300])
        traceback.print_exc()
    row["seconds"] = round(time.time() - t0, 1)
    q.put(row)


def jobs_from_config(cfg_path: pathlib.Path, wells=None, curves=None):
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    data = ROOT / cfg.get("data_dir", "example_data")
    out = []
    for w in cfg["wells"]:
        if wells and w["name"] not in wells:
            continue
        for r in w["runs"]:
            if curves and r["curve"].split(":")[0] not in curves:
                continue
            j = dict(cfg.get("defaults", {})); j.update(r)
            j.update(well=w["name"], scan_path=str(data / w["dir"] / r["scan"]), las_path=str(data / w["dir"] / w["las"]))
            out.append(j)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default=None, help="JSON dict of trace_track overrides applied to every job")
    ap.add_argument("--config", default=str(ROOT / "dataset_profiles" / "kansas_kgs_wells.yaml"))
    ap.add_argument("--wells", nargs="*"); ap.add_argument("--only", nargs="*", help="curves, e.g. GR DT ILD")
    ap.add_argument("--no-ocr", action="store_true"); ap.add_argument("--ocr-engine", default="auto", help="auto | rapidocr | none")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "qc" / "regression_metrics.csv"))
    ap.add_argument("--grid-rectify", action="store_true", help="straighten grid lines inside the track before tracing")
    ap.add_argument("--save-traces", default=None, help="directory: save each digitized trace as .npz (depth, value)")
    a = ap.parse_args(argv)
    jobs = jobs_from_config(pathlib.Path(a.config), a.wells, a.only)
    for j in jobs:
        j["save_dir"] = a.save_traces
        if a.grid_rectify:
            j["grid_rectify"] = True
    if a.trace:
        import json
        for j in jobs:
            j["trace"] = {**(j.get("trace") or {}), **json.loads(a.trace)}
    rows = []
    ctx = mp.get_context("spawn")
    for j in jobs:
        q = ctx.Queue(); p = ctx.Process(target=run_one, args=(j, False if a.no_ocr else (a.ocr_engine if a.ocr_engine != "auto" else True), q)); t = time.time()
        p.start(); p.join(a.timeout)
        if p.is_alive():
            p.kill(); p.join()
            row = {k: "" for k in FIELDS}; row.update(well=j["well"], scan=j["scan"], curve=j["curve"], status="error",
                                                     message=f"timeout after {a.timeout}s", seconds=a.timeout)
        elif not q.empty():
            row = q.get()
        else:
            row = {k: "" for k in FIELDS}; row.update(well=j["well"], scan=j["scan"], curve=j["curve"], status="error",
                                                     message=f"process died (exit {p.exitcode}; out of memory?)",
                                                     seconds=round(time.time() - t, 1))
        rows.append(row)
        def f(v):
            return f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"{row['well']:18s} {row['scan']:11s} {row['curve']:5s} {row['status']:6s} r={f(row['r'])} n={row['n']} "
              f"{row['seconds']}s {row['message'][:120]}", flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{ok}/{len(rows)} runs ok -> {a.out}")
    return rows


if __name__ == "__main__":
    main()
