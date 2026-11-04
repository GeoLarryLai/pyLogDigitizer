#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Error budget per KGS run: where the gap between the automatic digitization and the LAS is lost.

Diagnosis only - the LAS is read here to *measure* losses, never to set anything in the pipeline.

For runs with a guided reference trace (training_data/corrections, latest per well/curve):
  r_guided   guided trace (Larry's pins) with its own calibration vs LAS (registered r): the ceiling
             for this print with perfect tracing; 1 - r_guided = print/LAS mismatch the tracer cannot fix
  r_auto_gc  0-click trace of the same panel, valued with the guided calibration (same depth, scale
             and wrap rows): tracing alone
  r_fixwrong r_auto_gc with the rows on another curve (> 10 px off) replaced by the guided ones
  acc2/wrong pixel agreement with the guided trace (% rows within 2 px / beyond 10 px)
  r_pipe/r_pipe_reg  fully automatic run (run_regression CSV): r with a bulk shift, and registered
Buckets (in registered r):
  (c) wrong curve  = r_fixwrong - r_auto_gc
  (d) pixel        = r_guided - r_fixwrong
  (a) depth        = r_pipe_reg - r_pipe   (what local depth registration recovers)
  (b) calibration rest (scale ends / wraps / section / panel) = r_auto_gc - r_pipe_reg
      (pipeline vs guided calibration at equal tracing quality; negative = pipeline better)

    python scripts/error_budget.py --pipeline outputs/qc/regression_metrics.csv --out budget.csv --qc qc_dir
"""
from __future__ import annotations
import argparse, csv, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def las_for(cfg, well, scan, curve):
    for w in cfg["wells"]:
        if w["name"] != well:
            continue
        for r in w["runs"]:
            if r["scan"] == scan and r["curve"] == curve:
                return ROOT / "example_data" / w["dir"] / w["las"], r
    return None, None


def main(argv=None):
    import yaml, pandas as pd
    from PIL import Image
    from pylogdigitizer import curve_tracing as ct, qc_compare as qc
    from pylogdigitizer.las_export import read_las
    sys.path.insert(0, str(ROOT / "scripts"))
    from bench_trace import latest
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, help="run_regression CSV of the automatic pipeline")
    ap.add_argument("--config", default=str(ROOT / "dataset_profiles" / "kansas_kgs_wells.yaml"))
    ap.add_argument("--root", default=str(ROOT / "training_data" / "corrections"))
    ap.add_argument("--trace", default="{}", help="JSON trace_track overrides for the 0-click trace")
    ap.add_argument("--out", required=True); ap.add_argument("--qc", default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config)); kw = json.loads(a.trace)
    pipe = pd.read_csv(a.pipeline)
    refs = {}
    for d in latest([p for p in pathlib.Path(a.root).iterdir() if (p / "trace.csv").exists()]):
        m = json.loads((d / "metadata.json").read_text()); refs[(m["well"], m["scan"], m["curve"])] = d
    rows = []
    for _, p in pipe.iterrows():
        row = dict(well=p.well, scan=p.scan, curve=p.curve, status=p.status, r_pipe=p.r, r_pipe_reg=p.r_reg)
        row["a_depth"] = p.r_reg - p.r if np.isfinite(p.r) and np.isfinite(p.r_reg) else np.nan
        d = refs.get((p.well, p.scan, str(p.curve).split(":")[0]))
        las, run = las_for(cfg, p.well, p.scan, str(p.curve))
        if d is not None and las is not None:
            cal = json.loads((d / "calibration.json").read_text()); sc = cal["scale"]
            g = np.genfromtxt(d / "trace.csv", delimiter=",", names=True)
            pan = np.asarray(Image.open(d / "image.png")) < 128; W = pan.shape[1]
            T = ct.trace_track(pan, 0, px_per_ft=cal.get("px_per_ft"), **kw)
            xa = np.interp(g["panel_row"], np.asarray(T["rows"], float), np.asarray(T["x"], float))
            log = sc.get("type") == "log"
            def val(x, lo, hi):
                f = x / W
                return 10 ** (np.log10(lo) + f * (np.log10(hi) - np.log10(lo))) if log else lo + f * (hi - lo)
            prim = val(g["x_panel"], sc["left"], sc["right"])
            bk = sc.get("backup")
            use_bk = np.zeros(len(g), bool)
            if bk:
                use_bk = np.abs(val(g["x_panel"], *bk) - g["value"]) < np.abs(prim - g["value"])
            va = np.where(use_bk, val(xa, *bk) if bk else 0, val(xa, sc["left"], sc["right"]))
            err = np.abs(xa - g["x_panel"]); wrong = err > 10
            vfix = np.where(wrong, g["value"], va)
            df = read_las(las).df(); ref = df[str(p.curve).split(":")[0]].dropna(); ref = ref[ref > -999]
            rv = ref.values.astype(float)
            tf = (lambda v: np.log10(np.clip(v, 1e-3, None))) if log else (lambda v: v)
            def R(v):
                c = qc.compare(ref.index.values, tf(rv), g["depth_ft"], tf(np.asarray(v, float)))
                return c["registered"]["r"], c
            rg, cg = R(g["value"]); ra, ca = R(va); rfx, _ = R(vfix)
            row.update(ref=d.name, acc2=100 * float((err <= 2).mean()), wrong=100 * float(wrong.mean()),
                       r_guided=rg, r_auto_gc=ra, r_fixwrong=rfx, c_wrong_curve=rfx - ra, d_pixel=rg - rfx,
                       b_calib=ra - p.r_reg, ceiling_gap=1 - rg)
            if a.qc:
                from matplotlib.figure import Figure
                fig = Figure(figsize=(7, 10)); ax = fig.subplots(1, 2, sharey=True)
                for k, (lab, v, col) in enumerate([("guided", cg, "tab:green"), ("auto (guided cal.)", ca, "tab:red")]):
                    ax[k].plot(v["ref"], v["grid"], color="k", lw=.5, label="LAS")
                    ax[k].plot(v["test_registered"], v["grid"], color=col, lw=.5, label=lab)
                    ax[k].set_title(f"{lab}: r_reg={v['registered']['r']:.3f}", fontsize=9); ax[k].legend(fontsize=7)
                ax[0].invert_yaxis(); fig.suptitle(f"{p.well} {p.curve}  acc2={row['acc2']:.1f}% wrong={row['wrong']:.1f}%", fontsize=10)
                pathlib.Path(a.qc).mkdir(parents=True, exist_ok=True)
                fig.savefig(pathlib.Path(a.qc) / f"budget_{p.well}_{p.scan}_{str(p.curve).replace(':', '_')}.png", dpi=90)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
