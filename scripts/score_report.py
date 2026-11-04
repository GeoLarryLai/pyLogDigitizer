#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Shape-score report over run_regression CSVs: registered (gain/offset-fitted) r per run, split
into dev / held-out test, with and without suspect (machine-digitized) references.

Honesty rules: runs whose scale came from the LAS (scale_source las_fallback) are kept out of the
scores and listed separately; suspect references (scripts/ref_quality.py) are reported both ways.
The test wells are never used to choose settings.

    python scripts/score_report.py before.csv [after.csv ...] --names base new --out report.md
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
TEST_WELLS = ("USA_A_16", "GOHEEN_UNIT_1", "Hall_D_4", "Minzer_1")   # whole wells: no LAS shared with dev


def split_of(well: str, scan: str) -> str:
    return "test" if (well in TEST_WELLS or f"{well}:{scan}" in TEST_WELLS) else "dev"


def suspect_table(cfg_path) -> dict:
    from ref_quality import quality
    from run_regression import jobs_from_config
    from pylogdigitizer.las_export import read_las
    out, cache = {}, {}
    for j in jobs_from_config(pathlib.Path(cfg_path)):
        if j["las_path"] not in cache:
            cache[j["las_path"]] = read_las(j["las_path"]).df()
        df = cache[j["las_path"]]
        if j["curve"] not in df:
            continue
        v = df[j["curve"]].dropna(); v = v[v > -999]
        out[(j["well"], j["scan"], j["curve"])] = bool(quality(v.values, j["scale"], j.get("scale_type", "linear"))["suspect"])
    return out


def summarize(df, sus) -> dict:
    import pandas as pd
    df = df.copy()
    df["split"] = [split_of(w, s) for w, s in zip(df.well, df.scan)]
    df["suspect"] = [sus.get((w, s, c), False) for w, s, c in zip(df.well, df.scan, df.curve)]
    df["r_reg"] = pd.to_numeric(df.r_reg, errors="coerce")
    fb = df[df.scale_source.astype(str) == "las_fallback"]
    hon = df[df.scale_source.astype(str) != "las_fallback"]
    res = {}
    for sp in ("dev", "test"):
        d = hon[hon.split == sp]
        r_all = d.r_reg.fillna(0.0)                       # error/no-score counts as 0
        r_cln = d[~d.suspect].r_reg.fillna(0.0)
        res[sp] = dict(n=len(d), mean_all=float(r_all.mean()) if len(d) else np.nan,
                       n_clean=int((~d.suspect).sum()), mean_clean=float(r_cln.mean()) if len(r_cln) else np.nan,
                       median_clean=float(r_cln.median()) if len(r_cln) else np.nan)
    res["fallback"] = fb[["well", "scan", "curve", "r_reg"]].to_dict("records")
    return res, df


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+"); ap.add_argument("--names", nargs="*")
    ap.add_argument("--config", default=str(ROOT / "dataset_profiles" / "kansas_kgs_wells.yaml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    names = a.names or [pathlib.Path(c).stem for c in a.csvs]
    sus = suspect_table(a.config)
    lines = ["| set | split | runs | mean r_reg (all refs) | clean refs | mean r_reg (clean) | median (clean) |",
             "|---|---|---|---|---|---|---|"]
    per = {}
    for nm, c in zip(names, a.csvs):
        res, d = summarize(pd.read_csv(c), sus)
        per[nm] = d
        for sp in ("dev", "test"):
            x = res[sp]
            lines.append(f"| {nm} | {sp} | {x['n']} | {x['mean_all']:.3f} | {x['n_clean']} | {x['mean_clean']:.3f} | {x['median_clean']:.3f} |")
        if res["fallback"]:
            lines.append(f"| {nm} | las_fallback (not scored) | {len(res['fallback'])} | " +
                         ", ".join(f"{r['well']} {r['curve']} {r['r_reg']:.3f}" for r in res["fallback"]) + " | | | |")
    lines += ["", "| well | scan | curve | split | suspect ref | scale_source | " + " | ".join(f"r_reg {n}" for n in names) + " |",
              "|---" * (6 + len(names)) + "|"]
    base = per[names[0]]
    for _, r in base.iterrows():
        vals = []
        for n in names:
            m = per[n][(per[n].well == r.well) & (per[n].scan == r.scan) & (per[n].curve == r.curve)]
            vals.append(f"{m.r_reg.iloc[0]:.3f}" if len(m) and np.isfinite(m.r_reg.iloc[0]) else "-")
        lines.append(f"| {r.well} | {r.scan} | {r.curve} | {r.split} | {'yes' if r.suspect else ''} | {r.scale_source} | " + " | ".join(vals) + " |")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        pathlib.Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
