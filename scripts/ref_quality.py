"""Reference-quality check for the paired KGS LAS curves (no scan, no tracing involved).

Some KGS LAS files were machine-digitized and show stair steps: long constant-value runs, often
sitting on a printed grid-line value. Such references cap any tracer's score. For each run this
reports flat_frac (samples in constant runs >= 3 ft), linear_frac (samples in straight sloped runs
>= 3 ft, i.e. interpolated stair segments), stair_score = their sum, and suspect = score >= 0.15.
The flag uses only the reference itself, never digitized output, and is not used to tune anything.

    python scripts/ref_quality.py --out ref_quality.csv
"""
import argparse, csv, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RUN_FT, SUSPECT = 3.0, 0.15


def run_frac(mask, min_len):
    """Share of samples inside True-runs of at least min_len samples."""
    out = c = 0
    for x in list(mask) + [False]:
        if x:
            c += 1
        else:
            out += c if c >= min_len else 0
            c = 0
    return out / max(1, len(mask))


def quality(values, scale, stype="linear", step_ft=0.5):
    """flat = constant runs >= 3 ft; linear = straight sloped runs >= 3 ft (interpolated/stair
    segments of a machine digitization). Both relative to 0.1% of the print span."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if len(v) < 20:
        return dict(n=len(v), flat_frac=np.nan, linear_frac=np.nan, stair_score=np.nan, suspect=False)
    if stype == "log":
        v = np.log10(np.clip(v, 1e-3, None)); span = np.log10(max(scale) / min(scale))
    else:
        span = abs(scale[1] - scale[0])
    L = max(3, int(round(RUN_FT / step_ft)))
    d = np.abs(np.diff(v)); tol = 1e-3 * span
    flat = run_frac(d < tol, L)
    lin = run_frac((np.abs(np.diff(v, 2)) < tol) & (d[1:] >= tol), L)
    score = flat + lin
    return dict(n=len(v), flat_frac=round(flat, 3), linear_frac=round(lin, 3), stair_score=round(score, 3),
                suspect=bool(score >= SUSPECT))


def main(argv=None):
    from run_regression import jobs_from_config
    from pylogdigitizer.las_export import read_las
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "dataset_profiles" / "kansas_kgs_wells.yaml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for j in jobs_from_config(pathlib.Path(a.config)):
        ref = read_las(j["las_path"]).df()[j["curve"]].dropna(); ref = ref[ref > -999]
        step = float(np.median(np.diff(ref.index.values))) if len(ref) > 1 else 0.5
        q = quality(ref.values, j["scale"], j.get("scale_type", "linear"), step)
        rows.append(dict(well=j["well"], scan=j["scan"], curve=j["curve"], **q))
        print(f'{j["well"]:18s} {j["scan"]:11s} {j["curve"]:5s} n={q["n"]:6d} flat={q["flat_frac"]} linear={q["linear_frac"]} stair={q["stair_score"]} {"SUSPECT" if q["suspect"] else ""}')
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
