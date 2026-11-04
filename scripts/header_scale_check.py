"""Automatic scale reading from the scan header (header OCR) with a confidence flag.

For each configured run, OCR the track header (header_ocr.propose_header), take the proposal for
the run's track whose mnemonic matches the curve, and report scale/backup with a confidence:
  high = proposal found, no field flagged needs_review, mnemonic matched, backup joins the primary
  low  = anything else -> MANUAL FIX (read the print header, set scale_source: header). Never a
         silent guess. The LAS is never read here; the config value is shown only for comparison.

    python scripts/header_scale_check.py --wells Chitwood_12 ELY_1 --out header_scales.csv
"""
import argparse, csv, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def pick(curves, track, mnem):
    c = [p for p in curves if p.get("track") == track]
    m = [p for p in c if (p["mnemonic"]["value"] or "").upper().startswith(mnem)]
    return (m or c or [None])[0], bool(m)


def confidence(p, matched):
    if p is None or p["scale_left"]["value"] is None:
        return "low", "no scale proposal"
    why = [k for k in ("scale_left", "scale_right", "scale_type", "backup") if p[k]["needs_review"]]
    if not matched:
        why.append("mnemonic not matched")
    b = p["backup"]["value"]
    if b is not None:
        l, r = p["scale_left"]["value"], p["scale_right"]["value"]
        if min(abs(b[0] - r), abs(b[1] - l)) > 1e-6 * max(1, abs(r - l)):
            why.append("backup does not join primary")
    return ("low" if why else "high"), ";".join(why)


def main(argv=None):
    from run_regression import jobs_from_config
    from pylogdigitizer import header_ocr
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "dataset_profiles" / "kansas_kgs_wells.yaml"))
    ap.add_argument("--wells", nargs="*"); ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows, cache = [], {}
    for j in jobs_from_config(pathlib.Path(a.config), a.wells, a.only):
        if j["scan_path"] not in cache:
            try:
                cache[j["scan_path"]] = header_ocr.propose_header(j["scan_path"])
            except Exception as e:      # noqa: BLE001 - reported as low confidence
                cache[j["scan_path"]] = {"curves": [], "error": f"{type(e).__name__}: {e}"}
        H = cache[j["scan_path"]]
        p, matched = pick(H["curves"], j.get("track", "track1"), j["curve"].split(":")[0].upper())
        conf, why = confidence(p, matched)
        if H.get("error"):
            why = H["error"][:120]
        ocr = None if p is None else [p["scale_left"]["value"], p["scale_right"]["value"]]
        bk = None if p is None else p["backup"]["value"]
        agrees = bool(ocr and None not in ocr and [float(x) for x in ocr] == [float(x) for x in j["scale"]])
        r = dict(well=j["well"], scan=j["scan"], curve=j["curve"], ocr_scale=json.dumps(ocr), ocr_backup=json.dumps(bk),
                 confidence=conf, reason=why, config_scale=json.dumps(j["scale"]), config_backup=json.dumps(j.get("backup")),
                 config_source=j.get("scale_source", ""), agrees=agrees)
        rows.append(r)
        flag = "" if conf == "high" else "  -> MANUAL FIX"
        print(f'{r["well"]:18s} {r["scan"]:11s} {r["curve"]:5s} ocr={ocr} backup={bk} conf={conf} {why} | config={j["scale"]} {j.get("backup")} ({r["config_source"]}){flag}', flush=True)
    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
