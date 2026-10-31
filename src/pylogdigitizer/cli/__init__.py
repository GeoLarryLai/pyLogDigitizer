# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""cli: command-line entry point `pylogdigitizer`."""
from __future__ import annotations
import argparse, json

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pylogdigitizer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="scan or LAS info"); s.add_argument("path")
    s = sub.add_parser("preview", help="downsampled PNG of a scan"); s.add_argument("path"); s.add_argument("out"); s.add_argument("--factor", type=int, default=8)
    for nm in ("editor", "serve"):   # 'serve' kept as an alias
        s = sub.add_parser(nm, help="launch the local web editor on http://127.0.0.1:PORT")
        s.add_argument("project", nargs="?", default=".", help="project folder (with example_data/, dataset_profiles/)")
        s.add_argument("--port", type=int, default=8765); s.add_argument("--open", action="store_true", help="open a browser tab")
    sub.add_parser("backend", help="report tracer backend")
    a = ap.parse_args(argv)
    if a.cmd == "info":
        if a.path.lower().endswith(".las"):
            from pylogdigitizer.las_export import curves_summary
            print(json.dumps(curves_summary(a.path), indent=2))
        else:
            from pylogdigitizer.scan_io import read_info
            print(read_info(a.path))
    elif a.cmd == "preview":
        from pylogdigitizer.scan_io import load_scan, downsample, save_png
        print(save_png(downsample(load_scan(a.path), a.factor), a.out))
    elif a.cmd in ("editor", "serve"):
        from pylogdigitizer.web_editor import serve
        serve(a.project, a.port, a.open)
    elif a.cmd == "backend":
        from pylogdigitizer.curve_tracing import backend
        print(backend())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
