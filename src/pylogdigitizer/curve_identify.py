# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""curve_identify: one workflow for colour and black-ink scans.

Every curve on the sheet is found automatically, by its ink colour where the print has colour,
by its line style (solid / dashed / dotted / dash-dot, thin / thick) on black ink, or both
(a colour class that holds two clearly different line styles is split). Each curve becomes one
label class, gets its own working image (:func:`working_image`) and is traced the same way.

Names and scales come from the OCR'd header proposals of its track
(:func:`header_ocr.propose_header`): by the colour of the legend line under each printed name
on colour prints, otherwise in header order (largest coverage first); anything matched by
order only is flagged for review.
"""
from __future__ import annotations

import numpy as np

__all__ = ["identify_curves", "working_image", "trace_preset", "curve_mask", "track_ranges", "STYLE_COLORS"]

# drawing colours for style classes (colour classes are drawn in their own ink)
STYLE_COLORS = ("#1f4e9c", "#d9480f", "#2b8a3e", "#9c36b5", "#c2255c", "#0b7285")


def track_ranges(L) -> dict:
    """{track name: (x_left, x_right)} from the followed borders of a layout (median x)."""
    out = {}
    for n in L.track_names:
        li, ri = L.edge_names.index(n + ".left"), L.edge_names.index(n + ".right")
        out[n] = (float(np.median(L.edges[:, li])), float(np.median(L.edges[:, ri])))
    return out


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb[:3])


def _coverage(mask: np.ndarray) -> float:
    return float(mask.any(1).mean()) if mask.size else 0.0


def _rows_by_track(header) -> dict:
    out = {}
    for c in (header or {}).get("curves", []):
        out.setdefault(c["track"], []).append(c)
    for v in out.values():
        v.sort(key=lambda c: float((c.get("evidence") or {}).get("name_y") or 0.0))
    return out


def _meta_from_row(c) -> dict:
    v = lambda k: c[k]["value"]
    bk = v("backup")
    if bk and tuple(bk) == (v("scale_left"), v("scale_right")):     # a repeated scale is not a backup
        bk = None
    return dict(name=v("mnemonic"), unit=v("unit") or "", left=v("scale_left"), right=v("scale_right"),
                type=v("scale_type") or "linear", backup=tuple(bk) if bk else None, header=c.get("label", ""))


def identify_curves(scan, L, header=None, rgb=None, palette=None, min_coverage: float = 0.25,
                    split_styles: bool = True) -> dict:
    """All curves of the sheet: dict(kind, labels, names, base, body_rows, tracks, palette, curves).

    ``L``: layout of the scan (:func:`workflows.analyze_layout`). ``header``: header proposals.
    ``rgb``: the colour image (None = black-ink scan). ``palette``: confirmed palette (default:
    :func:`scan_color.suggest_palette`). Each curve: dict(id, track, cls, kind ('color' | 'style'
    | 'color+style'), pattern, color (hex, for drawing), coverage, name, unit, left, right, type,
    backup, header, review (list of what was guessed)).
    """
    from pylogdigitizer import line_styles as ls
    tracks = track_ranges(L)
    rows = _rows_by_track(header)
    curves = []
    if rgb is not None:
        from pylogdigitizer.scan_color import suggest_palette, classify_colors, find_log_body, legend_color
        from pylogdigitizer.curve_ink import make_base_ink, black_curve_only, _grey_names
        pal = dict(palette or suggest_palette(rgb))
        labels, names = classify_colors(rgb, pal)
        names = list(names)
        body = tuple(int(v) for v in find_log_body(labels, names))
        base = make_base_ink(labels, names, body)
        grey = set(_grey_names(names))
        r0, r1 = body
        for t, (xl, xr) in tracks.items():
            x0, x1 = int(round(xl)) + 4, int(round(xr)) - 4
            sub = labels[r0:r1, x0:x1]
            legend, n_legend = {}, {}
            for c in rows.get(t, []):
                y = float((c.get("evidence") or {}).get("name_y") or 0.0)
                lc = legend_color(labels, names, y - 10, y + 10, (xl, xr))
                legend.setdefault(lc, c)
                n_legend[lc] = n_legend.get(lc, 0) + 1
            found = []
            for i, n in enumerate(names):
                if n in ("white",) or n in grey:
                    continue
                m = black_curve_only(sub, names) if n == "black" else sub == i
                cov = _coverage(m)
                if (n == "black" and n not in legend) or (cov < min_coverage and n not in legend):
                    continue
                parts = [(n, None, m, cov)]
                if split_styles and n != "black" and n_legend.get(n, 0) != 1:   # one legend row = one curve
                    idx, st = ls.track_styles(m, min_coverage=0.4)
                    pats = {s["pattern"] for s in st}
                    if len(st) >= 2 and len(pats) >= 2:
                        parts = [(f"{n} {s['name']}", s["pattern"], idx == k + 1, s["coverage"]) for k, s in enumerate(st)]
                found.extend(parts)
            for cls, pat, m, cov in found:
                col = cls.split(" ")[0]
                if cls not in names:                     # colour + style split: its own class
                    names.append(cls)
                    lab_sub = labels[r0:r1, x0:x1]
                    lab_sub[m] = names.index(cls)
                meta = _meta_from_row(legend[col]) if col in legend and pat is None else None
                review = [] if meta else ["name", "scale"]
                curves.append(dict(track=t, cls=cls, kind="color" if pat is None else "color+style", pattern=pat or "",
                                   color="#000000" if col == "black" else _hex(pal[col]), coverage=round(cov, 3),
                                   review=review, **(meta or {})))
        out = dict(kind="color", labels=labels, names=names, base=base, body_rows=body, tracks=tracks,
                   palette={k: tuple(int(x) for x in v) for k, v in pal.items()})
    else:
        y_hdr = [w["y1"] for t in ((header or {}).get("tracks") or {}).values() for w in t.get("words", [])]
        body = (int(max([L.probe_row or 0] + [y + 10 for y in y_hdr])), int(L.ink.shape[0]))
        sug = ls.suggest_line_styles(L.ink, tracks, body, min_coverage=min_coverage)
        base = ls.make_style_base(L.ink, sug)
        k = 0
        for t, styles in sug["styles"].items():
            hdr = list(rows.get(t, []))
            for j, s in enumerate(styles):            # styles come sorted by coverage
                meta = _meta_from_row(hdr[j]) if j < len(hdr) else None
                curves.append(dict(track=t, cls=s["name"], kind="style", pattern=s["pattern"],
                                   color=STYLE_COLORS[k % len(STYLE_COLORS)], coverage=round(float(s["coverage"]), 3),
                                   review=["name (header order)"] if meta else ["name", "scale"], **(meta or {})))
                k += 1
        out = dict(kind="style", labels=sug["labels"], names=list(sug["names"]), base=base, body_rows=body,
                   tracks=tracks, palette=None, styles=sug["styles"])
    used = set()
    for i, c in enumerate(curves):
        if c.get("name") and (c.get("left") is None or c.get("right") is None):   # named, scale not read
            c.update(left=0.0, right=1.0, type=c.get("type") or "linear")
            c["review"] = sorted(set(c["review"]) | {"scale"})
        if not c.get("name"):
            c.update(name=f"{c['track'].upper()}_{c['cls'].replace(' ', '_').upper()}", unit="TRACK", left=0.0, right=1.0,
                     type="linear", backup=None, header="")
        nm, n = c["name"], 2
        while c["name"] in used:
            c["name"] = f"{nm}_{n}"; n += 1
        used.add(c["name"])
        c["id"] = f"c{i + 1}"
    out["curves"] = curves
    return out


def working_image(ID: dict, curve: dict) -> np.ndarray:
    """Working image of one identified curve: the base (grid, borders, depth numbers) plus this
    curve's ink inside its track, gaps closed (by line pattern for style classes) and thickened."""
    from pylogdigitizer.curve_ink import make_curve_ink
    from pylogdigitizer.line_styles import GAP_CLOSE_PX
    kw = {}
    pat = curve.get("pattern") or ""
    if pat:
        kw = dict(gap_close_px=GAP_CLOSE_PX.get(pat, 9), thicken_px=2)
    return make_curve_ink(ID["base"], ID["labels"], ID["names"], curve["cls"], ID["tracks"][curve["track"]],
                          ID["body_rows"], **kw)


def trace_preset(ID: dict) -> dict:
    """Tracer preset of the scan kind (colour-scan preset, or the black-ink line-style preset)."""
    if ID["kind"] == "color":
        from pylogdigitizer.scan_color import COLOR_TRACE_PRESET
        return dict(COLOR_TRACE_PRESET)
    from pylogdigitizer.line_styles import LINE_TRACE_PRESET
    return dict(LINE_TRACE_PRESET)


def curve_mask(ID: dict, curve: dict) -> np.ndarray:
    """Sheet-size bool mask of the curve's own ink (no gap closing): the highlight layer."""
    m = np.zeros(ID["labels"].shape[:2], bool)
    xl, xr = ID["tracks"][curve["track"]]
    r0, r1 = ID["body_rows"]
    x0, x1 = int(round(xl)) + 4, int(round(xr)) - 4
    if curve["cls"] == "black":
        from pylogdigitizer.curve_ink import black_curve_only
        m[r0:r1, x0:x1] = black_curve_only(ID["labels"][r0:r1, x0:x1], ID["names"])
    else:
        m[r0:r1, x0:x1] = ID["labels"][r0:r1, x0:x1] == ID["names"].index(curve["cls"])
    return m
