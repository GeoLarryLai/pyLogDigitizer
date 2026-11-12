# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Line-by-line header parser for colour-log prints (name, then 'left UNIT right').

Unlike :func:`pylogdigitizer.header_ocr.propose_curves`, this path does **not**
collapse distinct printed names onto a CWLS mnemonic (so RES(16N), RES(64N) and
LATERAL stay separate) and does not merge two name lines that share a scale.
It is an explicit option (``propose_header(..., parse="lines")``); the default
proposal path is unchanged.

Printed names are sanitised to LAS-safe tokens (letters, digits, underscore)
and kept as-is after a small unit/name spelling table (GAMMA -> GR, PORNEU ->
NPHI, APIGR -> GAPI). Words are used only when their box centre sits inside
the track, so a number printed on a shared border does not leak into the next
track. A scale end missed by OCR is copied from a sibling curve of the same
unit, or a porosity line without its right end is assumed to end at 0, and
the ``check`` note says so.
"""
from __future__ import annotations

import re

import numpy as np

from pylogdigitizer.curve_pairs import unit_key as _ukey

__all__ = ["printed_name", "propose_curves_from_lines", "apply_curve_fixes", "merge_backup_rows"]

_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
_UNIT_FIX = {"APIGR": "GAPI", "API": "GAPI", "PERCENT": "%", "OHMM": "OHMM", "OHM": "OHM"}
_NAME_FIX = {"GAMMA": "GR", "PORNEU": "NPHI"}
_UNITS = {"MV", "APIGR", "GAPI", "API", "APIN", "OHMM", "OHM", "PERCENT", "PU", "%", "CPS",
          "USFT", "USF", "GCC", "GCM", "IN", "FT"}


def printed_name(name: str) -> str:
    """LAS-safe printed curve name: 'RES(16N)' -> 'RES16N'. Not forced to a CWLS mnemonic."""
    m = re.sub(r"[^A-Za-z0-9_]", "", str(name)).upper()
    return _NAME_FIX.get(m, m) or "CURVE"


def _num(tok: str):
    t = tok.replace("−", "-").replace("–", "-").replace(",", ".").strip()
    t = re.sub(r"^[^\w\-.]+|[^\w.]+$", "", t)
    return float(t) if _NUM.match(t) else None


def _header_lines(words: list) -> list:
    h = np.median([w["y1"] - w["y0"] for w in words]) if words else 10
    lines = []
    for w in sorted(words, key=lambda w: (w["y0"] + w["y1"]) / 2):
        yc = (w["y0"] + w["y1"]) / 2
        if lines and abs(yc - lines[-1]["yc"]) <= 0.6 * h:
            lines[-1]["w"].append(w)
            lines[-1]["yc"] = float(np.mean([(q["y0"] + q["y1"]) / 2 for q in lines[-1]["w"]]))
        else:
            lines.append(dict(yc=yc, w=[w]))
    for ln in lines:
        ln["w"].sort(key=lambda w: w["x0"])
        ln["tok"] = [str(w["text"]).strip() for w in ln["w"] if str(w["text"]).strip()]
        ln["y0"] = min(w["y0"] for w in ln["w"]); ln["y1"] = max(w["y1"] for w in ln["w"])
    return lines


def _scale_of(tokens: list):
    """(left, right, unit) from a scale line like '60 PERCENT 0', else None."""
    for i, t in enumerate(tokens):
        if _num(t) is not None:
            continue
        if "(" in t or ")" in t:
            return None
        before = [_num(x) for x in tokens[:i] if _num(x) is not None]
        after = [_num(x) for x in tokens[i + 1:] if _num(x) is not None]
        if before and after and before[-1] != after[0]:
            return before[-1], after[0], t
        if not before and after and _ukey(t) in ("APIGR", "GAPI", "API") and after[0] > 0:
            return 0.0, after[0], t
        if (before or after) and _ukey(t) in _UNITS:
            return (before[-1] if before else None), (after[0] if after else None), t
        return None
    return None


def _name_of(tokens: list):
    for t in tokens:
        letters = [c for c in t.lower() if c.isalpha()]
        if _num(t) is None and len(set(letters)) >= 2:
            return t
    return None


def _fill_missing_end(found: list) -> None:
    for p in found:
        sc = p["scale"]
        if sc is None or (sc[0] is not None and sc[1] is not None):
            continue
        left, right, unit = sc
        twin = next((q for q in found if q is not p and q["scale"] and None not in q["scale"][:2]
                     and _ukey(q["scale"][2]) == _ukey(unit)
                     and ((left is None and q["scale"][1] == right) or (right is None and q["scale"][0] == left))), None)
        if twin is not None:
            p["scale"] = (twin["scale"][0] if left is None else left,
                          twin["scale"][1] if right is None else right, unit)
            p["note"] = f"{'left' if left is None else 'right'} end not read - copied from {twin['raw']}"
        elif left is not None and right is None and _ukey(unit) in ("PERCENT", "PU", "%"):
            p["scale"] = (left, 0.0, unit)
            p["note"] = "right end not read - assumed 0"
        else:
            p["scale"] = None
            p["note"] = f"scale only partly read ({left} .. {right})"


def propose_curves_from_lines(tracks: dict, labels: np.ndarray | None = None,
                              names: list | None = None) -> list[dict]:
    """Rebuild the curve table from OCR words, track by track.

    ``tracks`` is ``{track_name: {x: (left, right), words: [...], grid?: str}}``
    as returned by :func:`pylogdigitizer.header_ocr.propose_header`. Each output
    row: name (printed), label, track, color, left, right, scale_type, backup,
    unit, check. ``color`` is ``black`` unless ``labels``/``names`` are given.
    """
    from pylogdigitizer.scan_color import legend_color
    rows, seen = [], set()
    for tr, t in tracks.items():
        xa, xb = t["x"]
        words = [w for w in t.get("words", []) if xa <= (w["x0"] + w["x1"]) / 2 <= xb]
        pend, found = None, []
        for ln in _header_lines(words):
            sc = _scale_of(ln["tok"])
            if sc:
                if pend is not None and pend["scale"] is None:
                    pend["scale"] = sc
                    found.append(pend); pend = None
                continue
            nm = _name_of(ln["tok"])
            if nm:
                if pend is not None:
                    found.append(pend)
                pend = dict(raw=nm, label=" ".join(x for x in ln["tok"] if _num(x) is None),
                            y0=ln["y0"], y1=ln["y1"], scale=None)
        if pend is not None:
            found.append(pend)
        _fill_missing_end(found)
        kind = "log" if t.get("grid") == "log" else "linear"
        for p in found:
            name = printed_name(p["raw"])
            if name in seen:
                name = f"{name}_{tr[-1]}"
            base, n_ = name, 2
            while name in seen:          # still taken (same printed name twice in one track, or GR_1 read)
                name = f"{base}_{n_}"; n_ += 1
            seen.add(name)
            left, right, unit = p["scale"] or (None, None, "")
            u = re.sub(r"[^A-Za-z0-9%/]", "", str(unit)).upper()
            color = "black"
            if labels is not None and names is not None:
                color = legend_color(labels, names, p["y0"], p["y1"], t["x"])
            rows.append(dict(name=name, label=p["label"], track=tr, color=color,
                             left=left, right=right, scale_type=kind, backup=None,
                             unit=_UNIT_FIX.get(u, u),
                             check=p.get("note") or ("" if p["scale"] else "no scale read"),
                             evidence=dict(name_y=round(float(p["y0"]), 1))))
    return rows


def apply_curve_fixes(rows: list, fixes: dict) -> tuple[list, list]:
    """Edit rows of a curve table by name. Unknown names with track/color/left/right are added."""
    out, used = [], set()
    for r in rows:
        key = r["name"]
        if fixes.get(key):
            used.add(key)
            r = dict(r, **fixes[key], check="fixed by caller")
            r["name"] = printed_name(r["name"])
        out.append(r)
    unknown = []
    for k, f in fixes.items():
        if k in used:
            continue
        if all(f.get(x) is not None for x in ("track", "color", "left", "right")):
            out.append(dict(dict(name=k, label="", backup=None, unit="", scale_type="linear"), **f,
                            check="added by caller"))
            out[-1]["name"] = printed_name(out[-1]["name"])
        else:
            unknown.append(k)
    return out, unknown


def merge_backup_rows(rows: list) -> list:
    """Two rows that are one curve with a wrap-around (backup) scale.

    Same track, colour, unit and scale type, and the second scale continues the
    first (for example GR 0-150 and 150-300). They become one row: the first
    scale is the main one, the second its ``backup``.
    """
    out, gone = list(rows), set()
    for a in rows:
        for b in rows:
            if (a is b or id(a) in gone or id(b) in gone or a.get("backup") or b.get("backup")
                    or a.get("track") != b.get("track") or a.get("color") != b.get("color")
                    or a.get("scale_type") != b.get("scale_type")
                    or _ukey(a.get("unit", "")) != _ukey(b.get("unit", ""))
                    or a.get("right") != b.get("left")
                    or None in (a.get("left"), a.get("right"), b.get("left"), b.get("right"))
                    or (a["right"] - a["left"]) * (b["right"] - b["left"]) <= 0):
                continue
            m = dict(a, name=min(a["name"], b["name"], key=len), backup=(b["left"], b["right"]),
                     wrap_side="right",
                     check=f"one curve: {a['name']} {a['left']:g}-{a['right']:g}, wraps onto {b['name']} {b['left']:g}-{b['right']:g}")
            ia = next(i for i, r in enumerate(out) if r is a)
            out[ia] = m
            del out[next(i for i, r in enumerate(out) if r is b)]
            gone |= {id(a), id(b)}
    return out

