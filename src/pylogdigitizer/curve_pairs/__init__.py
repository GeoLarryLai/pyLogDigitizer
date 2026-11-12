# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""curve_pairs: one log printed twice in the same colour on two scales.

Colour cannot separate a gamma-ray 0-150 from the same red ink on 0-2000 in
the same track. Their order across the track can: both scales start at the
left edge, so a value sits farther right on the narrower scale.
:func:`shared_scale_pairs` finds such pairs in a curve table.

:func:`splice_pairs` is optional: join the pair into one continuous curve
(narrow scale, except where the wide scale reads past the narrow end).
Off by default — turn it on with a clear setting in the notebook or caller.
"""
from __future__ import annotations

import re

import numpy as np

__all__ = ["shared_scale_pairs", "splice_pairs", "unit_key", "las_token"]


def unit_key(unit: str) -> str:
    """Unit compared loosely: letters and % only, upper case ('g/cc' == 'G/CC')."""
    return re.sub(r"[^A-Za-z%]", "", str(unit or "")).upper()


def las_token(name: str) -> str:
    """LAS-safe token: letters, digits and _ only, upper case ('' -> 'CURVE')."""
    return re.sub(r"[^A-Za-z0-9_]", "", str(name)).upper() or "CURVE"


def shared_scale_pairs(rows: list) -> list:
    """Pairs of curves that are one log printed twice (same colour, track, unit, left end)."""
    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if (a.get("track") == b.get("track") and a.get("color") == b.get("color")
                    and a.get("scale_type") == b.get("scale_type")
                    and unit_key(a.get("unit", "")) == unit_key(b.get("unit", ""))
                    and a.get("left") == b.get("left")
                    and a.get("right") != b.get("right")
                    and (a["right"] - a["left"]) * (b["right"] - b["left"]) > 0):
                out.append((a, b))
    return out


def splice_pairs(rows: list, curves: dict, margin: float = 1.02):
    """One continuous curve per shared-scale pair (``<name>_SPLICED``).

    Uses the narrow-scale curve except where the wide-scale curve reads beyond
    the narrow scale's end (or the narrow curve is null). Nothing is changed
    in place. Opt-in — the caller decides whether to splice.
    """
    rows_out, curves = list(rows), dict(curves)
    for a, b in shared_scale_pairs([r for r in rows if not r.get("spliced")]):
        if a["name"] not in curves or b["name"] not in curves:
            continue
        narrow, wide = (a, b) if abs(a["right"] - a["left"]) < abs(b["right"] - b["left"]) else (b, a)
        vn, vw = np.asarray(curves[narrow["name"]], float), np.asarray(curves[wide["name"]], float)
        end = narrow["right"]
        with np.errstate(invalid="ignore"):
            beyond = np.isfinite(vw) & ((vw - end) * np.sign(end - narrow["left"]) > abs(end) * (margin - 1))
        v = np.where(beyond | ~np.isfinite(vn), vw, vn)
        name = las_token(re.sub(r"_\d+$", "", narrow["name"]) + "_SPLICED")
        right = narrow["right"] + (narrow["right"] - narrow["left"])
        rows_out.insert(rows_out.index(narrow), dict(narrow, name=name, right=right, spliced=True, panel=name,
                                                     check=f"{narrow['name']} + {wide['name']} where it reads past {end:g}"))
        curves[name] = v
    return rows_out, curves
