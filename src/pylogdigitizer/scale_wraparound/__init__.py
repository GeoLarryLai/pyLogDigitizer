# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""scale_wraparound: off-scale (wrap / backup-scale) handling.

When a curve exceeds its scale the recorder either wraps it (re-enters from the opposite edge on a
backup scale, e.g. GR 150-300) and often shades the backup excursion (hatching). The tracer gets
virtual off-scale columns on both sides of the track (see workflows.trace_panel); rows where the
path sits in a virtual column are wrap intervals. For those rows the backup value is read from the
extent of ink connected to the re-entry edge (the hatched area or the backup curve itself).
"""
from __future__ import annotations
import numpy as np

from pylogdigitizer.track_grid_detection import _runs


def detect_wraps(offscale: np.ndarray, min_len: int = 2) -> list[tuple[int, int]]:
    """Intervals [a, b) of consecutive True values (rows where the trace is off-scale)."""
    e = np.flatnonzero(np.diff(np.r_[0, np.asarray(offscale, np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2]) if b - a >= min_len]


def edge_jump_wraps(x: np.ndarray, width: float, wrap_side: str = "right", edge: float = 0.12,
                    off_out=None, off_in=None, jump: float = 0.5, cancel_at: float | None = None,
                    cancel_closes: bool = False) -> list[tuple[int, int]]:
    """Wraps the tracer itself followed: the path leaves through the ``wrap_side`` edge and next
    appears at the opposite edge (it is now on the backup trace) until it jumps back the other way.
    ``off_out`` / ``off_in``: optional per-row flags for the off-scale columns beyond each edge.
    Returns intervals [a, b) of rows on the backup trace."""
    x = np.asarray(x, float) / float(width)
    if wrap_side != "right":
        x = 1.0 - x
        off_out, off_in = off_in, off_out
    hi = (x > 1 - edge) | (np.zeros(len(x), bool) if off_out is None else np.asarray(off_out, bool))
    lo = (x < edge) | (np.zeros(len(x), bool) if off_in is None else np.asarray(off_in, bool))
    # a continuous curve cannot cross half the track between two samples: such a jump into the
    # entry-edge zone is a wrap (the exit side is often faint, so only the landing must be at the edge)
    far = np.where(hi, 1.0, np.where(lo, 0.0, x))
    out, a, last_lo = [], None, None
    for i in range(1, len(x)):
        if a is None and lo[i] and not hi[i] and far[i - 1] - far[i] > jump:
            a = i
        elif a is not None and lo[i - 1] and not lo[i] and far[i] - far[i - 1] > jump:
            out.append((a, i)); a = None
        elif a is not None and far[i] > (1 - jump if cancel_at is None else cancel_at) and not (far[i] - far[i - 1] > jump):
            # the "backup" stretch drifted across the track without jumping back: it was a full-width
            # excursion of the main curve (e.g. a cycle-skip spike), not a wrap. With ``cancel_closes``
            # the wrap is kept up to the last entry-edge sample (the main curve came back unseen)
            if cancel_closes and last_lo is not None and last_lo >= a:
                out.append((a, last_lo + 1))
            a = None
        if a is not None and lo[i]:
            last_lo = i
    if a is not None:
        out.append((a, len(x)))
    return out


def xr_seq(x, width, wrap_side):
    """Position as a fraction of the width measured from the entry (non-wrap) edge."""
    x = np.asarray(x, float) / float(width)
    return x if wrap_side == "right" else 1.0 - x


def ink_jump_wraps(x: np.ndarray, on_ink: np.ndarray, width: float, wrap_side: str = "right",
                   edge: float = 0.15, jump: float = 0.4, conn: float = 0.3, border: float = 4.0,
                   min_run: int = 5, gap: int = 10, look: int = 30):
    """Wraps found on the inked part of a trace only. Where the curve leaves through the wrap edge
    and the backup re-enters at the opposite edge, a row-wise tracer often draws a slanted
    connector across empty paper instead of one jump, which hides the jump from
    ``edge_jump_wraps``. Dropping the off-ink samples restores it.
    Returns (wraps [a, b) in sample indices, connectors: boolean mask of off-ink samples that bridge
    more than ``conn`` of the width; these carry no reading and should be null)."""
    x = np.asarray(x, float); n = len(x)
    on = np.asarray(on_ink, bool) & np.isfinite(x)
    on &= (x > border) & (x < width - 1 - border)           # border-hugging samples are not curve evidence
    p_ = np.r_[0, on.astype(np.int8), 0]; d_ = np.diff(p_)  # isolated specks inside a connector are not either
    for a_, b_ in zip(np.flatnonzero(d_ == 1), np.flatnonzero(d_ == -1)):
        if b_ - a_ < min_run:
            on[a_:b_] = False
    idx = np.flatnonzero(on)
    conn_m = np.zeros(n, bool)
    if len(idx) < 3:
        return [], conn_m
    # continuity: the curve ran into the wrap edge zone, vanished for >= ``gap`` samples, and came
    # back near the opposite edge -> an exit the tracer never drew; insert it (an edge sample)
    xr = x / float(width) if wrap_side == "right" else 1.0 - x / float(width)
    seq_x, seq_i = [], []
    for k, i in enumerate(idx):
        if k and i - idx[k - 1] >= gap:
            prev = xr[idx[max(0, k - look):k]]; nxt = xr[idx[k:k + look]]
            if prev.max() > 1 - edge and nxt.min() < edge and np.median(nxt) < edge:   # backup hugs the entry edge first
                e_hi = width - 1.0 if wrap_side == "right" else 0.0
                seq_x += [e_hi, width - 1.0 - e_hi]; seq_i += [i, i]
        seq_x.append(x[i]); seq_i.append(i)
    seq_x = np.array(seq_x); seq_i = np.array(seq_i)
    # continuity: once the curve left through the wrap edge and the backup appeared at the other,
    # the backup may swing far into the track (high values); only reaching the wrap edge zone itself
    # without a jump (impossible for a backup) cancels the wrap
    fw = edge_jump_wraps(seq_x, width, wrap_side, edge=edge, jump=jump, cancel_at=1 - edge, cancel_closes=True)
    # a backup enters at the edge and stays near it for a while (it starts just above the backup
    # scale's low end); a landing that moves straight back into the track is a mis-trace, not a wrap
    fw = [(a, b) for a, b in fw if np.median(np.r_[xr_seq(seq_x[a:a + look], width, wrap_side)]) < edge]
    wr = [(int(seq_i[a]), int(seq_i[b]) if b < len(seq_i) else n) for a, b in fw]
    wr = [(a, b) for a, b in wr if b > a]
    for p_, q_ in zip(idx[:-1], idx[1:]):
        if q_ - p_ > 1 and abs(x[q_] - x[p_]) > conn * width:
            conn_m[p_ + 1:q_] = True
    return wr, conn_m


def edge_extent(rows_ink: np.ndarray, from_left: bool = True, max_gap: int = 30, skip: int = 8) -> np.ndarray:
    """For each row, how far ink connected to the entry edge extends (gaps <= ``max_gap`` px are
    bridged, e.g. between hatch lines). Returns x per row (NaN if no ink near the edge)."""
    a = rows_ink if from_left else rows_ink[:, ::-1]
    h, w = a.shape
    out = np.full(h, np.nan)
    for i in range(h):
        xs = np.flatnonzero(a[i, skip:]) + skip
        if not len(xs) or xs[0] > skip + max_gap:
            continue
        g = np.flatnonzero(np.diff(xs) > max_gap)
        last = xs[g[0]] if len(g) else xs[-1]
        out[i] = last if from_left else w - 1 - last
    return out


def unwrap(values: np.ndarray, wraps: list[tuple[int, int]], backup_values: np.ndarray) -> np.ndarray:
    """Replace values inside wrap intervals by ``backup_values`` (already on the backup scale)."""
    v = np.asarray(values, float).copy()
    for a, b in wraps:
        seg = backup_values[a:b]
        ok = np.isfinite(seg)      # no backup ink found: keep the main-scale (edge) value
        v[a:b][ok] = seg[ok]
    return v


def backup_values_from_panel(panel: np.ndarray, row_idx: np.ndarray, fy: int, wraps: list[tuple[int, int]],
                             backup_scale, from_left: bool = True, max_gap: int = 30, use_hatch: bool = True,
                             edge_pad: float = 3.0) -> np.ndarray:
    """Backup-scale values for traced samples inside ``wraps``. With ``use_hatch`` the extent is
    measured on the hatch-line mask (robust against other curves in the track) and ``edge_pad``
    px are added for the outline curve; otherwise on all ink."""
    out = np.full(len(row_idx), np.nan)
    for a, b in wraps:
        y0, y1 = a * fy, b * fy
        src = hatch_mask(panel[y0:y1]) if use_hatch else panel[y0:y1]
        for k in range(a, b):
            x = edge_extent(src[(k - a) * fy:(k - a) * fy + fy], from_left, max_gap, skip=0 if use_hatch else 8)
            if np.isfinite(x).any():
                xv = (np.nanmax(x) + edge_pad) if from_left else (np.nanmin(x) - edge_pad)
                out[k] = backup_scale(xv)
    # fill short holes inside wraps by interpolation
    for a, b in wraps:
        seg = out[a:b]
        ok = np.isfinite(seg)
        if ok.sum() >= 2 and (~ok).any():
            seg[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), seg[ok])
    return out


def hatch_mask(panel: np.ndarray, length: int = 21) -> np.ndarray:
    """Pixels on straight diagonal strokes (either direction) >= ``length`` px: the hatch lines
    used to shade backup excursions. Curves rarely stay straight-diagonal that long."""
    import cv2
    u = panel.astype(np.uint8)
    k = np.eye(length, dtype=np.uint8)
    m1 = cv2.morphologyEx(u, cv2.MORPH_OPEN, k)
    m2 = cv2.morphologyEx(u, cv2.MORPH_OPEN, k[:, ::-1].copy())
    return (m1 | m2).astype(bool)


def hatch_rows(panel: np.ndarray, fy: int, min_runs: int = 3, max_gap: int = 40, min_extent: int = 30,
               close: int = 6, min_len: int = 5, length: int = 21,
               from_left: bool = True, edge: int = 40) -> np.ndarray:
    """Per traced sample (block of ``fy`` rows): True where hatching is present, i.e. a row has
    >= ``min_runs`` diagonal-stroke runs chained with gaps <= ``max_gap`` spanning >= ``min_extent``
    px and starting within ``edge`` px of the re-entry edge. Gaps shorter than ``close`` samples are closed and runs shorter than ``min_len`` dropped."""
    hm = hatch_mask(panel, length)
    if not from_left:
        hm = hm[:, ::-1]
    n = hm.shape[0] // fy
    row_ok = np.zeros(hm.shape[0], bool)
    cand_y = np.flatnonzero(hm[: n * fy].sum(axis=1) >= min_runs)
    for y in cand_y:
        xs = np.flatnonzero(hm[y])
        if len(xs) < 2:
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(xs) > 1) + 1]
        if len(starts) < min_runs:
            continue
        big = np.flatnonzero(np.diff(xs) > max_gap)
        segs = np.split(xs, big + 1)
        for sgm in segs[:1]:
            if sgm[0] > edge:
                break
            nr = 1 + int(np.sum(np.diff(sgm) > 1))
            if nr >= min_runs and sgm[-1] - sgm[0] >= min_extent:
                row_ok[y] = True
                break
    out = row_ok[: n * fy].reshape(n, fy).any(axis=1)
    idx = np.flatnonzero(out)
    for i, j in zip(idx[:-1], idx[1:]):
        if 1 < j - i <= close:
            out[i:j] = True
    e = np.flatnonzero(np.diff(np.r_[0, out.astype(np.int8), 0]))
    for a_, b_ in zip(e[::2], e[1::2]):
        if b_ - a_ < min_len:
            out[a_:b_] = False
    return out


def default_backup(cv: dict) -> tuple:
    """The usual wrap-around scale: the next full track width (0-150 -> 150-300; log 2-200 -> 200-20000)."""
    l, r = float(cv["left"]), float(cv["right"])
    return (r, r * r / l) if cv.get("scale_type") == "log" and l > 0 else (r, r + (r - l))


def fill_backups(labels: np.ndarray, names: list, fill_colors, x_range, body_rows, margin: int = 4,
                 start_px: int = 12, gap_px: int = 8, hole_rows: int = 6, min_rows: int = 6):
    """Shaded wrap-arounds (colour fill between the backup curve and the left edge).

    Where a curve runs off the right edge these prints draw its backup curve in from the
    left and shade the area between (yellow for GR, cyan for neutron, ...). Row by row,
    the right end of a fill that starts at the left edge is where the backup curve is.

    Attribution of a fill colour to a bounding curve colour is heuristic and can mis-assign
    (for example yellow shading next to a green curve). Every attribution is therefore
    flagged for review.

    Returns ``(fills, reviews)`` where ``fills`` is ``{curve_colour: (sheet_rows, sheet_x)}``
    (x is NaN where there is no fill) and ``reviews`` is ``{curve_colour: reason}``.
    """
    ids = [names.index(c) for c in (fill_colors or []) if c in names]
    if not ids:
        return {}, {}
    x0, x1 = int(round(x_range[0])) + margin, int(round(x_range[1])) - margin
    r0, r1 = body_rows
    lab = labels[r0:r1, x0:x1]
    fill = np.isin(lab, ids)
    h = fill.shape[0]
    ext = np.full(h, np.nan)
    for i in np.flatnonzero(fill[:, :start_px].any(axis=1)):
        xs = np.flatnonzero(fill[i])
        g = np.flatnonzero(np.diff(xs) > gap_px)
        ext[i] = xs[g[0]] if len(g) else xs[-1]
    ok = np.isfinite(ext)
    if ok.sum() < min_rows:
        return {}, {}
    idx = np.flatnonzero(ok)
    for p_, q in zip(idx[:-1], idx[1:]):
        if 1 < q - p_ <= hole_rows + 1:
            ext[p_ + 1:q] = np.interp(np.arange(p_ + 1, q), [p_, q], [ext[p_], ext[q]])
    for a, b in _runs(np.isfinite(ext)):
        if b - a < min_rows:
            ext[a:b] = np.nan
    if not np.isfinite(ext).any():
        return {}, {}
    skip = {"white"} | set(fill_colors)
    counts = {}
    for i in np.flatnonzero(np.isfinite(ext))[::3]:
        e = int(ext[i])
        for c in set(lab[i, e + 1:e + 7].tolist()):
            if names[c] not in skip:
                counts[names[c]] = counts.get(names[c], 0) + 1
    if not counts:
        return {}, {}
    color = max(counts, key=counts.get)
    used = [c for c in (fill_colors or []) if c in names]
    reason = (f"shading-backup attribution: {'/'.join(used)} fill bounded by {color} "
              f"({counts[color]} votes) — review the backup values")
    return {color: (np.arange(r0, r1, dtype=float), ext + x0)}, {color: reason}

