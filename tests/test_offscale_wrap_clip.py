"""Off-scale handling on a traced curve (synthetic sheet, no example data):
wrapped overflow (re-enters from the other edge) is read on the backup scale; a curve simply
clipped at the track edge (no wrapped trace on those rows) is written as null."""
import numpy as np
import pytest

H, X0, X1 = 3000, 100, 600
TW = X1 - X0


def _value(rows):
    v = 70 + 30 * np.sin(rows / 160.0)
    for c in (900, 2100):                             # two smooth overflows peaking near 230 API
        v = v + 140 * np.exp(-0.5 * ((rows - c) / 90.0) ** 2)
    return v


def _sheet(tmp_path, wrap_second: bool):
    import cv2
    from PIL import Image
    img = np.full((H, 700), 255, np.uint8)
    img[:, X0 - 4:X0] = 0; img[:, X1:X1 + 4] = 0          # track borders
    rows = np.arange(H); v = _value(rows)
    x = np.where(v <= 150, v / 150 * TW, np.nan)
    wrap = np.where((v > 150) & ((rows < 1500) | wrap_second), (v - 150) / 150 * TW, np.nan)   # backup trace
    clip = np.where((v > 150) & ~((rows < 1500) | wrap_second), TW - 1.0, np.nan)            # pinned at the edge
    for xs in (x, wrap, clip):
        ok = np.isfinite(xs)
        for a, b in zip(np.flatnonzero(np.diff(np.r_[0, ok.astype(int)]) == 1), np.flatnonzero(np.diff(np.r_[ok.astype(int), 0]) == -1) + 1):
            pts = np.c_[X0 + xs[a:b], rows[a:b]].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, 0, 4)
    p = tmp_path / f"wrap_{int(wrap_second)}.tif"; Image.fromarray(img).save(p)
    return p, rows, v


def _session(p, backup):
    from pylogdigitizer.assisted_tracing import AssistedSession
    s = AssistedSession(str(p), curve="GR", scale=(0, 150), backup=backup, wrap_side="right",
                        auto_layout=False, track=None).open()
    s.set_depth_anchors([(0.0, 1000.0), (float(H), 1300.0)])          # 10 px/ft
    s.set_section(box=(float(X0), float(X1)), rows=(0, H))
    s.trace_kw.update(min_area=0, exclude_cols=[(0, 6), (TW - 6, TW + 2)])
    s.auto_trace()
    return s


@pytest.mark.parametrize("wrap_second", [False, True])
def test_wrapped_overflow_read_on_backup_and_clipped_is_null(tmp_path, wrap_second):
    from pylogdigitizer.workflows import color_curve_values
    p, rows, v = _sheet(tmp_path, wrap_second)
    s = _session(p, backup=(150, 300))
    d, val = color_curve_values(s)
    truth = np.interp((d - 1000) * 10, rows, v)
    first = (truth > 165) & (d < 1150)                            # around row 900: wrapped on the print
    second = (truth > 165) & (d > 1150)                           # around row 2100
    assert np.isfinite(val[first]).mean() > 0.8
    assert np.nanmedian(np.abs(val[first] - truth[first])) < 10   # read on the backup scale, not the edge value
    if wrap_second:
        assert np.nanmedian(np.abs(val[second] - truth[second])) < 10
    else:
        assert np.isnan(val[second]).mean() > 0.8                 # clipped, no wrapped trace: null
    inside = (truth < 140) & (truth > 10)
    assert np.nanmedian(np.abs(val[inside] - truth[inside])) < 5


def test_clipped_without_backup_is_null(tmp_path):
    from pylogdigitizer.workflows import color_curve_values
    p, rows, v = _sheet(tmp_path, wrap_second=False)
    s = _session(p, backup=None)
    d, val = color_curve_values(s)
    truth = np.interp((d - 1000) * 10, rows, v)
    second = (truth > 165) & (d > 1150)
    assert np.isnan(val[second]).mean() > 0.8


def test_wrap_without_backup_assumes_next_track_width(tmp_path):
    from pylogdigitizer.workflows import apply_auto_backup, color_curve_values
    p, rows, v = _sheet(tmp_path, wrap_second=True)
    s = _session(p, backup=None)
    apply_auto_backup(s, dict(name="GR", left=0, right=150, scale_type="linear"))
    assert s.scale["backup"] == [150.0, 300.0] and "wrap_note" in s.errors
    s.auto_trace()
    d, val = color_curve_values(s)
    truth = np.interp((d - 1000) * 10, rows, v)
    over = truth > 165
    assert np.nanmedian(np.abs(val[over] - truth[over])) < 10
