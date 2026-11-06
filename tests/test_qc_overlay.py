import numpy as np
from types import SimpleNamespace
from pylogdigitizer import qc_overlay as qo


def _fake():
    H, W = 400, 300
    ink = np.zeros((H, W), bool)
    ink[:, 50:52] = True; ink[:, 250:252] = True            # track edges
    ink[100:102, 50:252] = True                             # grid row
    L = SimpleNamespace(ink=ink, centers=np.array([0.0, H]), edges=np.array([[50.0, 251.0], [50.0, 251.0]]),
                        edge_names=["track1.left", "track1.right"])
    pan = ink[:, 50:252].copy()
    rows = np.arange(0, H, 3).astype(float)
    T = dict(rows=rows, x=np.full(len(rows), 100.0))
    return dict(layout=L, trace=T, panel=pan, panel_rows=(0, H), value=np.full(len(rows), 50.0)), H


def test_overlay_draws_trace_edges_and_grid(tmp_path):
    R, H = _fake()
    img = qo.overlay_scan(R, "track1", str(tmp_path / "o.png"))
    assert (tmp_path / "o.png").exists() and img.shape[0] == H
    x_curve = 50 + int(round(201 * 100 / 201))
    assert (img[200, x_curve - 1:x_curve + 2] == qo.COLORS[0]).all(-1).any()   # red trace on its column
    assert (img[200, 49:53] == qo.EDGE_C).all(-1).any()                          # edge drawn
    assert (img[100, 120] == qo.GRID_C).all()                                    # grid row drawn


def test_overlay_skips_null_samples():
    R, H = _fake()
    R["value"][40:80] = np.nan                                  # rows 120-240 null
    img = qo.overlay_scan(R, "track1")
    x_curve = 150
    assert not (img[180, x_curve - 1:x_curve + 2] == qo.COLORS[0]).all(-1).any()
    assert qo.grid_lines(R["panel"])[0].tolist() == [100, 101]
