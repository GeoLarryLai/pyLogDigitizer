import warnings
import numpy as np
import pytest
from pylogdigitizer import las_export as le

pytest.importorskip("lasio")


def test_write_and_read_back(tmp_path):
    d = np.arange(1000, 1100.5, 0.5)
    gr = np.linspace(20, 120, d.size); gr[5] = np.nan
    p = le.write_las(tmp_path / "a.las", d, {"GR": gr}, units={"GR": "GAPI"})
    assert le.verify_las(p, d, {"GR": gr}) == []


def test_hard_errors(tmp_path):
    with pytest.raises(le.LASExportError, match="increase"):
        le.write_las(tmp_path / "b.las", np.array([10.0, 9.0, 8.0]), {"GR": np.ones(3)})
    with pytest.raises(le.LASExportError, match="fewer than 2"):
        le.write_las(tmp_path / "c.las", np.array([10.0]), {"GR": np.ones(1)})
    with pytest.raises(le.LASExportError, match="values but"):
        le.write_las(tmp_path / "d.las", np.arange(5.0), {"GR": np.ones(3)})


def test_soft_warnings():
    w = le.check_las_data(np.array([0, 0.5, 1.5, 2.0]), {"GR": np.array([1, 2, -999.25, 3])}, units={})
    txt = " ".join(w)
    assert "step is not constant" in txt and "No unit" in txt and "null value" in txt
    assert any("looks odd" in x for x in le.check_las_data(np.array([90000.0, 90001.0]), {"GR": np.ones(2)}, {"GR": "GAPI"}))
    assert le.check_las_data(np.arange(0, 10, 0.5), {"GR": np.ones(20)}, {"GR": "GAPI"}) == []
