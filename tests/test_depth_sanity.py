import numpy as np
from pylogdigitizer import header_ocr as hdo, workflows as wf


def test_depth_range_warning():
    hd = dict(top=3700.0, bottom=4464.0)
    assert hdo.depth_range_warning(3750, 4450, hd) is None
    assert "deeper" in hdo.depth_range_warning(3750, 5600, hd)
    assert "above" in hdo.depth_range_warning(602, 2803, hd)
    assert hdo.depth_range_warning(602, 2803, dict(top=None, bottom=None)) is None


def test_calibration_error_says_how_to_pass_depth():
    from pylogdigitizer import depth_calibration as dc
    rows = np.arange(5) * 300.0 + 1000
    try:
        dc.calibrate_labels(rows, rows, [None] * 5, None, None)
    except dc.CalibrationError as e:
        assert "first_depth=" in str(e)
    else:
        raise AssertionError("expected CalibrationError")


def test_section_choice_never_guesses():
    from pylogdigitizer import workflows as wf
    secs = [dict(index=0, top=600, base=2800, row0=0, row1=1, px_per_ft=3.0),
            dict(index=1, top=3700, base=4500, row0=2, row1=3, px_per_ft=7.6)]
    assert wf._header_match(secs, dict(top=3720, bottom=4480)) == [1]
    assert wf._header_match(secs, dict(top=419, bottom=4464)) == []       # spans both: no guess
    assert wf._header_match(secs, dict(top=None, bottom=None)) == []
    e = wf.SectionChoiceNeeded(secs)
    assert isinstance(e, wf.CalibrationError) and "section=1" in str(e) and "3700-4500" in str(e)
