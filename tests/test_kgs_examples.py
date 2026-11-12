"""On-demand KGS example download: registry, cache, checksum. No network (urlopen is mocked)."""
import hashlib
import io

import pytest

from pylogdigitizer import kgs_examples as kx


def _fake_registry(monkeypatch, payload=b"scan-bytes"):
    sha = hashlib.sha256(payload).hexdigest()
    reg = {"TEST_1": {"folder": "TEST_1_15-000-00000", "name": "TEST 1",
                      "files": [("1.tif", "https://example.invalid/1.tif", sha, len(payload)),
                                ("1.las", "https://example.invalid/1.las", sha, len(payload))]}}
    monkeypatch.setattr(kx, "KGS_EXAMPLES", reg)
    calls = []

    def urlopen(url, timeout=None):
        calls.append(getattr(url, "full_url", url))     # a Request (with the User-Agent) or a URL
        return io.BytesIO(payload)
    monkeypatch.setattr(kx.urllib.request, "urlopen", urlopen)
    return calls


def test_registry_covers_shipped_and_training_wells():
    for w in ("USA_A_16", "MOBIL_12-1", "Chitwood_12", "Hall_D_4", "Minzer_1", "MAKEMSON_A_1",
              "WEATHERFORD_B_1", "BACASTOW_1", "GOHEEN_UNIT_1"):
        files = kx.kgs_example_files(w)
        assert files and all(u.startswith("https://") for _, u, _, _ in files)
        assert all(len(s) == 64 for _, _, s, _ in files)
    assert kx.kgs_example_files("USA_A_16_15-055-20536") == kx.kgs_example_files("USA_A_16")   # folder name works
    with pytest.raises(KeyError):
        kx.kgs_example_files("NOPE")


def test_fetch_downloads_once_and_caches(monkeypatch, tmp_path):
    calls = _fake_registry(monkeypatch)
    d = kx.fetch_kgs_example("TEST_1", cache=tmp_path)
    assert d == tmp_path / "TEST_1_15-000-00000" and (d / "1.tif").read_bytes() == b"scan-bytes"
    assert len(calls) == 2
    kx.fetch_kgs_example("TEST_1", cache=tmp_path)                 # cached, checksum ok: no download
    assert len(calls) == 2
    kx.fetch_kgs_example("TEST_1", files=["1.las"], cache=tmp_path, force=True)
    assert calls[-1].endswith("1.las") and len(calls) == 3
    with pytest.raises(KeyError):
        kx.fetch_kgs_example("TEST_1", files=["2.tif"], cache=tmp_path)
    assert kx.fetch_kgs_examples(["TEST_1"], cache=tmp_path) == tmp_path


def test_checksum_mismatch_leaves_no_file(monkeypatch, tmp_path):
    _fake_registry(monkeypatch)
    reg = kx.KGS_EXAMPLES["TEST_1"]
    reg["files"] = [(n, u, "0" * 64, s) for n, u, _, s in reg["files"]]
    with pytest.raises(IOError):
        kx.fetch_kgs_example("TEST_1", cache=tmp_path)
    assert not list((tmp_path / "TEST_1_15-000-00000").glob("*"))


def test_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PYLOGDIGITIZER_CACHE", str(tmp_path))
    assert kx.cache_dir() == tmp_path / "kgs"
