# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""The IBDP example fetcher reads only the needed members of a remote zip (HTTP range requests),
checks SHA-256 and caches. The server is faked here; nothing is downloaded."""
from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from pylogdigitizer import ibdp_examples as ie


class _Resp(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data); self.status = status; self.headers = headers
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


@pytest.fixture()
def fake_archive(monkeypatch):
    members = {ie._DIR + "a.pdf": b"%PDF-1.4 fake" * 500, ie._DIR + "a.las": b"~V\n" * 3000,
               "other/big.bin": bytes(200_000)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in members.items():
            z.writestr(k, v)
    blob = buf.getvalue()
    calls = []

    def urlopen(req, timeout=None):
        rng = req.get_header("Range")
        calls.append(rng)
        if req.get_method() == "HEAD":
            return _Resp(b"", 200, {"Content-Length": str(len(blob)), "Accept-Ranges": "bytes"})
        a, b = (int(x) for x in rng.split("=")[1].split("-"))
        return _Resp(blob[a:b + 1], 206, {})

    monkeypatch.setattr(ie.urllib.request, "urlopen", urlopen)
    files = [("x.pdf", ie._DIR + "a.pdf", hashlib.sha256(members[ie._DIR + "a.pdf"]).hexdigest(), 0),
             ("x.las", ie._DIR + "a.las", hashlib.sha256(members[ie._DIR + "a.las"]).hexdigest(), 0)]
    monkeypatch.setitem(ie.IBDP_EXAMPLES, "T", {"name": "test", "files": files})
    return members, calls, len(blob)


def test_fetch_reads_members_by_range_and_caches(fake_archive, tmp_path):
    members, calls, size = fake_archive
    f = ie.fetch_ibdp_example("T", cache=tmp_path)
    assert (f / "x.pdf").read_bytes() == members[ie._DIR + "a.pdf"]
    assert (f / "x.las").read_bytes() == members[ie._DIR + "a.las"]
    assert all(c is None or c.startswith("bytes=") for c in calls)
    n = len(calls)
    ie.fetch_ibdp_example("T", cache=tmp_path)                 # cached: no request at all
    assert len(calls) == n
    (f / "x.las").write_bytes(b"changed")                       # a bad cached copy is fetched again
    ie.fetch_ibdp_example("T", cache=tmp_path)
    assert (f / "x.las").read_bytes() == members[ie._DIR + "a.las"]


def test_fetch_checks_sha256(fake_archive, tmp_path, monkeypatch):
    bad = [(n, m, "0" * 64, s) for n, m, _, s in ie.IBDP_EXAMPLES["T"]["files"]]
    monkeypatch.setitem(ie.IBDP_EXAMPLES, "T", {"name": "test", "files": bad})
    with pytest.raises(IOError, match="checksum"):
        ie.fetch_ibdp_example("T", cache=tmp_path)
    assert not list((tmp_path / "T").glob("*"))                 # nothing half-written is left
    with pytest.raises(KeyError):
        ie.fetch_ibdp_example("nope", cache=tmp_path)
