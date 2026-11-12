# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fetch-on-demand public vector-PDF example: the Illinois Basin - Decatur Project (IBDP) well CCS1.

The files are the Schlumberger merged petrophysical log of injection well CCS1 (CCS Well #1, ADM,
Macon County, Illinois; API 121152341500) as a vector PDF, and the LAS file it was plotted from.
They sit inside the public IBDP well-information archive on NETL's Energy Data eXchange (EDX):

    Illinois State Geological Survey (ISGS), Illinois Basin - Decatur Project (IBDP), U.S.
    Department of Energy (DOE): IBDP Well Information. https://doi.org/10.18141/1854144
    Licence: IBDP dataset licence (based on CC BY 4.0, non-commercial),
    https://co2datashare.org/view/license/e27b2b34-a3fc-463f-adbf-bd14c83640f4

Nothing is shipped in the repository. The archive is 4.6 GB, so :func:`fetch_ibdp_example` reads
only the two members it needs with HTTP range requests (about 2 MB transferred), checks their
SHA-256 and caches them under ``$PYLOGDIGITIZER_CACHE`` (or ``$XDG_CACHE_HOME``/``~/.cache``)
``/pylogdigitizer/ibdp``.

    >>> folder = fetch_ibdp_example("CCS1")            # doctest: +SKIP
    >>> pdf = folder / "CCS1_PEX_Merged_2in.pdf"        # doctest: +SKIP
"""
from __future__ import annotations

import io
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from pylogdigitizer.kgs_examples import _sha256, cache_dir as _root_cache

__all__ = ["IBDP_EXAMPLES", "IBDP_CREDIT", "fetch_ibdp_example"]

IBDP_ZIP = ("https://edx.netl.doe.gov/storage/f/edx/2021/07/2021-07-09T20:56:55.522Z/"
            "353c30db-2bb9-4bc9-8b36-a9fedf4024f7/ibdp_well_information.zip")
IBDP_CREDIT = ("Illinois State Geological Survey (ISGS), Illinois Basin - Decatur Project (IBDP), "
               "U.S. Department of Energy (DOE); https://doi.org/10.18141/1854144; IBDP dataset licence "
               "(CC BY 4.0-based, non-commercial): "
               "https://co2datashare.org/view/license/e27b2b34-a3fc-463f-adbf-bd14c83640f4")
_DIR = ("IBDP_Well_Information/IBDP_Geophysical_Logs/1211523415_CCS1/"
        "PEX_Petrophysical_Interpretations_V20150601/")

# key -> description and files: (local name, member of the archive, sha256, size in bytes)
IBDP_EXAMPLES: dict[str, dict] = {
    "CCS1": {
        "name": "CCS Well #1 (CCS1), ADM, Decatur, Macon County, Illinois; API 121152341500",
        "files": [
            ("CCS1_PEX_Merged_2in.pdf", _DIR + "1211523415_20140326_PEX_Merged_Petrophysical_Data_2in.pdf",
             "06c983447637ab27e84c8db02311c1056745ce2efee548b8e4823f9afe082db1", 1209760),
            ("CCS1_PEX_Merged.las", _DIR + "1211523415_PEX_Merged.las",
             "ba75f2de7c2cb44be8274b820d491d26b1a7a301412bda861633b9b88515deef", 2188275),
        ],
    },
}


class _RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url: str, timeout: float = 120.0):
        from pylogdigitizer import __version__
        self.url, self.timeout, self.pos = url, timeout, 0
        self.headers = {"User-Agent": f"pyLogDigitizer/{__version__}"}
        req = urllib.request.Request(url, method="HEAD", headers=self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise IOError(f"{url} does not serve byte ranges")
            self.size = int(r.headers["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={**self.headers, "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise IOError(f"{self.url}: range request answered with HTTP {r.status}")
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _extract(zf: zipfile.ZipFile, member: str, dest: Path, sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, zf.open(member) as src:     # zipfile checks the CRC
            while chunk := src.read(1 << 20):
                out.write(chunk)
        got = _sha256(Path(tmp))
        if got != sha256:
            raise IOError(f"checksum mismatch for {member}: expected {sha256}, got {got}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def fetch_ibdp_example(key: str = "CCS1", cache: str | Path | None = None, force: bool = False) -> Path:
    """Download (once) the files of one IBDP example and return its cached folder.

    Only the needed members of the public archive are read (HTTP range requests). Files already in
    the cache with the right SHA-256 are not downloaded again; ``force=True`` re-downloads.
    Credit: see :data:`IBDP_CREDIT`."""
    if key not in IBDP_EXAMPLES:
        raise KeyError(f"unknown IBDP example {key!r}; known: {', '.join(IBDP_EXAMPLES)}")
    root = Path(cache) if cache is not None else _root_cache().parent / "ibdp"
    folder = root / key
    folder.mkdir(parents=True, exist_ok=True)
    todo = [(name, member, sha) for name, member, sha, _size in IBDP_EXAMPLES[key]["files"]
            if force or not (folder / name).exists() or _sha256(folder / name) != sha]
    if todo:
        with zipfile.ZipFile(io.BufferedReader(_RangeFile(IBDP_ZIP), buffer_size=1 << 20)) as zf:
            for name, member, sha in todo:
                _extract(zf, member, folder / name, sha)
    return folder
