# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fetch-on-demand public KGS example logs (scans + LAS) into a local cache.

The repository ships only three small black-and-white examples and one colour crop. The
larger public Kansas Geological Survey files used for U-Net training, the segmenter notebook
and batch checks are downloaded on demand from the KGS URLs listed in
``example_data/SOURCES.txt``, checked against their SHA-256 and cached under
``$PYLOGDIGITIZER_CACHE`` (or ``$XDG_CACHE_HOME``/``~/.cache``) ``/pylogdigitizer/kgs``.

The source of this material is the Kansas Geological Survey website at
http://www.kgs.ku.edu/. All Rights Reserved.

    >>> root = fetch_kgs_examples(["Hall_D_4", "Minzer_1"])     # doctest: +SKIP
    >>> scan = fetch_kgs_example("Hall_D_4") / "193004.tif"      # doctest: +SKIP
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

__all__ = ["KGS_EXAMPLES", "cache_dir", "fetch_kgs_example", "fetch_kgs_examples", "kgs_example_files"]

# well key -> folder (as used by dataset profiles / ml_training_data.WELL_SPECS) and files:
# (file name, public KGS URL, sha256, size in bytes)
KGS_EXAMPLES: dict[str, dict] = {
    "USA_A_16": {"folder": "USA_A_16_15-055-20536", "name": "USA 'A' 16", "files": [
        ("4341.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/W/26S34/4341.tif",
         "e7825962a04201dce351ea73bcd2823f9e4972e6bc57b4ede50d2239fc590389", 4068050),
        ("4342.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/W/26S34/4342.tif",
         "19af2b56cca93f31c020b41f2166cca28b204b0055351ee8092ccc99ac02fb1e", 5596310),
        ("1042553273.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/26S34W/1042553273.las",
         "d5618bf182ed8c338c315928095b3c5d6dc752cbad4f68a96079501c90c190c6", 378324),
    ]},
    "MOBIL_12-1": {"folder": "MOBIL_12-1_15-189-20087", "name": "MOBIL 12-1", "files": [
        ("227233.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/31S35/227233.tif",
         "4dd28200dfb48b3d4bdffa152a438b95b87ea7f2992599fd6f308aa64e2c15cc", 546310),
        ("227234.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/31S35/227234.tif",
         "29976235f260e143786f0437b2e262691ffbec66522826486209f342bfe61db6", 1185698),
        ("227240.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/31S35/227240.tif",
         "61f5fa16b0f63ee79c92d636c0ed2b6ef38ac7f4be823f3d9a9f1a6b5c211ae7", 584952),
        ("1044084189.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/31S35W/1044084189.las",
         "b43a387898c20a8a3e55e806b2781146ef71aef1bcb10c19ca8a3f64fc5b7771", 293310),
    ]},
    "Chitwood_12": {"folder": "Chitwood_12_15-151-10469", "name": "Chitwood 12", "files": [
        ("260243.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/28S12/260243.tif",
         "9811cebf8f2a17e71e223e11dc205a453c34bd4981334a678c8a8196f45762f4", 295676),
        ("260244.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/28S12/260244.tif",
         "b4b2c93b1aa1e705a24d88772c38da1bb9f25a87b87ed5a51e2d5dc9056ed48f", 117376),
        ("260245.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/28S12/260245.tif",
         "a6762affd73f3ed3110e940738c883f88108d402f391fe6cfaeb835911a9843c", 786188),
        ("260246.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/28S12/260246.tif",
         "fe6a92c5d8dbd05fe7c0dba1828a9ef6114ecad656ebd98be58456796d04a933", 532570),
        ("1044194859.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/28S12W/1044194859.las",
         "b99df78ae35c7ec166d7e138059fceb899afdfe3bc2127ad2f7eb9d289d294d4", 576429),
    ]},
    "Hall_D_4": {"folder": "Hall_D_4_15-051-05370", "name": "Hall 'D' 4", "files": [
        ("193004.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/W/12S17/193004.tif",
         "b4a50bc606f1bed3a9dc2202138be30a6d953f96ef7260f160fa9908f7d75225", 450050),
        ("193005.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/W/12S17/193005.tif",
         "709abdf3cc3a1d6a3fed00ad4a1d3a1ac5bd8f358fade3cf3534d6dc06346c37", 777266),
        ("1043914416.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/12S17W/1043914416.las",
         "65fce66c92b53130c8cb92b513f34c76bd31403b3d99ee1be4f82d8c9e805e5c", 257134),
    ]},
    "Minzer_1": {"folder": "Minzer_1_15-077-30177", "name": "Minzer 1", "files": [
        ("171662.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/W/34S9/171662.tif",
         "886fdad797d3d1f2602dc19dbdc6adc6b285b06fc58ca65dc5f92649088cb2cc", 730298),
        ("171665.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/W/34S9/171665.tif",
         "a896914dcd14076df0a642c8478ef083b74c63f9eadd77613ee3ba4c69793bf9", 169340),
        ("1044076084.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/34S09W/1044076084.las",
         "bb959f909daf5ed134fe1df9989b4156c0076fa1f17d0a1ffd77c7d285b4af74", 463568),
    ]},
    "MAKEMSON_A_1": {"folder": "MAKEMSON_A_1_15-081-20018", "name": "MAKEMSON 'A' 1", "files": [
        ("225991.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/30S34/225991.tif",
         "5acd9f7f2f1633a934426bf678655f7e23c163d6f3f1705da3a34243c0494a2a", 1011668),
        ("225992.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/30S34/225992.tif",
         "a93100d0db14d9aa01f4a3cfef258f1f4a393c08e23e4a4c8107350b0f8fae52", 751566),
        ("225994.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/30S34/225994.tif",
         "e9b16ba08d71055e639ec0776745b85f92c3451811d0e920f4790b68608c4a9f", 408182),
        ("1044080517.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/30S34W/1044080517.las",
         "ba5e08fc6702cf746e33475d2a4353855d9d2e7f7a317dcbd1fb5a7c487def34", 229537),
    ]},
    "ELY_1": {"folder": "ELY_1_15-135-19136", "name": "ELY 1", "files": [
        ("341411.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/19S21/341411.tif",
         "487870d4c85a6e7094efd11076589784912117aeffd81766e9cdddb7b3fbdaaa", 804438),
        ("341412.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/19S21/341412.tif",
         "62bbfb471a6cfdc461dca84de97cea655856f97077e2a945f27ea4f1ff9634ca", 295996),
        ("341413.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/19S21/341413.tif",
         "849ca48df2c6564242b7679dc9225f16b9525b6b9f9a3d8de74915b0f37510ba", 186674),
        ("1006347170.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/19S21W/1006347170.las",
         "a43f9d4b975c83a4bcaa7e54e7f83f4b988d43c23f99847cdd376c6e956d797d", 65013),
    ]},
    "George_B_Strong_8": {"folder": "George_B_Strong_Unit_8_15-175-20098", "name": "George B. Strong Unit 8", "files": [
        ("226280.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/31S34/226280.tif",
         "90d02f298008bfbfdb25c1b0953f46375eb6ab151b72037c59c9a092760a197f", 1137082),
        ("226282.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/31S34/226282.tif",
         "5169233d6ee0fc5a8bb6ee794a8274a1b58188711af1d9ab2922b5eafa9e2ba6", 439730),
        ("226283.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/31S34/226283.tif",
         "f896b7f7533c303e03775935d5e5daf3d9c602bfc1ba74c36fa2432c7fe037fb", 305070),
        ("1044080524.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/31S34W/1044080524.las",
         "5eb4bdb570af2cc527f55d4b1cacf278818c996026e677316193a533c8a7d3b8", 373408),
    ]},
    "Swanson_1": {"folder": "Swanson_1_15-117-20008", "name": "Swanson 1", "files": [
        ("277930.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/E/3S10/277930.tif",
         "c24668ac982af7e779c73528d006369ddc67a000d8df5f5cd3c999bbffc9f21f", 338330),
        ("277931.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/E/3S10/277931.tif",
         "7395df141cbb6ff158b0908a3d9123cc1605d1948992aa420a3838f4b56b5ee7", 393438),
        ("1044077634.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/03S10E/1044077634.las",
         "2a824b4e86ce762648fabf3249639c861fb0e90637fd905cc676be91b28de0a7", 94777),
    ]},
    "WEATHERFORD_B_1": {"folder": "WEATHERFORD_B_1_15-129-20464", "name": "WEATHERFORD 'B' 1", "files": [
        ("194289.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/W/33S43/194289.tif",
         "6787ff06321d5c854d39e32e74927dda2e95d4c91034fff4ec2d40d6ee2efd1a", 2603850),
        ("194290.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/W/33S43/194290.tif",
         "b7803f8881cd47977fb8223ac5e1cbf603f7b5fcbf12e5c75986e02a618361d6", 4121160),
        ("1001178972.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/33S43W/1001178972.las",
         "e860433969330fa97ebfdde372f935f4e05cc3e521d2190a39c6995ebaea0b6b", 957647),
    ]},
    "GOHEEN_UNIT_1": {"folder": "GOHEEN_UNIT_1_15-191-30175", "name": "GOHEEN UNIT 1", "files": [
        ("171201.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/E/30S1/171201.tif",
         "99c05975acb7824d9f72b85ac968fb9bccbacb9bc86fe5151b13f018e933024b", 697220),
        ("171202.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/E/30S1/171202.tif",
         "708b49e346af05c7fb379112e68011eae852c055d73cd6db9822e085e9828749", 819558),
        ("171203.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/E/30S1/171203.tif",
         "d1c4c3fcb662614259d3685e4d2adc80322430b5a8248300e26ed6e08592b760", 784498),
        ("171204.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/E/30S1/171204.tif",
         "91688bbd4c8e3bea23f0bd98e4bd98591ea261ccb1d9d96cf95998fd0b2e244b", 332550),
        ("1042553292.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/30S01E/1042553292.las",
         "a29725515e0c0923efef8fed0362c97b35fd7685d8cbe56eaa8d41d5851a3a90", 124083),
    ]},
    "BACASTOW_1": {"folder": "BACASTOW_1_15-035-23415", "name": "BACASTOW 1", "files": [
        ("169723.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/E/35S4/169723.tif",
         "4542d2dc18e33cbb9225e741af7d17c519ef5fa8d932c6f5bc43f089a4e4b69c", 696742),
        ("169724.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/E/35S4/169724.tif",
         "4d44bd681cc32f5a76e1bcfcff266f78078c8e5ae5e6714e1e3c7f2a4ea41318", 2330750),
        ("169725.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/E/35S4/169725.tif",
         "2f23f94f3fc9f1e9e02d97d67e8ebe5cd8fb9228293254fd5df216841971475b", 4662720),
        ("169726.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog4/E/35S4/169726.tif",
         "3898e14fb22ca59fa4751412e9c9df5506d3a60e0219b36eb396709443c39018", 3164754),
        ("1043694739.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/35S04E/1043694739.las",
         "00916f59abdfa9d8d0c892b150111b65ebe1ae6ff1238b61fceb3062047bf3e4", 332151),
    ]},
    "Calbeck_6": {"folder": "Calbeck_6_15-151-10821", "name": "Calbeck 6", "files": [
        ("278167.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/26S13/278167.tif",
         "3aeeaefadc2c7ccd7341edf4c47cbde3cfe63131483fa852cdfa7d27c1c78c76", 449088),
        ("278169.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/W/26S13/278169.tif",
         "1d369b7d7d93bd9c4ea9afded81d3cfcb430bdd3401d0f34282e376c3de86de7", 317494),
    ]},
    "Rising_1": {"folder": "Rising_1_15-035-21757", "name": "Rising 1", "files": [
        ("208461.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/E/33S5/208461.tif",
         "d16a2b9b25fe4cfc974d94682531868bf57c7bf2bc1b45c607d10f0af454bc14", 1597944),
        ("208462.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/E/33S5/208462.tif",
         "33935f1e9bf71e4e28d6755ffda439e43c4fb333fc2ca17c84dad5c58d8c5773", 1136120),
    ]},
    "STONE_5-19": {"folder": "STONE_5-19_15-055-22611", "name": "STONE #5-19", "files": [
        ("600699.tif", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/Elog3/kcc_logs_2024/600699.tif",
         "2499729bcc367fc78ced1cb20bc35937faa05c19a200d083d7fa2857e7be37a1", 11061602),
        ("1055571405.las", "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/WellLogs/kcc_logs_2024/1055571405.las",
         "3539cd0b39ede919e88117d7717d1900420cf37367c0dfd74738d59df827a269", 2820909),
    ]},
}


def cache_dir() -> Path:
    """Root of the download cache: ``<cache>/pylogdigitizer/kgs``."""
    base = os.environ.get("PYLOGDIGITIZER_CACHE") or Path(
        os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "pylogdigitizer"
    return Path(base) / "kgs"


def _key(well: str) -> str:
    if well in KGS_EXAMPLES:
        return well
    for k, v in KGS_EXAMPLES.items():
        if well == v["folder"]:
            return k
    raise KeyError(f"unknown KGS example well {well!r}; known: {', '.join(KGS_EXAMPLES)}")


def kgs_example_files(well: str) -> list[tuple[str, str, str, int]]:
    """(name, url, sha256, size) of every file listed for ``well``."""
    return list(KGS_EXAMPLES[_key(well)]["files"])


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str, timeout: float = 120.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as r:
            shutil.copyfileobj(r, out)
        got = _sha256(Path(tmp))
        if got != sha256:
            raise IOError(f"checksum mismatch for {url}: expected {sha256}, got {got}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def fetch_kgs_example(well: str, files: list[str] | None = None, cache: str | Path | None = None,
                      force: bool = False) -> Path:
    """Download (once) the files of one KGS example well and return its cached folder.

    ``files``: subset of file names (default: all scans and the LAS). Files already in the
    cache with the right SHA-256 are not downloaded again; ``force=True`` re-downloads."""
    k = _key(well)
    spec = KGS_EXAMPLES[k]
    root = Path(cache) if cache is not None else cache_dir()
    folder = root / spec["folder"]
    want = set(files) if files else None
    if want:
        missing = want - {f[0] for f in spec["files"]}
        if missing:
            raise KeyError(f"{k}: no such file(s) {sorted(missing)}")
    for name, url, sha, _size in spec["files"]:
        if want is not None and name not in want:
            continue
        dest = folder / name
        if dest.exists() and not force and _sha256(dest) == sha:
            continue
        _download(url, dest, sha)
    return folder


def fetch_kgs_examples(wells, files_for=None, cache: str | Path | None = None) -> Path:
    """Fetch several wells; returns the cache root, usable as ``data_root`` for
    :mod:`pylogdigitizer.ml_training_data` / :mod:`pylogdigitizer.ml_batch_digitize`.
    ``files_for``: optional ``{well: [file names]}`` to limit downloads."""
    files_for = files_for or {}
    root = Path(cache) if cache is not None else cache_dir()
    for w in wells:
        fetch_kgs_example(w, files=files_for.get(w), cache=root)
    return root
