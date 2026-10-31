# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""scan_io: Load raster/PDF scans of well logs (TIFF via tifffile/Pillow), tiling and downsampling for viewers."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Scan:
    """A loaded log scan: grayscale uint8 array (0=black ink, 255=paper) plus metadata."""
    path: Path
    image: np.ndarray
    dpi: tuple[float, float] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]

def _pil_open(path: Path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(path)

def read_info(path: str | Path) -> dict:
    """Return width, height, mode and dpi without decoding all pixels."""
    im = _pil_open(Path(path))
    return {"width": im.size[0], "height": im.size[1], "mode": im.mode, "dpi": im.info.get("dpi")}

def load_scan(path: str | Path, crop: tuple[int, int, int, int] | None = None) -> Scan:
    """Load a scan (optionally a crop box left, top, right, bottom) as grayscale uint8."""
    path = Path(path)
    im = _pil_open(path)
    dpi = im.info.get("dpi")
    if crop is not None:
        im = im.crop(crop)
    arr = np.asarray(im.convert("L"), dtype=np.uint8)
    return Scan(path=path, image=arr, dpi=tuple(dpi) if dpi else None)

def load_ink(path: str | Path, crop: tuple[int, int, int, int] | None = None, threshold: int = 128) -> np.ndarray:
    """Load a scan (or crop box left, top, right, bottom) as a boolean ink mask (True = dark).
    Bilevel TIFFs are read without a uint8 copy, so large scans stay memory-friendly."""
    im = _pil_open(Path(path))
    if crop is not None:
        im = im.crop(crop)
    if im.mode == "1":
        return ~np.asarray(im, dtype=bool)
    return np.asarray(im.convert("L"), dtype=np.uint8) < threshold

def downsample(scan: Scan, factor: int) -> np.ndarray:
    """Block-mean downsample by an integer factor (preview / overview images)."""
    h, w = scan.shape
    a = scan.image[: h - h % factor, : w - w % factor].astype(np.float32)
    return a.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)).astype(np.uint8)

def save_png(array: np.ndarray, out: str | Path) -> Path:
    """Write an array to PNG (for previews)."""
    from PIL import Image
    Image.fromarray(array).save(out)
    return Path(out)

def to_dzi(path: str | Path, out_dir: str | Path, tile_size: int = 512) -> Path:
    """Write a Deep Zoom (DZI) tile pyramid for OpenSeadragon."""
    raise NotImplementedError
