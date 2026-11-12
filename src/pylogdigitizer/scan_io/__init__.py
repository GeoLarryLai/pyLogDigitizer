# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""scan_io: Load raster/PDF scans of well logs (TIFF via Pillow), tiling and downsampling for viewers."""

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
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        rgb = pdf_embedded_image(path)
        return {"width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "mode": "RGB", "dpi": None}
    im = _pil_open(path)
    return {"width": im.size[0], "height": im.size[1], "mode": im.mode, "dpi": im.info.get("dpi")}

def load_scan(path: str | Path, crop: tuple[int, int, int, int] | None = None) -> Scan:
    """Load a scan (optionally a crop box left, top, right, bottom) as grayscale uint8."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        rgb = load_rgb(path, crop)
        from PIL import Image as _Im
        arr = np.asarray(_Im.fromarray(rgb).convert("L"), dtype=np.uint8)
        return Scan(path=path, image=arr, dpi=None)
    im = _pil_open(path)
    dpi = im.info.get("dpi")
    if crop is not None:
        im = im.crop(crop)
    arr = np.asarray(im.convert("L"), dtype=np.uint8)
    return Scan(path=path, image=arr, dpi=tuple(dpi) if dpi else None)

def load_ink(path: str | Path, crop: tuple[int, int, int, int] | None = None, threshold: int = 128) -> np.ndarray:
    """Load a scan (or crop box left, top, right, bottom) as a boolean ink mask (True = dark).
    Bilevel TIFFs are read without a uint8 copy, so large scans stay memory-friendly."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        rgb = load_rgb(path, crop)
        from PIL import Image as _Im
        return np.asarray(_Im.fromarray(rgb).convert("L"), dtype=np.uint8) < threshold
    im = _pil_open(path)
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


def load_rgb(path: str | Path, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Load a scan (TIFF / JPEG / PNG, or a raster PDF page) as an RGB uint8 array."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        rgb = pdf_embedded_image(path)
    else:
        im = _pil_open(path)
        if crop is not None:
            im = im.crop(crop)
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return rgb
    if crop is not None:
        l, u, r, b = crop
        rgb = rgb[u:b, l:r]
    return rgb


def pdf_embedded_image(pdf_path, page: int = 0) -> np.ndarray:
    """The scan stored inside a raster PDF page, as an RGB array (lossless, native resolution).

    Vector (digital-print) PDFs have no embedded image — use
    :func:`pylogdigitizer.pdf_vector.digitize_pdf` for those.
    """
    from pypdf import PdfReader
    pg = PdfReader(str(pdf_path)).pages[page]
    imgs = [i.image for i in pg.images]
    if not imgs:
        raise ValueError(f"{Path(pdf_path).name} page {page}: no embedded image. This looks like a "
                         "vector (digital) PDF — use pylogdigitizer.pdf_vector.digitize_pdf for it instead.")
    im = max(imgs, key=lambda m: m.size[0] * m.size[1])
    return np.asarray(im.convert("RGB"))


def is_color_scan(path: str | Path) -> bool:
    """True when ``path`` decodes to a chromatic RGB image (not a 1-bit / gray scan)."""
    from pylogdigitizer.scan_color import is_color_rgb
    path = Path(path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return is_color_rgb(load_rgb(path))
    if path.suffix.lower() == ".pdf":
        try:
            return is_color_rgb(pdf_embedded_image(path))
        except ValueError:
            return False
    im = _pil_open(path)
    if im.mode in ("1", "L", "LA", "I;16"):
        return False
    if im.mode == "P":
        pal = im.getpalette() or []
        used = np.unique(np.asarray(im)[::8, ::8])
        for i in used:
            i = int(i)
            if i * 3 + 2 >= len(pal):
                continue
            c = (pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2])
            if max(c) - min(c) >= 18 and min(c) < 200:
                return True
        return False
    return is_color_rgb(np.asarray(im.convert("RGB")))

