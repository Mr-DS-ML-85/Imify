"""Perceptual feature extraction for Imify.

Pure numpy + Pillow — no scipy, no external API. Everything needed to turn an
image into a content fingerprint:

- ``p_hash``  — 64-bit DCT hash (the primary near-duplicate signal)
- ``d_hash``  — 64-bit gradient hash
- ``a_hash``  — 64-bit average hash
- ``color_histogram`` — 64-dim L2-normalized HSV histogram (color matching)

Key trick: a 64-bit hash encoded as 64 floats of ±1 turns Hamming distance
into cosine distance (cosine_dist = hamming/32). That means StrikeDB's
``VSEARCH`` ANN does near-duplicate search directly — sub-millisecond at
million-image scale.
"""

from __future__ import annotations

import hashlib
import io
from typing import Iterable

import numpy as np
from PIL import Image

PHASH_SIZE = 8
HASH_BITS = 64
COLOR_DIM = 64  # 8 hue x 4 sat x 2 val
_HUE_BINS, _SAT_BINS, _VAL_BINS = 8, 4, 2


def _dct_matrix(n: int) -> np.ndarray:
    """DCT-II transform matrix for a length-n signal (separable 2D via matmul)."""
    k = np.arange(n, dtype=np.float64)
    u = np.arange(n, dtype=np.float64)
    mat = np.cos((2 * k + 1)[None, :] * u[:, None] * np.pi / (2 * n))
    mat[0, :] /= np.sqrt(2)
    return mat / np.sqrt(n)


_DCT_CACHE: dict[int, np.ndarray] = {}


def dct2d(pixels: np.ndarray) -> np.ndarray:
    n = pixels.shape[0]
    mat = _DCT_CACHE.get(n)
    if mat is None:
        mat = _dct_matrix(n)
        _DCT_CACHE[n] = mat
    return mat @ pixels @ mat.T


def _bits_to_int(bits: Iterable[bool]) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def _gray_resized(img: Image.Image, w: int, h: int) -> np.ndarray:
    return np.asarray(img.convert("L").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float64)


def p_hash(img: Image.Image, hash_size: int = PHASH_SIZE, highfreq_factor: int = 4) -> int:
    """64-bit perceptual hash: low-frequency DCT coefficients vs their median."""
    img_size = hash_size * highfreq_factor  # 32
    pixels = _gray_resized(img, img_size, img_size)
    dct = dct2d(pixels)
    lowfreq = dct[:hash_size, :hash_size]
    med = np.median(lowfreq)
    return _bits_to_int((lowfreq > med).flatten())


def d_hash(img: Image.Image, hash_size: int = PHASH_SIZE) -> int:
    """64-bit difference hash: horizontal gradient of a 9x8 grayscale image."""
    pixels = _gray_resized(img, hash_size + 1, hash_size)
    return _bits_to_int((pixels[:, :-1] > pixels[:, 1:]).flatten())


def a_hash(img: Image.Image, hash_size: int = PHASH_SIZE) -> int:
    """64-bit average hash: compare each pixel to the mean."""
    pixels = _gray_resized(img, hash_size, hash_size)
    return _bits_to_int((pixels > pixels.mean()).flatten())


def color_histogram(img: Image.Image) -> list[float]:
    """64-dim L2-normalized HSV histogram (8 hue x 4 sat x 2 val)."""
    rgb = np.asarray(img.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    if mask.any():
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        # delta_safe = 1 where mask is False, so no NaN ever reaches the casts.
        delta_safe = np.where(mask, delta, 1.0)
        h1 = ((g - b) / delta_safe) % 6.0
        h2 = (b - r) / delta_safe + 2.0
        h3 = (r - g) / delta_safe + 4.0
        hh = np.where(mask & (maxc == r), h1, np.where(mask & (maxc == g), h2, np.where(mask, h3, 0.0)))
        hue = hh / 6.0

    sat = np.where(maxc > 0, delta / np.maximum(maxc, 1e-9), 0.0)
    val = maxc

    hb = np.clip((hue * _HUE_BINS).astype(np.int64), 0, _HUE_BINS - 1)
    sb = np.clip((sat * _SAT_BINS).astype(np.int64), 0, _SAT_BINS - 1)
    vb = np.clip((val * _VAL_BINS).astype(np.int64), 0, _VAL_BINS - 1)
    flat = hb * (_SAT_BINS * _VAL_BINS) + sb * _VAL_BINS + vb

    hist = np.bincount(flat.ravel(), minlength=COLOR_DIM).astype(np.float64)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist.tolist()


def encode_hash_vector(hash64: int) -> list[float]:
    """Encode a 64-bit hash as 64 floats of ±1 so cosine distance == hamming/32."""
    return [1.0 if (hash64 >> (63 - i)) & 1 else -1.0 for i in range(64)]


def decode_hamming_from_distance(distance: float) -> float:
    """Invert VSEARCH cosine distance back to Hamming distance for ±1 vectors."""
    return float(np.clip(round(distance * 32.0), 0, 64))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_features(img: Image.Image, raw: bytes | None = None) -> dict:
    """Extract the full FeatureSet dict from a decoded PIL image."""
    from app.models import FeatureSet  # local import to avoid cycles

    fs = FeatureSet(
        phash=p_hash(img),
        dhash=d_hash(img),
        ahash=a_hash(img),
        color=color_histogram(img),
        sha256=sha256_bytes(raw) if raw is not None else "",
        width=img.width,
        height=img.height,
        size_bytes=len(raw) if raw is not None else 0,
    )
    return fs.to_dict()


def decode_image(raw: bytes) -> Image.Image | None:
    """Decode image bytes safely; returns None for non-images / corrupt data.

    Oversized / decompression-bomb images are rejected BEFORE full decode so a
    single huge CDN answer (or a crafted zip-of-pixels) cannot balloon memory
    (a search verifies many candidates concurrently → a few giant images was
    enough to OOM the whole 29G rig).
    """
    try:
        # Peek at the header dimensions without decoding the full pixel grid.
        with Image.open(io.BytesIO(raw)) as probe:
            w, h = probe.size
            if probe.format == "PNG" and (w > 16384 or h > 16384):
                return None
            area = w * h
            if area > 32_000_000:  # ~32 megapixels cap; decodes are bounded
                return None
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert("RGB") if img.mode not in ("RGB", "L", "RGBA") else img
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        return img
    except Exception:
        return None
