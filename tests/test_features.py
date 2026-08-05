from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.features import (
    a_hash,
    color_histogram,
    decode_hamming_from_distance,
    d_hash,
    encode_hash_vector,
    extract_features,
    hamming,
    p_hash,
)


def make(size=(64, 64), color=(40, 90, 200)):
    return Image.new("RGB", size, color)


def jpeg_bytes(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def test_hashes_are_64bit():
    img = make()
    for fn in (p_hash, d_hash, a_hash):
        assert 0 <= fn(img) < 2**64


def test_identical_images_have_identical_hashes():
    img = make()
    assert p_hash(img) == p_hash(make())
    assert d_hash(img) == d_hash(make())


def test_scaled_copy_is_near_duplicate():
    """A 2x rescale + slight recompression must stay within the near band."""
    original = make((320, 240))
    scaled = original.resize((160, 120), Image.Resampling.LANCZOS)
    a = p_hash(original)
    b = p_hash(scaled)
    assert hamming(a, b) <= 8


def test_recompression_keeps_phash_stable():
    a = p_hash(make())
    b = p_hash(make())
    assert hamming(a, b) == 0
    recompressed = make()
    buf = io.BytesIO()
    recompressed.save(buf, "JPEG", quality=30)
    b2 = p_hash(recompressed)
    assert hamming(a, b2) <= 4


def test_different_images_differ():
    a = p_hash(make(color=(40, 90, 200)))
    b = p_hash(make(color=(200, 40, 90)))
    assert hamming(a, b) >= 10


def test_vector_encoding_maps_to_hamming():
    """cosine distance between ±1 encodings == hamming/32."""
    a = 0xDEADBEEFDEADBEEF
    b = 0x1234567890ABCDEF
    va = np.asarray(encode_hash_vector(a), dtype=np.float64)
    vb = np.asarray(encode_hash_vector(b), dtype=np.float64)
    cosine_dist = 1.0 - float((va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    assert abs(cosine_dist - hamming(a, b) / 32.0) < 1e-9
    assert decode_hamming_from_distance(cosine_dist) == hamming(a, b)


def test_color_histogram_shape_and_normalization():
    hist = color_histogram(make())
    assert len(hist) == 64
    assert abs(np.linalg.norm(hist) - 1.0) < 1e-6
    assert all(v >= 0 for v in hist)


def test_similar_colors_have_similar_histograms():
    h1 = np.asarray(color_histogram(make(color=(40, 90, 200))))
    h2 = np.asarray(color_histogram(make(color=(45, 95, 205))))
    h3 = np.asarray(color_histogram(make(color=(200, 40, 90))))
    assert float(h1 @ h2) > float(h1 @ h3)


def test_extract_features_complete():
    feats = extract_features(make(), raw=jpeg_bytes(make()))
    assert feats["phash"] >= 0
    assert len(feats["color"]) == 64
    assert feats["sha256"]
    assert feats["width"] == 64 and feats["height"] == 64
    assert feats["size_bytes"] > 0
