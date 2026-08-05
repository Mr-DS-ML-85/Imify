from __future__ import annotations

import io
import time

from PIL import Image

from app.features import decode_image, encode_hash_vector, extract_features
from app.models import ImageMeta

from conftest import index_image, make_image_bytes


def _query_features(raw: bytes) -> dict:
    img = decode_image(raw)
    return extract_features(img, raw)


def test_near_duplicate_found(ctx):
    repo, matcher = ctx["repo"], ctx["matcher"]
    original = make_image_bytes(mode="photo")
    # recompress at lower quality + tiny resize via PIL
    img = Image.open(io.BytesIO(original)).resize((140, 100))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=40)
    scaled = buf.getvalue()

    id_a = index_image(repo, ctx["settings"], original, url="https://a.com/orig.jpg", title="original")
    id_b = index_image(repo, ctx["settings"], scaled, url="https://b.com/scaled.jpg", title="scaled copy")

    results = matcher.search_local(_query_features(original), mode="best", limit=20)
    ids = [r.image_id for r in results]
    assert id_a in ids
    assert id_b in ids
    best = results[0]
    assert best.hamming is not None and best.hamming <= 8
    assert best.match_type in ("exact", "near")
    assert best.source == "local"


def test_unrelated_image_not_returned(ctx):
    """Texture-bearing images with very different layouts must not match."""
    repo, matcher = ctx["repo"], ctx["matcher"]
    index_image(repo, ctx["settings"], make_image_bytes(mode="gradient", gradient_dir="h"), url="https://x/1.jpg")
    index_image(repo, ctx["settings"], make_image_bytes(mode="gradient", gradient_dir="v"), url="https://x/2.jpg")
    query = make_image_bytes(mode="photo")
    results = matcher.search_local(_query_features(query), mode="best", limit=20)
    assert all(r.hamming > 16 for r in results)  # nothing close


def test_exact_duplicate_merged_phash(ctx):
    repo, matcher = ctx["repo"], ctx["matcher"]
    raw = make_image_bytes()
    id1 = index_image(repo, ctx["settings"], raw, url="https://one.com/x.jpg")
    id2 = index_image(repo, ctx["settings"], raw, url="https://two.com/x.jpg")
    assert id1 != id2  # byte-identical -> different ids via test helper
    # but the matcher must return both as exact matches
    results = matcher.search_local(_query_features(raw), mode="exact", limit=20)
    ids = {r.image_id for r in results}
    assert {id1, id2} <= ids
    for r in results:
        assert r.match_type == "exact"


def test_color_match_mode(ctx):
    repo, matcher = ctx["repo"], ctx["matcher"]
    index_image(repo, ctx["settings"], make_image_bytes(mode="solid", color=(40, 90, 200)), url="https://c/blue.jpg")
    index_image(repo, ctx["settings"], make_image_bytes(mode="solid", color=(210, 45, 95)), url="https://c/red.jpg")
    results = matcher.search_color(_query_features(make_image_bytes(mode="solid", color=(45, 95, 205))), limit=10)
    assert results
    assert results[0].url == "https://c/blue.jpg"
    assert results[0].match_type == "color"


def test_biggest_and_newest_sorting(ctx):
    repo, matcher = ctx["repo"], ctx["matcher"]
    base = make_image_bytes(mode="photo")
    big = ImageMeta(
        id=repo.next_image_id(), url="https://s/big.jpg", phash=0xAAAAAAAAAAAAAAAA, color=[0.0] * 64,
        width=4000, height=3000, first_seen=time.time(), last_seen=time.time(),
    )
    small = ImageMeta(
        id=repo.next_image_id(), url="https://s/small.jpg", phash=0xAAAAAAAAAAAAAAAA, color=[0.0] * 64,
        width=400, height=300, first_seen=time.time() - 100, last_seen=time.time() - 100,
    )
    for m in (big, small):
        repo.db.vadd(m.id, encode_hash_vector(m.phash))
        repo.put_image(m)
        repo.register_url(m.url, m.id)

    q = {"phash": 0xAAAAAAAAAAAAAAAA, "color": [0.0] * 64}
    biggest = matcher.search_local(q, mode="biggest", limit=10)
    assert biggest[0].url == "https://s/big.jpg"
    newest = matcher.search_local(q, mode="newest", limit=10)
    assert newest[0].url == "https://s/big.jpg"  # last_seen newer


def test_fusion_dedupes_and_prefers_local(ctx):
    from app.matcher import fuse_results
    from app.models import SearchResult

    local = [SearchResult(url="https://site.com/a/b.jpg", source="local", score=0.9, match_type="exact")]
    web = [SearchResult(url="https://www.SITE.com/a/b.jpg", source="web", score=0.4)]
    fused = fuse_results(local, web)
    assert len(fused) == 1
    assert fused[0].source == "both"
    assert fused[0].url == "https://site.com/a/b.jpg"
