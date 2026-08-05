"""Real-time + TinEye-model tests:

1. verified web results are cached per query-image hash — a repeat search of
   the same image is served from memory (no fresh provider crawl, cached=true)
2. local index copies expose their accumulated source_urls (\"found on N sites\")
3. verified results survive a SearchResult round-trip through the cache
"""

from __future__ import annotations

from app.models import ImageHit

from conftest import make_image_bytes, patch_image_fetch


def test_verified_web_results_are_cached_and_reused(client, ctx, monkeypatch):
    """First search crawls the web; the second search of the SAME image is
    served from the verified-results cache — no fresh provider call, and the
    response says cached=true (real-time repeat search)."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["searx"].reverse_hits = [
        ImageHit(title="Elon Musk", url="https://page.example/x", img_src="https://img.example/musk.jpg", engine="bing")
    ]
    raw = make_image_bytes()
    patch_image_fetch(monkeypatch, raw)  # candidate hash-matches the query

    # first search -> real crawl
    d1 = client.post(
        "/v1/search",
        files={"file": ("q.jpg", raw, "image/jpeg")},
        data={"mode": "best", "num_results": "10", "include_web": "true"},
    ).json()
    assert d1["found"] is True
    assert d1["cached"] is False
    assert ctx["searx"].total_requests == 1

    # second search of the identical image -> served from cache, no re-crawl
    requests_before = ctx["searx"].total_requests
    d2 = client.post(
        "/v1/search",
        files={"file": ("q.jpg", raw, "image/jpeg")},
        data={"mode": "best", "num_results": "10", "include_web": "true"},
    ).json()
    assert d2["found"] is True
    assert d2["cached"] is True
    assert "web" in d2["providers"]
    assert ctx["searx"].total_requests == requests_before  # cache hit, no crawl


def test_cache_round_trips_search_results(client, ctx, monkeypatch):
    """Cached verified results survive a SearchResult round-trip (fields the
    UI renders — url/page_url/match_type/resolution — are intact)."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["searx"].reverse_hits = [
        ImageHit(title="Found it", url="https://page.example/p", img_src="https://img.example/pic.jpg", engine="bing")
    ]
    raw = make_image_bytes()
    patch_image_fetch(monkeypatch, raw)
    client.post(
        "/v1/search",
        files={"file": ("q.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "true"},
    )
    d = client.post(
        "/v1/search",
        files={"file": ("q.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "true"},
    ).json()
    assert d["cached"] is True
    r = d["results"][0]
    assert r["url"] == "https://img.example/pic.jpg"
    assert r["page_url"] == "https://page.example/p"
    assert r["match_type"] in ("exact", "near")
    assert r["source"] == "web"


def test_cache_ttl_expiry_forces_fresh_crawl(client, ctx, monkeypatch):
    """Expired cache is dropped — the next search crawls the web again."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["settings"].web_results_cache_ttl_seconds = 0  # cache is instantly stale
    ctx["searx"].reverse_hits = [
        ImageHit(title="Elon", url="https://page.example/x", img_src="https://img.example/m.jpg", engine="bing")
    ]
    raw = make_image_bytes()
    patch_image_fetch(monkeypatch, raw)
    client.post(
        "/v1/search",
        files={"file": ("q.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "true"},
    )
    d = client.post(
        "/v1/search",
        files={"file": ("q.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "true"},
    ).json()
    assert d["cached"] is False  # stale cache dropped, fresh crawl happened
    assert ctx["searx"].total_requests == 2


def test_local_match_exposes_source_urls(client, ctx, monkeypatch):
    """An indexed copy that has accumulated verified source links surfaces
    them as remote links (TinEye \"found on N sites\") so the UI can show the
    removal-request list without re-crawling."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    raw = make_image_bytes()
    from conftest import index_image

    image_id = index_image(ctx["repo"], ctx["settings"], raw, url="https://orig.example/one.jpg")
    ctx["repo"].add_source_link(image_id, "https://mirror.example/two.jpg", "https://page.example/a", "site A")
    ctx["repo"].add_source_link(image_id, "https://cdn.example/three.jpg", "https://page.example/b", "site B")

    d = client.post(
        "/v1/search",
        files={"file": ("q.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "false"},
    ).json()
    assert d["found"] is True
    r = d["results"][0]
    assert r["source_count"] == 2  # the 2 verified source links
    assert "https://mirror.example/two.jpg" in r["source_urls"]
    assert "https://cdn.example/three.jpg" in r["source_urls"]
    assert r["source_urls"]  # TinEye "found on N sites" list surfaced to the UI
