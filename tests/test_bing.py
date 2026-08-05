"""Bing Visual Search provider tests: parser + byte-upload routing."""

from __future__ import annotations

import pytest

from app.engines import EngineError
from app.engines.bing_images import BingImagesReverse, parse_bing_results

# Real-world sample captured from Bing Visual Search (HTML-escaped JSON attrs)
SAMPLE_BING = """
<div class="iusc" data-m="{&quot;murl&quot;:&quot;https://i.pinimg.com/originals/c6/6f/f1/x.jpg&quot;,&quot;turl&quot;:&quot;https://tse1.mm.bing.net/th?id=abc&quot;,&quot;purl&quot;:&quot;https://www.artofit.org/gallery/1/&quot;}">
<div class="iusc" data-m="{&quot;murl&quot;:&quot;https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=716&quot;,&quot;purl&quot;:&quot;https://www.facebook.com/HairLounge/&quot;}">
<div class="iusc" data-m="{&quot;murl&quot;:&quot;https://img.example.com/data:bad.jpg&quot;,&quot;purl&quot;:&quot;https://x.com/&quot;}">
"""


def test_parse_bing_extracts_murl_purl_pairs():
    hits = parse_bing_results(SAMPLE_BING)
    urls = {h.img_src for h in hits}
    assert "https://i.pinimg.com/originals/c6/6f/f1/x.jpg" in urls
    assert "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=716" in urls
    # data: URLs and duplicates never pass through
    assert not any(h.img_src.startswith("data:") for h in hits)
    # page urls paired
    by_url = {h.img_src: h.url for h in hits}
    assert by_url["https://i.pinimg.com/originals/c6/6f/f1/x.jpg"] == "https://www.artofit.org/gallery/1/"
    # engine tag
    assert all(h.engine == "bing" for h in hits)


def test_parse_bing_plain_json_fallback():
    page = '<script>{"murl":"https://cdn.example.com/a.jpg","purl":"https://page.example.com/a"}</script>'
    hits = parse_bing_results(page)
    assert len(hits) == 1
    assert hits[0].img_src == "https://cdn.example.com/a.jpg"
    assert hits[0].url == "https://page.example.com/a"


def test_parse_bing_dedupes():
    page = SAMPLE_BING + SAMPLE_BING  # repeated blocks
    hits = parse_bing_results(page)
    urls = [h.img_src for h in hits]
    assert len(urls) == len(set(urls))


def test_bing_upload_via_tunnel(monkeypatch):
    captured = {}

    def fake_post(url, data, timeout, follow_redirects, proxy, headers):
        captured.update(url=url, data=data, proxy=proxy, headers=headers)

        class R:
            def raise_for_status(self):
                pass

            status_code = 200
            text = SAMPLE_BING

        return R()

    monkeypatch.setattr("app.engines.bing_images.httpx.post", fake_post)
    b = BingImagesReverse(proxify_fetch_url="", fetch_proxy_url="http://127.0.0.1:8888")
    raw = b"\xff\xd8fakejpeg\x00\x01\xfe"
    hits = b.search_upload(raw, filename="q.jpg")

    assert captured["url"] == "https://www.bing.com/images/search"
    assert captured["proxy"] == "http://127.0.0.1:8888"
    # base64 of the raw bytes travels as the image_url form field
    import base64

    assert captured["data"]["image_url"] == base64.b64encode(raw).decode("ascii")
    assert any(h.img_src == "https://i.pinimg.com/originals/c6/6f/f1/x.jpg" for h in hits)


def test_bing_upload_direct_fallback(monkeypatch):
    """No tunnel configured -> direct POST as last resort (residential IPs)."""
    captured = {}

    def fake_post(url, data, timeout, follow_redirects, proxy, headers):
        captured.update(proxy=proxy, data=data)

        class R:
            def raise_for_status(self):
                pass

            status_code = 200
            text = SAMPLE_BING

        return R()

    monkeypatch.setattr("app.engines.bing_images.httpx.post", fake_post)
    b = BingImagesReverse(proxify_fetch_url="", fetch_proxy_url="")
    hits = b.search_upload(b"\x01\x02\x03", filename="q.jpg")
    assert captured["proxy"] is None
    assert len(hits) >= 1


def test_bing_upload_all_paths_fail_raises(monkeypatch):
    def fake_post(url, data, timeout, follow_redirects, proxy, headers):
        raise RuntimeError("network down")

    monkeypatch.setattr("app.engines.bing_images.httpx.post", fake_post)
    b = BingImagesReverse(proxify_fetch_url="", fetch_proxy_url="")
    with pytest.raises(EngineError, match="bing-upload failed via all paths"):
        b.search_upload(b"\x01")


def test_bing_url_reverse_requires_proxify():
    b = BingImagesReverse(proxify_fetch_url="", fetch_proxy_url="")
    with pytest.raises(EngineError, match="PROXIFY_FETCH_URL"):
        b.search("https://public.example/x.jpg")
