"""Google Images reverse provider (proxify-only) + reverse chain tests."""

from __future__ import annotations

import pytest

from app.engines import EngineError, run_reverse_chain
from app.engines.google_images import GoogleImagesReverse, parse_google_results
from app.models import ImageHit

SAMPLE_GOOGLE_PAGE = """
<html><body>
<script>
var d = [{"ou":"https://cdn.example.com/photo1.jpg","ru":"https://blog.example.com/posts/1","pt":"First match"},
         {"ou":"https://img.example.net/photo2.png","ru":"https://news.example.org/a/b","pt":"Second match"}];
</script>
<a href="/imgres?imgurl=https://legacy.example.com/x.jpeg&imgrefurl=https://legacy.example.com/x.html">legacy</a>
<img src="https://grid.example.com/y.webp">
</body></html>
"""


def test_google_provider_requires_some_transport(monkeypatch):
    """Without proxify, Google works only via curl_cffi (chrome146 impersonation).
    With no transport at all it degrades instead of burning a doomed request."""
    monkeypatch.setattr("app.engines.google_images._impersonating_session", lambda p: None)
    g = GoogleImagesReverse(proxify_fetch_url="", fetch_proxy_url="", direct_upload=False)
    assert g.available is False  # nothing configured -> provider off
    with pytest.raises(EngineError, match="curl_cffi or PROXIFY_FETCH_URL"):
        g._fetch("https://www.google.com/searchbyimage?image_url=x")


def test_google_parses_embedded_json():
    hits = parse_google_results(SAMPLE_GOOGLE_PAGE)
    urls = {h.img_src for h in hits}
    assert "https://cdn.example.com/photo1.jpg" in urls
    assert "https://img.example.net/photo2.png" in urls
    # classic imgres + grid fallback
    assert "https://legacy.example.com/x.jpeg" in urls
    assert "https://grid.example.com/y.webp" in urls
    # data: urls never pass through
    assert not any(h.img_src.startswith("data:") for h in hits)
    # titles paired best-effort
    titled = {h.img_src: h.title for h in hits}
    assert "First match" in titled.get("https://cdn.example.com/photo1.jpg", "")


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.request = None

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeHttpx:
    """Records the exact proxify /fetch request and returns canned payloads."""

    def __init__(self, payload):
        self.payload = payload
        self.calls: list[dict] = []

    def post(self, url, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResp(self.payload)


def test_google_fetch_uses_proxify_contract(monkeypatch):
    """The user's rule: Google ONLY via proxify, with the exact /fetch body."""
    fake = _FakeHttpx({"success": True, "html": "<html>" + SAMPLE_GOOGLE_PAGE + "</html>", "markdown": "ignored"})
    monkeypatch.setattr("app.engines.google_images.httpx.post", fake.post)

    g = GoogleImagesReverse(proxify_fetch_url="http://proxify:8081", use_browser=True)
    hits = g.search("https://public.example/q.jpg")

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"] == "http://proxify:8081/fetch"
    # exact contract: url + timeout + use_browser, pointing at searchbyimage
    assert call["json"]["timeout"] == 45
    assert call["json"]["use_browser"] is True
    assert call["json"]["url"].startswith("https://www.google.com/searchbyimage?image_url=https%3A%2F%2Fpublic.example%2Fq.jpg")
    # html parsed into hits
    assert any(h.img_src == "https://cdn.example.com/photo1.jpg" for h in hits)


def test_google_fetch_prefers_html_over_markdown(monkeypatch):
    fake = _FakeHttpx({"success": True, "html": "", "markdown": "no structured hits here"})
    monkeypatch.setattr("app.engines.google_images.httpx.post", fake.post)
    g = GoogleImagesReverse(proxify_fetch_url="http://proxify:8081")
    assert g.search("https://public.example/q.jpg") == []  # markdown-only -> no parseable hits


def test_google_fetch_success_false_raises(monkeypatch):
    fake = _FakeHttpx({"success": False, "error": "google sorry block", "status_code": 429})
    monkeypatch.setattr("app.engines.google_images.httpx.post", fake.post)
    g = GoogleImagesReverse(proxify_fetch_url="http://proxify:8081")
    with pytest.raises(EngineError, match="proxify fetch failed: google sorry block"):
        g.search("https://public.example/q.jpg")


def test_google_fetch_non_json_degrades(monkeypatch):
    class _BadResp:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("not json")

    def post(url, json, timeout):
        return _BadResp()

    monkeypatch.setattr("app.engines.google_images.httpx.post", post)
    g = GoogleImagesReverse(proxify_fetch_url="http://proxify:8081")
    with pytest.raises(EngineError, match="non-JSON"):
        g.search("https://public.example/q.jpg")


def test_google_upload_requires_proxify():
    g = GoogleImagesReverse(proxify_fetch_url="", fetch_proxy_url="", direct_upload=False)
    with pytest.raises(EngineError, match="failed via all paths"):
        g.search_upload(b"\xff\xd8notreallyjpg")


def test_google_upload_via_fetch_body_b64(monkeypatch):
    """Primary path: proxify GUI Chrome submits the file as a real in-page form."""
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"success": True, "html": "<html>" + SAMPLE_GOOGLE_PAGE + "</html>", "strategy_used": "gui_chrome"}

        return R()

    monkeypatch.setattr("app.engines.google_images._impersonating_session", lambda p: None)
    monkeypatch.setattr("app.engines.google_images.httpx.post", fake_post)
    g = GoogleImagesReverse(proxify_fetch_url="http://127.0.0.1:8081", fetch_proxy_url="", direct_upload=False)
    raw = b"\xff\xd8fakejpeg\x00\x01\xfe"
    hits = g.search_upload(raw, filename="q.jpg", content_type="image/jpeg")

    assert captured["url"] == "http://127.0.0.1:8081/fetch"
    j = captured["json"]
    assert j["url"].startswith("https://lens.google.com/v3/upload?")
    assert j["method"] == "POST"
    # Lens results are bound to the browser session that uploaded: the GUI
    # Chrome must do the upload itself, from the Lens page.
    assert j["force_strategy"] == "gui_chrome"
    assert j["in_page_upload"] is True
    assert j["upload_landing"] == "https://lens.google.com/"
    import base64

    decoded = base64.b64decode(j["body_b64"])
    assert b"encoded_image" in decoded
    assert raw in decoded  # raw binary preserved byte-for-byte through base64
    assert j["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    assert any(h.img_src == "https://cdn.example.com/photo1.jpg" for h in hits)


def test_google_upload_via_impersonated_direct(monkeypatch):
    """Fallback: no proxify configured -> curl_cffi chrome146 impersonation
    (the path that beats Google's TLS-level datacenter block)."""
    captured = {}

    class _FakeSession:
        def __init__(self, *a, **k):
            pass

        def post(self, url, data, headers, allow_redirects):
            captured.update(url=url, data=data, headers=headers, allow_redirects=allow_redirects)

            class R:
                status_code = 200
                text = "<html>" + SAMPLE_GOOGLE_PAGE + "</html>"
                url = "https://www.google.com/search?vsrid=x"

            return R()

    monkeypatch.setattr("app.engines.google_images._impersonating_session", lambda p: _FakeSession())
    g = GoogleImagesReverse(proxify_fetch_url="", fetch_proxy_url="", direct_upload=True)
    raw = b"\xff\xd8fakejpeg\x00\x01\xfe"
    hits = g.search_upload(raw, filename="q.jpg", content_type="image/jpeg")

    assert captured["url"] == "https://lens.google.com/upload"
    assert captured["allow_redirects"] is True
    assert raw in captured["data"]
    assert any(h.img_src == "https://cdn.example.com/photo1.jpg" for h in hits)


def test_google_upload_via_tunnel_fallback(monkeypatch):
    """Fallback: only the CONNECT tunnel configured -> raw bytes through it."""
    captured = {}

    def fake_post(url, content, timeout, follow_redirects, proxy, headers):
        captured.update(url=url, content=content, proxy=proxy, headers=headers)

        class R:
            def raise_for_status(self):
                pass

            status_code = 200
            text = "<html>" + SAMPLE_GOOGLE_PAGE + "</html>"

        return R()

    monkeypatch.setattr("app.engines.google_images._impersonating_session", lambda p: None)
    monkeypatch.setattr("app.engines.google_images.httpx.post", fake_post)
    g = GoogleImagesReverse(proxify_fetch_url="", fetch_proxy_url="http://127.0.0.1:8888", direct_upload=False)
    raw = b"\xff\xd8fakejpeg\x00\x01\xfe"
    hits = g.search_upload(raw, filename="q.jpg", content_type="image/jpeg")

    assert captured["url"] == "https://lens.google.com/upload"
    assert captured["proxy"] == "http://127.0.0.1:8888"
    assert raw in captured["content"]
    assert any(h.img_src == "https://cdn.example.com/photo1.jpg" for h in hits)


def test_google_skips_data_and_google_domains():
    page = (
        '<img src="data:image/png;base64,AAAA">'
        '<img src="https://www.google.com/logo.png">'
        '<img src="https://encrypted-tbn0.gstatic.com/images?q=x">'
        '<img src="https://real.example.com/pic.jpg">'
    )
    hits = parse_google_results(page)
    assert [h.img_src for h in hits] == ["https://real.example.com/pic.jpg"]


def test_chain_queries_all_providers_in_parallel_and_merges():
    """TinEye model: every configured provider is asked (in parallel), and
    ALL their candidates are merged + deduped — never stop at the first
    provider that answered (each engine sees a different slice of the web)."""
    called = set()

    def searx(url):
        called.add("searx")
        return []

    def google(url):
        called.add("google")
        return [ImageHit(title="g", url="https://g/x", img_src="https://g/x.jpg")]

    hits, degraded = run_reverse_chain("https://img/x.jpg", ["searxng", "google"], searx, google)
    assert called == {"searx", "google"}  # both providers were consulted
    assert len(hits) == 1
    assert degraded == []


def test_chain_merges_and_dedupes_hits():
    """Same image found by two providers appears once (deduped by URL)."""

    def searx(url):
        return [ImageHit(title="s", url="https://s/x", img_src="https://same.jpg")]

    def google(url):
        return [ImageHit(title="g", url="https://g/x", img_src="https://same.jpg")]

    hits, degraded = run_reverse_chain("https://img/x.jpg", ["searxng", "google"], searx, google)
    assert len(hits) == 1  # deduped
    assert hits[0].img_src == "https://same.jpg"
    assert degraded == []


def test_chain_degrades_on_failure():
    def searx(url):
        raise EngineError("down")

    def google(url):
        raise EngineError("blocked")

    hits, degraded = run_reverse_chain("https://img/x.jpg", ["searxng", "google"], searx, google)
    assert hits == []
    assert degraded == ["provider:searxng", "provider:google"]


def test_chain_sequential_mode_keeps_order():
    """parallel=False preserves the old one-after-another ladder (used by
    callers that need deterministic provider order / stop-at-first-hit)."""
    called = []

    def searx(url):
        called.append("searx")
        return [ImageHit(title="s", url="https://s/x", img_src="https://s/x.jpg")]

    def google(url):
        called.append("google")
        return [ImageHit(title="g", url="https://g/x", img_src="https://g/x.jpg")]

    hits, degraded = run_reverse_chain(
        "https://img/x.jpg", ["searxng", "google"], searx, google, parallel=False
    )
    assert called == ["searx", "google"]  # sequential order preserved
    # sequential mode merges every provider's hits too (TinEye aggregation)
    assert {h.img_src for h in hits} == {"https://s/x.jpg", "https://g/x.jpg"}
    assert degraded == []


def test_lens_visual_matches_carry_source_pages():
    """Lens result records: thumbnail, full image, then the source page +
    title in a "2003" record; escapes (\\u003d, \\u0026) decoded."""
    from app.engines.google_images import parse_google_results

    rec = (
        '["https://encrypted-tbn0.gstatic.com/images?q\\u003dtbn:ABC",168,300],'
        '["https://cdn.example.com/photo.jpg?w\\u003d1920\\u0026h\\u003d1080",1080,1920],null,0,'
        '{"22":0,"2003":[null,"id1","https://news.example.com/story","A \\"quoted\\" title",null]}'
    )
    hits = parse_google_results("<script>" + rec + "</script>")
    assert len(hits) == 1
    h = hits[0]
    assert h.img_src == "https://cdn.example.com/photo.jpg?w=1920&h=1080"
    assert h.url == "https://news.example.com/story"
    assert h.title == 'A "quoted" title'
    assert h.thumbnail_src.startswith("https://encrypted-tbn0.gstatic.com/images?q=tbn:")
    assert (h.width, h.height) == (1920, 1080) and h.engine == "google lens"
