"""Yandex Images provider tests: cbir-page JSON parser + byte-upload routing."""

from __future__ import annotations

import pytest

from app.engines import EngineError
from app.engines.yandex_images import YandexImagesReverse, parse_yandex_results

SAMPLE_CBIR = """
<script>
window.Ya && (function() {
  var o = {
    "cbir-page": {
      "content": [
        {"img_href":"https://avatars.mds.yandex.net/get-images-cbir/123/photo1.jpeg","original_url":"https://site-a.com/photo","title":"photo one"},
        {"img_href":"https://cdn.example.net/photo2.png","link":"https://site-b.org/page2","title":"photo two"},
        {"img_href":"https://static.example.com/favicon.png","link":"https://x.com"},
        {"img_href":"data:image/png;base64,AAAA","link":"https://y.com"}
      ]
    }
  };
})();
</script>
"""


def test_parse_yandex_cbir_json():
    hits = parse_yandex_results(SAMPLE_CBIR)
    by_url = {h.img_src: h for h in hits}
    assert "https://avatars.mds.yandex.net/get-images-cbir/123/photo1.jpeg" not in by_url  # yandex CDN filtered
    assert "https://cdn.example.net/photo2.png" in by_url
    assert by_url["https://cdn.example.net/photo2.png"].url == "https://site-b.org/page2"
    assert not any(h.img_src.startswith("data:") for h in hits)
    assert all(h.engine == "yandex images" for h in hits)


def test_parse_yandex_filter_keeps_cbir_img_but_meta_only():
    """The CDN filter must not be so broad that it kills real cbir results."""
    page = (
        '<script>{"cbir-page":{"content":['
        '{"img_href":"https://cdn.example.com/pic.jpg","link":"https://s.com/p"},'
        '{"img_href":"https://avatars.mds.yandex.net/get-images-cbir/9/x.jpg","link":"https://t.com"}'
        ']}}</script>'
    )
    hits = parse_yandex_results(page)
    assert any(h.img_src == "https://cdn.example.com/pic.jpg" for h in hits)


def test_parse_yandex_markup_fallback():
    page = '<a href="https://page.com/a"><img src="https://img.com/a.jpg"></a>'
    hits = parse_yandex_results(page)
    assert len(hits) >= 1
    assert hits[0].img_src == "https://img.com/a.jpg"


def test_yandex_upload_via_tunnel(monkeypatch):
    captured = {}

    def fake_post(url, data, files, timeout, follow_redirects, proxy, headers):
        captured.update(url=url, data=data, files=files, proxy=proxy)

        class R:
            def raise_for_status(self):
                pass

            status_code = 200
            text = SAMPLE_CBIR

        return R()

    monkeypatch.setattr("app.engines.yandex_images.httpx.post", fake_post)
    y = YandexImagesReverse(proxify_fetch_url="", fetch_proxy_url="http://127.0.0.1:8888")
    raw = b"\xff\xd8fakejpeg\x00\x01\xfe"
    hits = y.search_upload(raw, filename="q.jpg")

    assert captured["url"] == "https://yandex.com/images/search"
    assert captured["proxy"] == "http://127.0.0.1:8888"
    assert captured["data"]["rpt"] == "imageview"
    assert captured["files"]["upfile"][0] == "q.jpg"
    assert any(h.img_src == "https://cdn.example.net/photo2.png" for h in hits)


def test_yandex_upload_all_paths_fail(monkeypatch):
    def fake_post(url, data, files, timeout, follow_redirects, proxy, headers):
        raise RuntimeError("network down")

    monkeypatch.setattr("app.engines.yandex_images.httpx.post", fake_post)
    y = YandexImagesReverse(proxify_fetch_url="", fetch_proxy_url="")
    with pytest.raises(EngineError, match="yandex-upload failed via all paths"):
        y.search_upload(b"\x01")


def test_parse_yandex_js_shell_returns_no_junk_hits():
    """A modern JS-gated imageview shell (auth/support links only) must NOT
    be parsed into fake hits — the chain would try to verify passport/tune
    URLs and pollute results with junk that can never match."""
    shell = """
    <html><head><title>Yandex Images</title></head><body>
    <a href="https://passport.yandex.com/auth?origin=images_head_auth">sign in</a>
    <a href="https://yandex.com/support/images/troubleshooting.html">help</a>
    <a href="https://yandex.com/tune/search/">settings</a>
    <a href="https://yandex.com/company/">about</a>
    <div id="cbirPreview" data-empty="true"></div>
    </body></html>
    """
    hits = parse_yandex_results(shell)
    assert hits == []


def test_parse_yandex_real_page_ignores_support_links():
    """Real results are kept; support/auth links mixed in are dropped."""
    page = (
        '<script>{"cbir-page":{"content":['
        '{"img_href":"https://cdn.example.com/pic.jpg","link":"https://s.com/p"},'
        '{"img_href":"https://passport.yandex.com/auth?x=1","link":"https://yandex.com"}'
        ']}}</script>'
    )
    hits = parse_yandex_results(page)
    assert len(hits) == 1
    assert hits[0].img_src == "https://cdn.example.com/pic.jpg"


def test_yandex_url_reverse_requires_proxify():
    y = YandexImagesReverse(proxify_fetch_url="", fetch_proxy_url="")
    with pytest.raises(EngineError, match="PROXIFY_FETCH_URL"):
        y.search("https://public.example/x.jpg")


# ------------------------------------------------------------ stage 2 (render)


def test_yandex_url_reverse_renders_when_cbir_absent(monkeypatch):
    """Stage 2: when the plain proxify fetch returns a JS-gated shell (no
    cbir-page — Yandex 2026), the imageview URL is rendered through proxify's
    real browser (gui_chrome) and the rendered DOM is parsed for hits."""
    shell = (
        '<div id="cbirPreview" data-empty="true"></div>'
        '<a href="https://passport.yandex.com/auth?x=1">sign in</a>'
    )
    rendered = (
        '<script>{"cbir-page":{"content":['
        '{"img_href":"https://cdn.example.com/pic.jpg","link":"https://s.com/p"}'
        ']}}</script>'
    )
    calls = []

    class FakePost:
        def post(self, url, json, timeout):
            calls.append(json)

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    # first call = plain fetch (shell), second = gui_chrome render
                    if json.get("force_strategy") == "gui_chrome":
                        return {"success": True, "html": rendered, "strategy_used": "gui_chrome"}
                    return {"success": True, "html": shell, "strategy_used": "simple"}

            return R()

    monkeypatch.setattr("app.engines.yandex_images.httpx.post", FakePost().post)
    y = YandexImagesReverse(proxify_fetch_url="http://127.0.0.1:8081")
    hits = y.search("https://public.example/x.jpg")
    assert len(hits) == 1
    assert hits[0].img_src == "https://cdn.example.com/pic.jpg"
    assert hits[0].url == "https://s.com/p"
    # the render call forced gui_chrome + bypassed cache for fresh signatures
    render_call = [c for c in calls if c.get("force_strategy") == "gui_chrome"][0]
    assert "rpt=imageview&url=" in render_call["url"]
    assert render_call["bypass_cache"] is True


def test_yandex_url_reverse_stage1_wins_when_hits(monkeypatch):
    """Stage 1 hits short-circuit — the render is only the fallback."""
    rendered_ok = (
        '<script>{"cbir-page":{"content":['
        '{"img_href":"https://cdn.example.com/a.jpg","link":"https://a.com"}'
        ']}}</script>'
    )
    calls = []

    class FakePost:
        def post(self, url, json, timeout):
            calls.append(json)

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"success": True, "html": rendered_ok}

            return R()

    monkeypatch.setattr("app.engines.yandex_images.httpx.post", FakePost().post)
    y = YandexImagesReverse(proxify_fetch_url="http://127.0.0.1:8081")
    hits = y.search("https://public.example/x.jpg")
    assert len(hits) == 1
    assert not any(c.get("force_strategy") == "gui_chrome" for c in calls)  # stage 1 only


def test_yandex_upload_tries_proxify_fetch_post(monkeypatch):
    """Byte-upload routes through proxify /fetch as the PRIMARY stage (browser
    POST — gui_chrome + wait_selector). Its rendered verdict is authoritative:
    when it returns hits, the tunnel is never attempted."""
    good = (
        '<script>{"cbir-page":{"content":['
        '{"img_href":"https://cdn.example.com/p.jpg","link":"https://p.com"}'
        ']}}</script>'
    )
    calls = []

    def fake_post(url, data=None, files=None, timeout=None, follow_redirects=None, proxy=None, headers=None, json=None):
        # proxify /fetch with json -> good; tunnel POST (files) -> not reached
        calls.append((url, bool(files), bool(json)))

        class R:
            def raise_for_status(self):
                pass

            status_code = 200
            text = ""

            def json(self):
                return {"success": True, "html": good, "strategy_used": "gui_chrome"}

        return R()

    monkeypatch.setattr("app.engines.yandex_images.httpx.post", fake_post)
    y = YandexImagesReverse(proxify_fetch_url="http://127.0.0.1:8081", fetch_proxy_url="http://127.0.0.1:8888")
    hits = y.search_upload(b"\xff\xd8fake\x00", filename="q.jpg")
    assert len(hits) == 1
    assert hits[0].img_src == "https://cdn.example.com/p.jpg"
    assert any(c[2] for c in calls)  # proxify /fetch POST attempted (browser route)
    assert not any(c[1] for c in calls)  # tunnel NOT attempted — browser verdict won
    fetch_call = next(c[2] for c in calls if c[2])
    assert fetch_call


def test_yandex_upload_falls_to_tunnel_when_fetch_errors(monkeypatch):
    """When the browser /fetch route raises (blocked/timeout), the tunnel is
    attempted next — the ladder still works."""
    good = (
        '<script>{"cbir-page":{"content":['
        '{"img_href":"https://cdn.example.com/t.jpg","link":"https://t.com"}'
        ']}}</script>'
    )
    calls = []

    def fake_post(url, data=None, files=None, timeout=None, follow_redirects=None, proxy=None, headers=None, json=None):
        calls.append((url, bool(files), bool(json)))
        if json:
            # proxify /fetch route raises (httpx error) -> degrade, next route
            import httpx as _httpx

            class E:
                def raise_for_status(self):
                    raise _httpx.ConnectError("fetch route down")

                def json(self):
                    raise ValueError("non-json")

            return E()

        class R:
            def raise_for_status(self):
                pass

            status_code = 200
            text = good

        return R()

    monkeypatch.setattr("app.engines.yandex_images.httpx.post", fake_post)
    y = YandexImagesReverse(proxify_fetch_url="http://127.0.0.1:8081", fetch_proxy_url="http://127.0.0.1:8888")
    hits = y.search_upload(b"\xff\xd8fake\x00", filename="q.jpg")
    assert len(hits) == 1
    assert hits[0].img_src == "https://cdn.example.com/t.jpg"
    assert any(c[1] for c in calls)  # tunnel attempted as fallback
