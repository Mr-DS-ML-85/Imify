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
    """Byte-upload uses the true two-step flow: POST image-download (raw bytes
    in memory, no disk download) -> read cbir_id -> GET /images/search?cbir_id=."""
    captured = {}

    class R:
        def raise_for_status(self):
            pass

        status_code = 200
        text = (
            '{"cbir_id":"123/abc","namespace":"images-cbir",'
            '"sizes":{"orig":{"path":"https://avatars.mds.yandex.net/get-images-cbir/123/abc/orig"}}}'
        )

    class FakeClient:
        def __init__(self, *a, **k):
            captured["client_kwargs"] = k
            self.follow_redirects = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, content=None, headers=None, **k):
            captured.setdefault("posts", []).append((url, bool(content), headers))
            return R()

        def get(self, url, **k):
            captured["get_url"] = url
            return R(text=SAMPLE_CBIR) if False else R2()

    class R2:
        def raise_for_status(self):
            pass

        status_code = 200
        text = SAMPLE_CBIR

    monkeypatch.setattr("app.engines.yandex_images.httpx.Client", FakeClient)
    y = YandexImagesReverse(proxify_fetch_url="", fetch_proxy_url="http://127.0.0.1:8888")
    raw = b"\xff\xd8fakejpeg\x00\x01\xfe"
    hits = y.search_upload(raw, filename="q.jpg")

    assert captured["client_kwargs"]["proxy"] == "http://127.0.0.1:8888"
    assert captured["posts"][0][0] == "https://yandex.com/images-apphost/image-download?cbird=37&images_avatars_size=preview&images_avatars_namespace=images-cbir"
    assert captured["posts"][0][1] is True  # raw bytes posted in-memory
    assert "cbir_id=123%2Fabc" in captured["get_url"] or "cbir_id=123/abc" in captured["get_url"]
    assert "rpt=imageview" in captured["get_url"]
    assert any(h.img_src == "https://cdn.example.net/photo2.png" for h in hits)


def test_yandex_upload_all_paths_fail(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("app.engines.yandex_images.httpx.Client", boom)
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
    """Byte-upload routes through proxify /fetch (browser gui_chrome) as a
    carrier for the two-step flow: image-download POST -> cbir_id -> GET."""
    step1 = '{"cbir_id":"44/x","namespace":"images-cbir","sizes":{"orig":{"path":"https://avatars.mds.yandex.net/get-images-cbir/44/x/orig"}}}'
    good = (
        '<script>{"cbir-page":{"content":['
        '{"img_href":"https://cdn.example.com/p.jpg","link":"https://p.com"}'
        ']}}</script>'
    )
    calls = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, content=None, headers=None, **k):
            return R(step1, 200)

        def get(self, url, **k):
            calls.append(url)
            return R(good, 200)

    class R:
        def __init__(self, text, status):
            self.text = text
            self.status_code = status

        def raise_for_status(self):
            pass

    monkeypatch.setattr("app.engines.yandex_images.httpx.Client", FakeClient)
    y = YandexImagesReverse(proxify_fetch_url="http://127.0.0.1:8081", fetch_proxy_url="http://127.0.0.1:8888")
    hits = y.search_upload(b"\xff\xd8fake\x00", filename="q.jpg")
    assert len(hits) == 1
    assert hits[0].img_src == "https://cdn.example.com/p.jpg"
    assert any("cbir_id=44%2Fx" in u or "cbir_id=44/x" in u for u in calls)


def test_yandex_upload_falls_to_tunnel_when_browser_fails(monkeypatch):
    """When the browser /fetch carrier errors (no cbir id, transport), the
    tunnel httpx path is attempted next — the ladder still works."""
    good = (
        '<script>{"cbir-page":{"content":['
        '{"img_href":"https://cdn.example.com/t.jpg","link":"https://t.com"}'
        ']}}</script>'
    )
    step1 = '{"cbir_id":"7/z","namespace":"images-cbir","sizes":{"orig":{"path":"https://avatars.mds.yandex.net/get-images-cbir/7/z/orig"}}}'
    calls = []

    def fake_httpx_post(url, json=None, timeout=None, **k):
        if json is not None:
            raise RuntimeError("fetch route down")

    class R:
        def __init__(self, text, status):
            self.text = text
            self.status_code = status

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, content=None, headers=None, **k):
            calls.append(("post", url))
            return R(step1, 200)

        def get(self, url, **k):
            calls.append(("get", url))
            return R(good, 200)

    import app.engines.yandex_images as yi

    real_httpx_post = yi.httpx.post
    yi.httpx.post = fake_httpx_post
    try:
        monkeypatch.setattr("app.engines.yandex_images.httpx.Client", FakeClient)
        y = YandexImagesReverse(proxify_fetch_url="http://127.0.0.1:8081", fetch_proxy_url="http://127.0.0.1:8888")
        hits = y.search_upload(b"\xff\xd8fake\x00", filename="q.jpg")
    finally:
        yi.httpx.post = real_httpx_post
    assert len(hits) == 1
    assert hits[0].img_src == "https://cdn.example.com/t.jpg"
    assert any(c[0] == "get" for c in calls)
