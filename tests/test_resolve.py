"""Tests for resolve_image_bytes — the page-URL → image-bytes resolver used by
the /v1/search image_url path (fixes \"fails to decode the FB photo URL\")."""

from __future__ import annotations

import io
import re

import pytest
from PIL import Image, ImageDraw

from app.resolve import resolve_image_bytes


def _jpg_bytes() -> bytes:
    img = Image.new("RGB", (160, 120), (200, 80, 40))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 100, 100], fill=(40, 200, 90))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


class _Resp:
    def __init__(self, status_code=200, content=b"", ctype="image/jpeg", url=""):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": ctype}
        self.text = content.decode("utf-8", errors="replace")
        self.url = url


RAW = _jpg_bytes()
ESCAPED_SIGNED = (
    "https:\\/\\/scontent.fdac184-1.fna.fbcdn.net\\/v\\/t39.30808-6\\/725803402_139122029321284"
    "_12345_n.jpg?_nc_cat=111&ccb=1-7&_nc_sid=7da55a&_nc_oc=Adp9jaI47oQ5AnLprBFReHIZmXQ22K0bIvPpkZUYcMoZaun8EgAtzE3e-2u7bTaCTGg"
)


def test_og_image_page_resolves(monkeypatch):
    """A generic photo page with og:image -> the image bytes."""
    calls = []

    class FakeGet:
        def get(self, url, **kwargs):
            calls.append(url)
            if url == "https://site.com/photo":
                return _Resp(
                    200,
                    b'<meta property="og:image" content="https://cdn.site.com/pic.jpg">',
                    "text/html; charset=utf-8",
                )
            if url == "https://cdn.site.com/pic.jpg":
                return _Resp(200, RAW, "image/jpeg", url=url)
            return _Resp(404)

    monkeypatch.setattr("app.resolve.httpx.get", FakeGet().get)
    content, resolved = resolve_image_bytes("https://site.com/photo")
    assert content == RAW
    assert resolved == "https://cdn.site.com/pic.jpg"
    assert len(calls) == 2


def test_direct_image_url_returns_bytes(monkeypatch):
    monkeypatch.setattr("app.resolve.httpx.get", lambda url, **kw: _Resp(200, RAW, "image/jpeg", url=url))
    content, resolved = resolve_image_bytes("https://cdn.site.com/direct.jpg")
    assert content == RAW
    assert resolved == "https://cdn.site.com/direct.jpg"


def test_fb_escaped_scontent_resolves_with_cookies(monkeypatch, tmp_path):
    """The FB photo page (the exact URL the user pasted) embeds ESCAPED
    scontent URLs in JS JSON — the resolver unescapes, fetches with session
    cookies through the tunnel and returns decodable pixels."""
    cookie_file = tmp_path / "c.txt"
    cookie_file.write_text(".facebook.com\tTRUE\t/\tTRUE\t4102444800\tc_user\tsecret123\n")
    page = (
        '<html><body><script>var d = {'
        f'"reactions":{{"1":"{ESCAPED_SIGNED}"}}'
        "};</script></body></html>"
    )
    seen = {}

    class FakeGet:
        def get(self, url, **kwargs):
            seen[url] = kwargs
            if "facebook.com/photo" in url:
                return _Resp(200, page.encode(), "text/html; charset=utf-8", url=url)
            if "scontent" in url:
                return _Resp(200, RAW, "image/jpeg", url=url)
            return _Resp(404)

    monkeypatch.setattr("app.resolve.httpx.get", FakeGet().get)
    content, resolved = resolve_image_bytes(
        "https://www.facebook.com/photo/?fbid=1391220289879510&set=a.646142547720625",
        cookie_file=str(cookie_file),
        proxy="http://127.0.0.1:8888",
    )
    assert content == RAW
    assert "scontent" in resolved
    # session cookies attached on the facebook.com page fetch (the photo page
    # is the logged-in comet shell); the signed scontent URL is its own auth
    page_cookie = seen["https://www.facebook.com/photo/?fbid=1391220289879510&set=a.646142547720625"].get("headers", {}).get("Cookie", "")
    assert "secret123" in page_cookie
    # the scontent fetch carries the page as referer (hotlink protection)
    assert seen[resolved].get("headers", {}).get("Referer", "").startswith("https://www.facebook.com")


def test_fb_page_renders_via_proxify_when_no_embedded_url(monkeypatch):
    """FB comet shell with no server-side scontent URL -> gui_chrome render
    reveals it -> fetch through the tunnel."""
    cookie_file = ""
    comet = "<html><body>comet shell, photo only after JS</body></html>"
    signed_plain = ESCAPED_SIGNED.replace("\\/", "/")
    render_calls = []

    class FakePost:
        def post(self, url, json, timeout):
            render_calls.append(json)

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"success": True, "html": f'<img src="{signed_plain}">'}

            return R()

    class FakeGet:
        def get(self, url, **kwargs):
            if "facebook.com/photo" in url:
                return _Resp(200, comet.encode(), "text/html; charset=utf-8", url=url)
            if "scontent" in url:
                return _Resp(200, RAW, "image/jpeg", url=url)
            return _Resp(404)

    monkeypatch.setattr("app.resolve.httpx.post", FakePost().post)
    monkeypatch.setattr("app.resolve.httpx.get", FakeGet().get)
    content, resolved = resolve_image_bytes(
        "https://www.facebook.com/photo/?fbid=1391220289879510",
        cookie_file=cookie_file,
        proxify_fetch="http://127.0.0.1:8081",
    )
    assert content == RAW
    assert len(render_calls) == 1
    assert render_calls[0]["force_strategy"] == "gui_chrome"
    assert "scontent" in resolved


def test_unresolvable_returns_none(monkeypatch):
    class FakeGet:
        def get(self, url, **kwargs):
            return _Resp(404)

    monkeypatch.setattr("app.resolve.httpx.get", FakeGet().get)
    content, resolved = resolve_image_bytes("https://example.com/nothere")
    assert content is None
