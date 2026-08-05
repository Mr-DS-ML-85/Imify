"""Cookie-file reader + Facebook redirect-stub resolution in candidate verification."""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from app.cookies import CookieFile, build_cookie_header
from app.matcher import Matcher

NETSCAPE = """# Netscape HTTP Cookie File
# hand-made for tests
.facebook.com\tTRUE\t/\tTRUE\t4102444800\tc_user\t100069746561234
.facebook.com\tTRUE\t/\tTRUE\t4102444800\txs\t20%3Atest
.facebook.com\tTRUE\t/\tTRUE\t4102444800\tdatr\tdatr_val
.fbsbx.com\tTRUE\t/\tTRUE\t4102444800\tsb\tsb_val
.example.com\tTRUE\t/\tFALSE\t4102444800\tnormal\t1
.expired.com\tTRUE\t/\tTRUE\t1\texpired\tgone
"""


@pytest.fixture
def cookie_file(tmp_path):
    p = tmp_path / "cookies.txt"
    p.write_text(NETSCAPE)
    return str(p)


def test_header_for_facebook_domain(cookie_file):
    hdr = build_cookie_header(cookie_file, "www.facebook.com")
    assert "c_user=100069746561234" in hdr
    assert "xs=20%3Atest" in hdr
    assert "datr=datr_val" in hdr


def test_header_for_lookaside_fbsbx(cookie_file):
    """lookaside.fbsbx.com is a sibling host — the .fbsbx.com cookie applies."""
    hdr = build_cookie_header(cookie_file, "lookaside.fbsbx.com")
    assert "sb=sb_val" in hdr


def test_expired_cookies_skipped(cookie_file):
    hdr = build_cookie_header(cookie_file, "www.expired.com")
    assert hdr == ""


def test_no_match_returns_empty(cookie_file):
    assert build_cookie_header(cookie_file, "unrelated.example") == ""


def test_missing_file_returns_empty(tmp_path):
    assert build_cookie_header(str(tmp_path / "nope.txt"), "www.facebook.com") == ""


def test_reload_on_change(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text(".a.com\tTRUE\t/\tTRUE\t4102444800\tk\tv1\n")
    cf = CookieFile(str(p))
    assert "k=v1" in cf.header_for("www.a.com")
    p.write_text(".a.com\tTRUE\t/\tTRUE\t4102444800\tk\tv2\n")
    assert "k=v2" in cf.header_for("www.a.com")


# ------------------------------------------------------------- verify path


class _Resp:
    def __init__(self, status_code=200, content=b"", ctype="image/jpeg"):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": ctype}
        self.text = content.decode("utf-8", errors="replace")


def _jpg_bytes():
    img = Image.new("RGB", (160, 120), (200, 80, 40))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 100, 100], fill=(40, 200, 90))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _make_matcher(tmp_path, cookie_file, proxify_fetch_url=""):
    from app.config import Settings
    from app.repo import Repository
    from app.storage import SqliteStorage

    settings = Settings(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "t.db"),
        data_dir=str(tmp_path / "data"),
        thumbs_dir=str(tmp_path / "data" / "thumbs"),
        cookie_file=cookie_file,
        proxify_fetch_url=proxify_fetch_url,
        web_verify_candidates=True,
        web_verify_concurrency=2,
    )
    settings.ensure_dirs()
    repo = Repository(SqliteStorage(settings.sqlite_path))
    return Matcher(repo, settings)


def test_verify_resolves_facebook_lookaside_stub(tmp_path, monkeypatch):
    """lookaside stub -> location.href redirect -> real image -> hash match."""
    raw = _jpg_bytes()
    matcher = _make_matcher(tmp_path, cookie_file="")

    from app.features import decode_image, extract_features
    from app.models import ImageHit

    img = decode_image(raw)
    feats = extract_features(img, raw)

    stub_html = (
        "<html><head><script type=\"text/javascript\" nonce=\"x\">"
        "location.href = \"https:\\/\\/www.facebook.com\\/photo.php?fbid=123&from_lookaside=1\";"
        "</script></head></html>"
    )
    calls = []

    class FakeGet:
        def get(self, url, **kwargs):
            calls.append((url, kwargs.get("headers", {})))
            # photo.php first — its query string contains 'from_lookaside=1'
            if "photo.php" in url:
                return _Resp(200, raw, "image/jpeg")
            if "lookaside" in url:
                return _Resp(200, stub_html.encode(), "text/html; charset=utf-8")
            return _Resp(404)

    monkeypatch.setattr("app.matcher.httpx.get", FakeGet().get)
    hit = ImageHit(title="fb", url="https://www.facebook.com/", img_src="https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=123", engine="bing")
    results = matcher.verify_web_hits(feats, [hit])
    assert len(results) == 1
    r = results[0]
    assert r.match_type in ("exact", "near")
    assert r.source == "web"
    # the stub was resolved through the redirect chain
    assert len(calls) == 2
    assert "lookaside.fbsbx.com" in calls[0][0]
    assert "photo.php" in calls[1][0]


def test_verify_attaches_cookies_for_fb_host(tmp_path, monkeypatch):
    """The .facebook.com session cookies are attached when verifying a
    facebook.com-hosted candidate (the photo.php hop in the lookaside flow)."""
    raw = _jpg_bytes()
    cookie_file = tmp_path / "c.txt"
    cookie_file.write_text(".facebook.com\tTRUE\t/\tTRUE\t4102444800\tc_user\tsecret123\n")
    matcher = _make_matcher(tmp_path, cookie_file=str(cookie_file))

    from app.features import decode_image, extract_features
    from app.models import ImageHit

    img = decode_image(raw)
    feats = extract_features(img, raw)
    seen_headers = {}

    class FakeGet:
        def get(self, url, **kwargs):
            seen_headers.update(kwargs.get("headers", {}))
            return _Resp(200, raw, "image/jpeg")

    monkeypatch.setattr("app.matcher.httpx.get", FakeGet().get)
    hit = ImageHit(title="x", url="https://www.facebook.com/x", img_src="https://www.facebook.com/photo.php?fbid=123", engine="bing")
    matcher.verify_web_hits(feats, [hit])
    assert "c_user=secret123" in seen_headers.get("Cookie", "")


def test_verify_fb_comet_page_renders_via_proxify(tmp_path, monkeypatch):
    """A JS-rendered FB comet photo page (no og:image, no img) is resolved by
    rendering through proxify gui_chrome and fetching the signed scontent URL."""
    raw = _jpg_bytes()
    matcher = _make_matcher(tmp_path, cookie_file="", proxify_fetch_url="http://127.0.0.1:8081")

    from app.features import decode_image, extract_features
    from app.models import ImageHit

    img = decode_image(raw)
    feats = extract_features(img, raw)
    render_calls = []
    comet = "<html><body>lazy comet shell, no photo url here</body></html>"
    signed = "https://scontent.fdac165-1.fna.fbcdn.net/v/t1.6435-9/12345_n.jpg?stp=x&amp;_nc_sid=abc&amp;oe=6A9A"

    class FakePost:
        def post(self, url, json, timeout):
            render_calls.append(json)

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"success": True, "html": f'<img src="{signed}">'}

            return R()

    class FakeGet:
        def get(self, url, **kwargs):
            if "photo.php" in url:
                return _Resp(200, comet.encode(), "text/html; charset=utf-8")
            if "scontent" in url:
                return _Resp(200, raw, "image/jpeg")
            return _Resp(404)

    monkeypatch.setattr("app.matcher.httpx.post", FakePost().post)
    monkeypatch.setattr("app.matcher.httpx.get", FakeGet().get)

    hit = ImageHit(title="fb", url="https://www.facebook.com/", img_src="https://www.facebook.com/photo.php?fbid=123", engine="bing")
    results = matcher.verify_web_hits(feats, [hit])
    assert len(results) == 1
    r = results[0]
    assert r.match_type in ("exact", "near")
    assert len(render_calls) == 1
    assert render_calls[0]["url"] == "https://www.facebook.com/photo.php?fbid=123"
    assert render_calls[0]["force_strategy"] == "gui_chrome"


def test_verify_render_budget_caps_renders(tmp_path, monkeypatch):
    """Multiple FB comet candidates share one render budget per search."""
    raw = _jpg_bytes()
    from app.config import Settings
    from app.repo import Repository
    from app.storage import SqliteStorage

    settings = Settings(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "t.db"),
        data_dir=str(tmp_path / "data"),
        thumbs_dir=str(tmp_path / "data" / "thumbs"),
        cookie_file="",
        proxify_fetch_url="http://127.0.0.1:8081",
        web_verify_candidates=True,
        web_verify_concurrency=4,
        web_verify_render_max=2,
    )
    settings.ensure_dirs()
    repo = Repository(SqliteStorage(settings.sqlite_path))
    matcher = Matcher(repo, settings)

    from app.features import decode_image, extract_features
    from app.models import ImageHit

    img = decode_image(raw)
    feats = extract_features(img, raw)
    render_calls = []
    comet = "<html><body>comet shell</body></html>"
    signed = "https://scontent.example.fbcdn.net/v/9.jpg?oe=abc"

    class FakePost:
        def post(self, url, json, timeout):
            render_calls.append(json["url"])

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"success": True, "html": f'<img src="{signed}">'}

            return R()

    class FakeGet:
        def get(self, url, **kwargs):
            if "photo.php" in url:
                return _Resp(200, comet.encode(), "text/html; charset=utf-8")
            if "scontent" in url:
                return _Resp(200, raw, "image/jpeg")
            return _Resp(404)

    monkeypatch.setattr("app.matcher.httpx.post", FakePost().post)
    monkeypatch.setattr("app.matcher.httpx.get", FakeGet().get)

    hits = [
        ImageHit(title="f", url="https://www.facebook.com/", img_src=f"https://www.facebook.com/photo.php?fbid={i}", engine="bing")
        for i in range(5)
    ]
    results = matcher.verify_web_hits(feats, hits)
    assert len(render_calls) <= 2  # budget respected
    assert any(r.match_type in ("exact", "near") for r in results)  # some matched


def test_verify_rejects_unrelated(tmp_path, monkeypatch):
    """An unrelated candidate is reported unverified, never as a match."""
    raw = _jpg_bytes()
    matcher = _make_matcher(tmp_path, cookie_file="")
    unrelated = io.BytesIO()
    Image.new("RGB", (160, 120), (10, 10, 10)).save(unrelated, "JPEG")

    from app.features import decode_image, extract_features
    from app.models import ImageHit

    img = decode_image(raw)
    feats = extract_features(img, raw)

    monkeypatch.setattr("app.matcher.httpx.get", lambda url, **kw: _Resp(200, unrelated.getvalue(), "image/jpeg"))
    hit = ImageHit(title="x", url="https://example.com/x", img_src="https://example.com/pic.jpg", engine="bing")
    results = matcher.verify_web_hits(feats, [hit])
    assert len(results) == 1
    assert results[0].match_type == "unverified"
