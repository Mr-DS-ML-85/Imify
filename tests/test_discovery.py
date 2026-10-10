"""Text-guided discovery (Lens best guess), rotation invariance, bandwidth
guards and cookie-expiry handling."""

from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from app.engines.text_images import (
    DiscoveryReport,
    derive_queries,
    is_iconlike,
    parse_bing_images,
)
from app.models import ImageHit

from conftest import index_image, patch_image_fetch
from test_accuracy import _client_with, jpeg, textured

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------ best guess
def test_best_guess_from_titles_ignores_site_names_and_boilerplate():
    titles = [
        "Python Logo - PNG and Vector - Logo Download",
        "Python logo png | Free download",
        "File:Python-logo-notext.svg - Wikipedia",
        "Python Logo History and Meaning",
        "Python programming language logo",
    ]
    hits = [ImageHit(title=t, url=f"https://site{i}.example/p", img_src="") for i, t in enumerate(titles)]
    queries = derive_queries(hits)
    assert queries and queries[0] == "python logo"


def test_best_guess_needs_agreement_then_falls_back_to_ocr():
    hits = [ImageHit(title="Totally unrelated words", url="https://a.example", img_src="")]
    assert derive_queries(hits) == []
    assert derive_queries(hits, ocr_text="BREAKING NEWS tonight") == ["breaking news tonight"]


def test_bing_rendered_results_parse_to_image_and_source_page():
    hits = parse_bing_images((HERE / "fixtures_bing_images.html").read_text())
    assert len(hits) >= 2
    for h in hits:
        assert h.img_src.startswith("http") and "bing.com" not in h.img_src
        assert h.url.startswith("http") and "bing.com" not in h.url
        assert h.engine == "bing images"
    assert any(h.title for h in hits)


def test_iconlike_detects_logos_not_photos():
    logo = Image.new("RGB", (256, 256), (255, 255, 255))
    ImageDraw.Draw(logo).ellipse([40, 40, 216, 216], fill=(30, 90, 200))
    assert is_iconlike(logo) is True
    assert is_iconlike(textured(9)) is False


class FakeDiscovery:
    available = True

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, queries, include_icons=False, budget_seconds=6.0):
        self.calls.append((list(queries), include_icons))
        return DiscoveryReport(queries=list(queries), hits=list(self.hits), engines={"pinterest": len(self.hits)})


def test_text_discovery_finds_copy_the_reverse_engines_missed(ctx, monkeypatch):
    """Reverse engines only return similar-looking junk, but their titles say
    what the image is; the best-guess text search finds the real copy, which
    the verifier confirms."""
    base = textured(11)
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["settings"].text_discovery_enabled = True
    ctx["settings"].web_phase_budget_seconds = 30
    ctx["searx"].reverse_hits = [
        ImageHit(title=f"Red Panda Sticker - shop{i}", url=f"https://shop{i}.example/p", img_src=f"https://cdn{i}.example/x.jpg", engine="bing")
        for i in range(3)
    ]
    copy_hit = ImageHit(title="red panda sticker", url="https://pins.example/pin/1", img_src="https://pins.example/full.jpg", engine="pinterest")
    disc = FakeDiscovery([copy_hit])

    def fake_get(url, **kwargs):
        class R:
            status_code = 200
            headers = {"content-type": "image/jpeg"}
            content = jpeg(base) if "pins.example" in url else jpeg(textured(12))
        return R()

    monkeypatch.setattr("app.matcher.httpx.get", fake_get)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import build_api

    app = FastAPI()
    app.include_router(build_api(
        ctx["settings"], ctx["repo"], ctx["matcher"], ctx["indexer"], ctx["auth"], ctx["searx"], ctx["limiter"],
        ctx["google"], ctx["bing"], ctx["yandex"], text_discovery=disc,
    ))
    d = TestClient(app).post("/v1/search", files={"file": ("q.jpg", jpeg(base), "image/jpeg")}, data={"include_web": "true"}).json()
    assert disc.calls and disc.calls[0][0][0] == "red panda sticker"
    assert d["best_guess"] == "red panda sticker"
    assert "text" in d["providers"]
    confirmed = [r for r in d["results"] if r["engine"] == "pinterest"]
    assert confirmed and confirmed[0]["match_type"] in ("exact", "near")
    assert d["found"] is True


# ------------------------------------------------------------ rotation
def test_rotated_copy_found_in_local_index(client, ctx):
    base = textured(13)
    image_id = index_image(ctx["repo"], ctx["settings"], jpeg(base), url="https://lib.example/a.jpg")
    rotated = base.rotate(90, expand=True)
    d = client.post("/v1/search", files={"file": ("r.jpg", jpeg(rotated), "image/jpeg")}, data={"include_web": "false"}).json()
    hit = [r for r in d["results"] if r["image_id"] == image_id]
    assert hit and hit[0]["match_type"] in ("exact", "near")
    assert "rotated" in hit[0]["ai_modalities"]


def test_exif_orientation_is_applied():
    from app.features import decode_image

    base = textured(14).resize((200, 100))
    sideways = base.rotate(90, expand=True)  # stored pixels: 100x200
    exif = Image.Exif()
    exif[0x0112] = 8  # "rotate 90 CW to display"
    buf = io.BytesIO()
    sideways.save(buf, "JPEG", exif=exif.tobytes())
    img = decode_image(buf.getvalue())
    assert img is not None and img.size == (200, 100)


# ------------------------------------------------------------ bandwidth
class CountingFetch:
    def __init__(self, thumb: bytes, full: bytes):
        self.thumb, self.full = thumb, full
        self.urls: list[str] = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        body = self.thumb if "thumb" in url else self.full

        class R:
            status_code = 200
            headers = {"content-type": "image/jpeg"}
            content = body

        return R()


class StubClip:
    """'CLIP' that calls everything unrelated (0.1) except identical pixels."""

    available = True

    def embed_image(self, img):
        import numpy as np

        v = np.asarray(img.convert("L").resize((8, 8)), dtype=np.float32).reshape(-1)
        v = v - v.mean()
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist()

    @staticmethod
    def match(a, b):
        import numpy as np

        return float(np.dot(a, b)) if a and b and len(a) == len(b) else 0.0


def test_unrelated_candidate_rejected_on_thumbnail_without_full_download(ctx, monkeypatch):
    fetch = CountingFetch(thumb=jpeg(textured(15).resize((120, 90))), full=jpeg(textured(15)))
    monkeypatch.setattr("app.matcher.httpx.get", fetch.get)
    ctx["matcher"].clip = StubClip()
    q = textured(16)
    from app.features import extract_features

    hit = ImageHit(title="x", url="https://site.example/p", img_src="https://site.example/full.jpg", thumbnail_src="https://site.example/thumb.jpg")
    out = ctx["matcher"].verify_web_hits(
        extract_features(q), [hit], query_clip=StubClip().embed_image(q), query_img=q, budget_seconds=10
    )
    assert out[0].match_type == "unverified" and "thumbnail" in out[0].note
    assert fetch.urls == ["https://site.example/thumb.jpg"]  # full image never fetched


def test_candidate_bytes_are_cached_per_url(ctx, monkeypatch):
    base = textured(17)
    fake = patch_image_fetch(monkeypatch, jpeg(base))
    from app.features import extract_features

    hit = ImageHit(title="x", url="https://site.example/p", img_src="https://site.example/full.jpg")
    feats = extract_features(base)
    first = ctx["matcher"].verify_web_hits(feats, [hit], budget_seconds=10)
    second = ctx["matcher"].verify_web_hits(feats, [hit], budget_seconds=10)
    assert first[0].match_type in ("exact", "near") and second[0].match_type == first[0].match_type
    assert fake.calls == 1
    assert "cache" in second[0].resolution


# ------------------------------------------------------------ cookies
def test_chromium_epoch_expiry_is_normalized(tmp_path):
    from app.cookies import CookieFile, cookie_health

    past_unix = int(time.time()) - 86400
    future_unix = int(time.time()) + 86400
    path = tmp_path / "c.txt"
    path.write_text(
        # exported WITHOUT the 1601->1970 shift (looks ~369 years in the future)
        f".facebook.com\tTRUE\t/\tTRUE\t{past_unix + 11_644_473_600}\txs\tdead\n"
        f".facebook.com\tTRUE\t/\tTRUE\t{future_unix + 11_644_473_600}\tc_user\talive\n"
    )
    assert CookieFile(path).header_for("www.facebook.com") == "c_user=alive"
    report = cookie_health(path)["domains"]["facebook.com"]
    assert report["expired"] == 1 and report["live"] == 1
    assert 20 < report["soonest_expiry_h"] < 28
