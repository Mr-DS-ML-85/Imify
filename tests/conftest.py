from __future__ import annotations

import io
import time

import pytest
from fastapi import FastAPI
from PIL import Image

from app.auth import AuthService, RateLimiter
from app.config import Settings
from app.indexer import Indexer
from app.matcher import Matcher
from app.models import ImageHit
from app.repo import Repository
from app.storage import SqliteStorage


class FakeSearx:
    def __init__(self):
        self.available = True
        self.total_requests = 0
        self.failures = 0
        self.hits: list[ImageHit] = []
        self.reverse_hits: list[ImageHit] = []
        self.fail_reverse = False

    def discover(self, query: str, safesearch: int = 0):
        self.total_requests += 1
        return self.hits

    def reverse_search(self, image_url: str, safesearch: int = 0):
        self.total_requests += 1
        if self.fail_reverse:
            from app.searx import SearxError

            raise SearxError("SearXNG unreachable (simulated)")
        return self.reverse_hits


class FakeGoogle:
    def __init__(self):
        self.hits: list[ImageHit] = []
        self.upload_hits: list[ImageHit] = []
        self.fail = False
        self.fail_upload = False
        self.uploads: list[tuple[bytes, str]] = []

    @property
    def available(self):
        return True

    def search(self, image_url: str):
        if self.fail:
            from app.engines import EngineError

            raise EngineError("google blocked (simulated)")
        return self.hits

    def search_upload(self, image_bytes: bytes, filename: str = "image.jpg", content_type: str = "image/jpeg"):
        self.uploads.append((image_bytes, filename))
        if self.fail_upload:
            from app.engines import EngineError

            raise EngineError("google upload blocked (simulated)")
        return self.upload_hits


class FakeBing:
    def __init__(self):
        self.hits: list[ImageHit] = []
        self.upload_hits: list[ImageHit] = []
        self.fail = False
        self.fail_upload = False
        self.uploads: list[tuple[bytes, str]] = []

    @property
    def available(self):
        return True

    def search(self, image_url: str):
        if self.fail:
            from app.engines import EngineError

            raise EngineError("bing blocked (simulated)")
        return self.hits

    def search_upload(self, image_bytes: bytes, filename: str = "image.jpg"):
        self.uploads.append((image_bytes, filename))
        if self.fail_upload:
            from app.engines import EngineError

            raise EngineError("bing upload blocked (simulated)")
        return self.upload_hits


class FakeYandex:
    def __init__(self):
        self.hits: list[ImageHit] = []
        self.upload_hits: list[ImageHit] = []
        self.fail = False
        self.fail_upload = False
        self.uploads: list[tuple[bytes, str]] = []

    @property
    def available(self):
        return True

    def search(self, image_url: str):
        if self.fail:
            from app.engines import EngineError

            raise EngineError("yandex blocked (simulated)")
        return self.hits

    def search_upload(self, image_bytes: bytes, filename: str = "image.jpg"):
        self.uploads.append((image_bytes, filename))
        if self.fail_upload:
            from app.engines import EngineError

            raise EngineError("yandex upload blocked (simulated)")
        return self.upload_hits


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        uploads_dir=str(tmp_path / "data" / "uploads"),
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "test.db"),
        data_dir=str(tmp_path / "data"),
        thumbs_dir=str(tmp_path / "data" / "thumbs"),
        indexer_enabled=False,
        indexer_fetch_workers=0,
        searxng_url="http://fake-searx:8080",
        searxng_query_interval=0,
        bootstrap_admin_key="sk_test_admin",
        web_reverse_enabled=False,
        public_base_url="",
        auth_mode="open",
        rate_limit_per_minute=1000,
        cookie_file="",
    )


@pytest.fixture
def ctx(settings):
    settings.ensure_dirs()
    storage = SqliteStorage(settings.sqlite_path)
    repo = Repository(storage)
    searx = FakeSearx()
    google = FakeGoogle()
    bing = FakeBing()
    yandex = FakeYandex()
    matcher = Matcher(repo, settings)
    indexer = Indexer(repo, settings, searx, google, bing, yandex)
    auth = AuthService(repo, settings.bootstrap_admin_key)
    limiter = RateLimiter(per_minute=settings.rate_limit_per_minute, per_hour=10000, per_day=50000)
    return {
        "settings": settings,
        "storage": storage,
        "repo": repo,
        "searx": searx,
        "google": google,
        "bing": bing,
        "yandex": yandex,
        "matcher": matcher,
        "indexer": indexer,
        "auth": auth,
        "limiter": limiter,
    }


@pytest.fixture
def client(ctx):
    from app.api import build_api

    app = FastAPI()
    app.include_router(
        build_api(
            ctx["settings"], ctx["repo"], ctx["matcher"], ctx["indexer"], ctx["auth"], ctx["searx"], ctx["limiter"],
            ctx["google"], ctx["bing"], ctx["yandex"],
        )
    )
    from fastapi.testclient import TestClient

    return TestClient(app)


class FakeImageFetch:
    """Patches httpx.get so candidate downloads return canned image bytes —
    lets tests exercise the real hash-verification path (engines find, the
    fused-hash matcher judges)."""

    def __init__(self, image_bytes: bytes):
        self.image_bytes = image_bytes
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1

        class R:
            status_code = 200
            headers = {"content-type": "image/jpeg"}
            content = self.image_bytes

        return R()


def patch_image_fetch(monkeypatch, image_bytes: bytes) -> FakeImageFetch:
    """Serve `image_bytes` for every candidate download (verify path)."""
    fake = FakeImageFetch(image_bytes)
    monkeypatch.setattr("app.matcher.httpx.get", fake.get)
    monkeypatch.setattr("app.indexer.httpx.get", fake.get)
    return fake


def make_image_bytes(mode: str = "photo", size=(160, 120), color=None, gradient_dir="h") -> bytes:
    """Synthetic image bytes; near-duplicates share most pixels."""
    if mode == "gradient":
        img = Image.new("RGB", size)
        px = img.load()
        for y in range(size[1]):
            for x in range(size[0]):
                if gradient_dir == "h":
                    px[x, y] = (int(x * 255 / size[0]), int(y * 255 / size[1]), 128)
                else:
                    px[x, y] = (int(y * 255 / size[1]), int(x * 255 / size[0]), 128)
    elif mode == "solid":
        img = Image.new("RGB", size, color or (40, 90, 200))
    else:
        img = Image.new("RGB", size, color or (40, 90, 200))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.ellipse([20, 20, 100, 100], fill=(220, 60, 40))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def image_features(raw: bytes) -> dict:
    """Extract the fused feature dict from raw image bytes (for trackers)."""
    from app.features import decode_image, extract_features

    img = decode_image(raw)
    assert img is not None
    return extract_features(img, raw)


def index_image(repo, settings, raw: bytes, url: str = "", phash_override: int | None = None, **kw) -> int:
    """Extract features + index an image through the same path the worker uses."""
    from app.features import decode_image, encode_hash_vector, extract_features
    from app.models import ImageMeta

    img = decode_image(raw)
    assert img is not None
    feats = extract_features(img, raw)
    image_id = repo.next_image_id()
    meta = ImageMeta(
        id=image_id,
        url=url or f"https://example.com/{image_id}.jpg",
        phash=phash_override if phash_override is not None else int(feats["phash"]),
        dhash=int(feats["dhash"]),
        ahash=int(feats["ahash"]),
        color=feats["color"],
        sha256=feats["sha256"],
        width=feats["width"],
        height=feats["height"],
        first_seen=time.time(),
        last_seen=time.time(),
        **kw,
    )
    repo.db.vadd(image_id, encode_hash_vector(meta.phash))
    repo.put_image(meta)
    repo.register_sha(meta.sha256, image_id)
    repo.register_url(meta.url, image_id)
    repo.register_phash(meta.phash, image_id)
    return image_id
