"""Accuracy layers: face identity, geometric (crop/edit) verification,
mirrored copies, CLIP leads, the local AI grid and the real-time budget."""

from __future__ import annotations

import io
import random
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

from app.face import FaceId
from app.models import ImageHit

from conftest import index_image, patch_image_fetch

ROOT = Path(__file__).resolve().parent.parent
FACE_MODELS = ROOT / "data" / "models" / "face"
SAMPLE_FACE = ROOT / "images" / "images.webp"


def textured(seed: int = 7, size=(480, 360)) -> Image.Image:
    """A keypoint-rich synthetic 'photo' (random shapes), deterministic."""
    rng = random.Random(seed)
    img = Image.new("RGB", size, (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    draw = ImageDraw.Draw(img)
    for _ in range(140):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        w, h = rng.randrange(8, 70), rng.randrange(8, 70)
        color = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        if rng.random() < 0.5:
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0))
        else:
            draw.ellipse([x, y, x + w, y + h], fill=color, outline=(255, 255, 255))
    return img


def jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _search(client, raw: bytes) -> dict:
    return client.post(
        "/v1/search",
        files={"file": ("q.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "true"},
    ).json()


def _web_ctx(ctx, img_src="https://cdn.example/c.jpg"):
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["searx"].reverse_hits = [
        ImageHit(title="candidate", url="https://site.example/post", img_src=img_src, engine="bing")
    ]


# ------------------------------------------------------------- face identity
def test_match_identity_returns_every_photo_of_the_person():
    """Regression: the old mutual-nearest-neighbour test dropped a person's
    photos whenever they appeared more than once in the library (each copy's
    runner-up was the other copy)."""
    f = FaceId(enabled=False, same_threshold=0.42)
    rng = np.random.default_rng(0)
    person = rng.normal(size=512)
    person /= np.linalg.norm(person)

    def noisy(v, s):
        x = v + rng.normal(scale=s, size=512)
        return x / np.linalg.norm(x)

    stranger = noisy(rng.normal(size=512), 0)
    candidates = [(1, noisy(person, 0.03)), (2, noisy(person, 0.03)), (3, noisy(person, 0.03)), (4, stranger)]
    matched = f.match_identity(noisy(person, 0.03), candidates)
    assert sorted(i for _, i in matched) == [1, 2, 3]
    assert all(sim >= 0.42 for sim, _ in matched)


def test_confidence_tier_reads_like_facecheck():
    f = FaceId(enabled=False, same_threshold=0.42)
    assert f.confidence_tier(0.85) == (99, "certain")
    conf, tier = f.confidence_tier(0.45)
    assert tier == "uncertain" and 70 <= conf < 80
    conf, tier = f.confidence_tier(0.10)
    assert tier == "weak" and conf < 70


@pytest.mark.skipif(
    not (FACE_MODELS / "w600k_mbf.onnx").exists() or not SAMPLE_FACE.exists(),
    reason="face models / sample face not available",
)
def test_face_embedding_is_pose_and_scale_stable():
    """With correct ArcFace alignment + normalization, a mirrored and
    rescaled copy of a face embeds as the same identity (the old pipeline
    fed misaligned, unnormalized crops and put strangers at ~0.74)."""
    f = FaceId(enabled=True, models_dir=str(FACE_MODELS))
    if not f.available:
        pytest.skip("face models failed to load")
    img = Image.open(SAMPLE_FACE).convert("RGB")
    a = f.embed_image(img)
    b = f.embed_image(ImageOps.mirror(img).resize((int(img.width * 0.8), int(img.height * 0.8))))
    assert a and b
    assert f.match(a[0]["embedding"], b[0]["embedding"]) >= 0.80
    # a different "face" (the same crop scrambled) is nowhere near
    scrambled = Image.fromarray(np.asarray(img)[:, ::-1][::-1].copy())
    c = f.embed_face(scrambled, [[30, 30], [60, 30], [45, 45], [33, 60], [57, 60]])
    if c is not None:
        assert f.match(a[0]["embedding"], c) < 0.42


# ----------------------------------------------------- geometric verification
def test_geometry_accepts_crop_rejects_unrelated():
    from app import geometry

    base = textured(1)
    q = geometry.Keypoints.from_image(base)
    qm = geometry.Keypoints.from_image(geometry.mirror(base))
    crop = ImageOps.expand(base.crop((90, 60, 420, 330)), border=30, fill=(255, 255, 255))
    good = geometry.verify(q, qm, crop)
    bad = geometry.verify(q, qm, textured(2))
    assert good is not None and good.inliers >= 40
    assert bad is None or bad.inliers < 18


def test_cropped_bordered_copy_is_verified_as_modified(client, ctx, monkeypatch):
    """TinEye's core trick: a crop with a border breaks every hash, but the
    keypoints still prove it is the same picture."""
    base = textured(3)
    edited = ImageOps.expand(base.crop((100, 70, 430, 330)), border=40, fill=(250, 250, 250))
    _web_ctx(ctx)
    patch_image_fetch(monkeypatch, jpeg(edited))
    d = _search(client, jpeg(base))
    web = [r for r in d["results"] if r["source"] == "web"]
    assert web and web[0]["match_type"] == "modified"
    assert "geometry" in web[0]["ai_modalities"]
    assert web[0]["geo_inliers"] >= 40
    assert d["found"] is True


def test_unrelated_candidate_stays_unconfirmed(client, ctx, monkeypatch):
    _web_ctx(ctx)
    patch_image_fetch(monkeypatch, jpeg(textured(4)))
    d = _search(client, jpeg(textured(5)))
    assert d["found"] is False
    assert all(r["match_type"] in ("unverified",) for r in d["results"] if r["source"] == "web")


def test_mirrored_copy_is_a_near_duplicate(client, ctx, monkeypatch):
    base = textured(6)
    ctx["settings"].geometric_verify_enabled = False  # prove the hash path alone
    _web_ctx(ctx)
    patch_image_fetch(monkeypatch, jpeg(ImageOps.mirror(base)))
    d = _search(client, jpeg(base))
    web = [r for r in d["results"] if r["source"] == "web"]
    assert web and web[0]["match_type"] in ("exact", "near")
    assert "mirrored" in web[0]["ai_modalities"]


# ------------------------------------------------------------ AI layers (fakes)
class FakeClip:
    """Deterministic 'CLIP': embedding = normalized mean color + brightness.
    Similar colors -> similar content."""

    available = True
    provider = "CPU"

    def embed_image(self, img):
        arr = np.asarray(img.convert("RGB").resize((16, 16)), dtype=np.float32).reshape(-1, 3).mean(0)
        v = np.concatenate([arr, [128.0]])
        return (v / np.linalg.norm(v)).tolist()

    @staticmethod
    def match(a, b):
        return float(np.dot(a, b)) if a and b and len(a) == len(b) else 0.0


def _client_with(ctx, clip=None, face=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import build_api

    app = FastAPI()
    app.include_router(
        build_api(
            ctx["settings"], ctx["repo"], ctx["matcher"], ctx["indexer"], ctx["auth"], ctx["searx"],
            ctx["limiter"], ctx["google"], ctx["bing"], ctx["yandex"], clip=clip, face=face,
        )
    )
    return TestClient(app)


def test_local_ai_grid_surfaces_similar_content(ctx):
    """Regression: _local_ai_grid built SearchResult with fields the model
    did not have; the TypeError was swallowed and the AI grid NEVER returned
    anything. Similar content from the index must now reach the main grid,
    labeled as a lead (not counted as found)."""
    clip = FakeClip()
    ctx["settings"].clip_similar_min_sim = 0.95
    other = Image.new("RGB", (200, 150), (200, 40, 40))
    ImageDraw.Draw(other).rectangle([10, 10, 60, 60], fill=(190, 50, 30))  # different pixels, same palette
    raw_other = jpeg(other)
    image_id = index_image(ctx["repo"], ctx["settings"], raw_other, url="https://lib.example/red.jpg")
    meta = ctx["repo"].get_image(image_id)
    meta.clip_embedding = clip.embed_image(other)
    ctx["repo"].put_image(meta)

    query = Image.new("RGB", (160, 120), (205, 45, 38))
    ImageDraw.Draw(query).ellipse([30, 20, 120, 100], fill=(195, 40, 45))
    client = _client_with(ctx, clip=clip)
    d = client.post("/v1/search", files={"file": ("q.jpg", jpeg(query), "image/jpeg")}, data={"include_web": "false"}).json()
    sims = [r for r in d["results"] if r["match_type"] == "similar"]
    assert sims and sims[0]["image_id"] == image_id
    assert sims[0]["clip_sim"] >= 0.95 and sims[0]["ai_modalities"] == ["clip"]
    assert "ai" in d["providers"]
    assert d["found"] is False  # a lead, not a confirmed match


# --------------------------------------------------------------- real time
def test_verification_respects_budget(ctx, monkeypatch):
    """A hanging candidate host must not hold the response past the budget
    (the old `with ThreadPoolExecutor` waited for every straggler)."""

    def slow_get(url, **kwargs):
        time.sleep(3)
        raise RuntimeError("too slow")

    monkeypatch.setattr("app.matcher.httpx.get", slow_get)
    hits = [ImageHit(title=f"h{i}", url=f"https://s{i}.example/p", img_src=f"https://s{i}.example/i.jpg") for i in range(4)]
    feats = {"phash": 0, "dhash": 0, "ahash": 0}
    started = time.monotonic()
    out = ctx["matcher"].verify_web_hits(feats, hits, budget_seconds=0.5)
    assert time.monotonic() - started < 2.0
    assert len(out) == 4 and all(r.match_type == "unverified" for r in out)
    assert any("budget" in r.note for r in out)
