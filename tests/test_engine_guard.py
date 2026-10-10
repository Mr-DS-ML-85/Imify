"""EngineGuard: block detection, exponential cooldown, pacing, and that a
cooling engine is skipped (not hammered) by live searches."""

from __future__ import annotations

import time

from app.engine_guard import EngineGuard, looks_blocked
from app.engines import EngineError
from app.models import ImageHit

from conftest import make_image_bytes


def test_block_signals_vs_plain_failures():
    assert looks_blocked("google blocked the upload (status=429)")
    assert looks_blocked("Our systems have detected unusual traffic from your computer")
    assert looks_blocked("https://www.google.com/sorry/index?continue=...")
    assert looks_blocked("yandex: showcaptcha")
    assert not looks_blocked("proxify /fetch unreachable: timed out")
    assert not looks_blocked("")


def test_cooldown_grows_exponentially_and_clears_on_success(ctx):
    g = EngineGuard(ctx["repo"].db, base_cooldown=100, max_cooldown=10_000)
    g.report("google-upload", ok=False, text="timed out")  # not a block: ignored
    assert g.allow("google-upload")[0] is True

    g.report("google-upload", ok=False, text="google blocked the fetch (status=403)")
    first = g.status(["google-upload"])["google-upload"]
    assert g.allow("google-upload")[0] is False and first["strikes"] == 1
    assert 80 <= first["cooling_down_s"] <= 140

    g.report("google-upload", ok=False, text="captcha")
    second = g.status(["google-upload"])["google-upload"]
    assert second["strikes"] == 2 and second["cooling_down_s"] > first["cooling_down_s"]

    g.report("google-upload", ok=True)
    assert g.status(["google-upload"])["google-upload"]["strikes"] == 0
    assert g.allow("google-upload")[0] is True


def test_cooldown_survives_a_new_guard_instance(ctx):
    EngineGuard(ctx["repo"].db).report("yandex-upload", ok=False, text="SmartCaptcha")
    assert EngineGuard(ctx["repo"].db).allow("yandex-upload")[0] is False


def test_pacing_spaces_calls_and_refuses_bursts(ctx):
    g = EngineGuard(ctx["repo"].db, intervals={"x": 0.2}, max_wait=1.0)
    t = time.monotonic()
    assert g.pace("x") and g.pace("x")
    assert time.monotonic() - t >= 0.13  # second call waited (0.2 * >= 0.7)
    slow = EngineGuard(ctx["repo"].db, intervals={"y": 30}, max_wait=0.5)
    assert slow.pace("y") is True
    assert slow.pace("y") is False  # would wait ~30s: skip instead of queueing


def test_daily_budget(ctx):
    g = EngineGuard(ctx["repo"].db, intervals={"z": 0}, daily_budget={"z": 2})
    assert g.pace("z") and g.pace("z")
    assert g.allow("z") == (False, "daily budget 2 used")


def test_search_skips_cooling_engine_and_backs_off_on_block(client, ctx):
    """A blocked engine is put to rest; the next search does not call it at
    all (degraded as cooldown) while the other engines keep answering."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].google_upload_enabled = True
    ctx["settings"].bing_upload_enabled = True
    ctx["settings"].yandex_upload_enabled = True
    ctx["yandex"].upload_hits = [ImageHit(title="y", url="https://y.example/p", img_src="https://y.example/i.jpg")]

    calls = {"n": 0}

    def blocked_upload(*a, **k):
        calls["n"] += 1
        raise EngineError("google blocked the upload (status=429)")

    ctx["google"].search_upload = blocked_upload
    raw1 = make_image_bytes(color=(10, 200, 30))
    d1 = client.post("/v1/search", files={"file": ("a.jpg", raw1, "image/jpeg")}, data={"include_web": "true"}).json()
    assert calls["n"] == 1
    assert "provider:google-upload" in d1["degraded"]

    raw2 = make_image_bytes(color=(200, 10, 30))
    d2 = client.post("/v1/search", files={"file": ("b.jpg", raw2, "image/jpeg")}, data={"include_web": "true"}).json()
    assert calls["n"] == 1  # not hammered while cooling down
    assert "cooldown:google" in d2["degraded"]
    stats = client.get("/v1/stats").json()["engine_guard"]["google-upload"]
    assert stats["cooling_down_s"] > 0 and stats["strikes"] == 1
