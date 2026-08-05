"""LIVE circle-to-search test through the freshly-started proxify.

1. proxify health + cookie status
2. Google byte-upload reverse of the two test images via proxify /fetch (body_b64)
3. Google URL reverse plumbing check
"""

from __future__ import annotations

import time

import httpx

from app.engines.google_images import GoogleImagesReverse, parse_google_results

PROXIFY_API = "http://127.0.0.1:8081"
PROXIFY_TUNNEL = "http://127.0.0.1:8888"

print("=== proxify health ===")
try:
    r = httpx.get(f"{PROXIFY_API}/health", timeout=8)
    print("health:", r.status_code, r.text[:200])
except Exception as exc:
    print("health FAILED:", exc)

print("\n=== cookies loaded? ===")
try:
    r = httpx.get(f"{PROXIFY_API}/cookies/status", timeout=8)
    print(r.text[:250])
except Exception as exc:
    print("cookies status failed:", exc)

g = GoogleImagesReverse(proxify_fetch_url=PROXIFY_API, fetch_proxy_url=PROXIFY_TUNNEL, use_browser=True)

for path, name in [
    ("images/images.webp", "images.webp"),
    ("images/726002745_1391225456545660_7750183954756539772_n.jpg", "profile.jpg"),
]:
    raw = open(path, "rb").read()
    print(f"\n=== GOOGLE UPLOAD reverse: {name} ({len(raw)} bytes) ===")
    t0 = time.monotonic()
    try:
        hits = g.search_upload(raw, filename=name, content_type="image/jpeg")
        dt = time.monotonic() - t0
        print(f"OK in {dt:.1f}s -> {len(hits)} hits")
        for h in hits[:6]:
            print(f"   - {h.title[:50]!r} img={h.img_src[:70]} page={h.url[:60]} engine={h.engine}")
    except Exception as exc:
        dt = time.monotonic() - t0
        print(f"FAILED after {dt:.1f}s: {type(exc).__name__}: {exc}")
    time.sleep(2)

# plumbing check: can proxify fetch a normal page at all?
print("\n=== proxify /fetch plumbing (example.com) ===")
try:
    r = httpx.post(
        f"{PROXIFY_API}/fetch",
        json={"url": "https://example.com/", "timeout": 30},
        timeout=40,
    )
    d = r.json()
    print("success:", d.get("success"), "status:", d.get("status_code"), "strategy:", d.get("strategy_used"), "html_bytes:", len(d.get("html") or ""))
except Exception as exc:
    print("plumbing failed:", exc)

print("\nDONE")
