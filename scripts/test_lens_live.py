"""LIVE test: Google Lens byte-upload reverse search (circle-to-search)."""

from __future__ import annotations

import base64
import time

import httpx

from app.engines.google_images import GoogleImagesReverse, _build_multipart, parse_google_results

PROXIFY_API = "http://127.0.0.1:8081"
PROXIFY_TUNNEL = "http://127.0.0.1:8888"

raw = open("images/images.webp", "rb").read()
print(f"query image: images.webp ({len(raw)} bytes)")

# ---- 1. direct path: POST to lens, follow the 303, parse -----------------
print("\n=== 1. DIRECT upload flow ===")
body, boundary = _build_multipart(raw, "images.webp", "image/webp")
t0 = time.monotonic()
try:
    resp = httpx.post(
        "https://lens.google.com/upload",
        content=body,
        timeout=60,
        follow_redirects=True,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/146.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://lens.google.com/",
        },
    )
    page = resp.text
    print(f"final url: {resp.url} status={resp.status_code} html={len(page)} bytes in {time.monotonic()-t0:.1f}s")
    hits = parse_google_results(page)
    print(f"parsed hits: {len(hits)}")
    for h in hits[:8]:
        print(f"   - {h.title[:45]!r} img={h.img_src[:60]} page={h.url[:55]} engine={h.engine}")
    # save a sample for parser debugging
    open("data/lens-direct.html", "w", encoding="utf-8", errors="replace").write(page)
except Exception as exc:
    print(f"direct FAILED: {type(exc).__name__}: {exc}")

# ---- 2. through proxify /fetch ------------------------------------------
print("\n=== 2. proxify /fetch flow ===")
t0 = time.monotonic()
try:
    r = httpx.post(
        f"{PROXIFY_API}/fetch",
        json={
            "url": "https://lens.google.com/upload",
            "method": "POST",
            "body_b64": base64.b64encode(body).decode("ascii"),
            "headers": {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://lens.google.com/",
            },
            "timeout": 60,
            "use_browser": False,
        },
        timeout=90,
    )
    d = r.json()
    print(f"success={d.get('success')} status={d.get('status_code')} strategy={d.get('strategy_used')} "
          f"html={len(d.get('html') or '')} in {time.monotonic()-t0:.1f}s error={d.get('error')}")
    html = d.get("html") or ""
    if d.get("success"):
        hits = parse_google_results(html)
        print(f"parsed hits: {len(hits)}")
        for h in hits[:8]:
            print(f"   - {h.title[:45]!r} img={h.img_src[:60]} page={h.url[:55]} engine={h.engine}")
        open("data/lens-proxify.html", "w", encoding="utf-8", errors="replace").write(html)
except Exception as exc:
    print(f"proxify FAILED: {type(exc).__name__}: {exc}")

print("\nDONE")
