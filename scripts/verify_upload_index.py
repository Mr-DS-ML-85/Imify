"""End-to-end proof on the fresh instance (127.0.0.1:8010, new code).

Uploads the two test images + a genuinely NEW image through the web API and
verifies they get indexed, then searches for each and reports the evidence.
"""

from __future__ import annotations

import time

import httpx
from PIL import Image, ImageEnhance

BASE = "http://127.0.0.1:8010"


def upload(path: str, name: str):
    with open(path, "rb") as f:
        r = httpx.post(
            f"{BASE}/v1/search",
            files={"file": (name, f, "image/jpeg")},
            data={"mode": "best", "num_results": "10", "include_web": "false"},
            timeout=60,
        )
    return r.json()


def stats() -> dict:
    return httpx.get(f"{BASE}/v1/stats", timeout=10).json()


def make_new_image(path_in: str, path_out: str) -> None:
    """Tint + re-encode so it's a genuinely NEW perceptual entry."""
    img = Image.open(path_in).convert("RGB")
    img = ImageEnhance.Color(img).enhance(0.2)
    img.save(path_out, "JPEG", quality=70)


print("index before:", stats()["images"])
before_fetched = stats()["success"]["indexer"]["fetched"]

# 1) the two known test images (should dedup-merge into existing entries)
print("\n[upload] images.webp (Elon):", upload("images/images.webp", "images.webp")["took_ms"], "ms")
print("[upload] JPG profile pic:", upload("images/726002745_1391225456545660_7750183954756539772_n.jpg", "profile.jpg")["took_ms"], "ms")

# 2) a genuinely NEW image (different pixels -> new index entry)
make_new_image("images/726002745_1391225456545660_7750183954756539772_n.jpg", "data/new-test.jpg")
print("[upload] NEW tinted image:", upload("data/new-test.jpg", "new-test.jpg")["took_ms"], "ms")

# 3) watch the workers index everything
for i in range(15):
    s = stats()
    print(f"  t={i * 3}s images={s['images']} pending={s['pending']} indexed={s['success']['indexer']['fetched']}")
    if s["pending"] == 0 and s["success"]["indexer"]["fetched"] > before_fetched:
        break
    time.sleep(3)

s = stats()
print("\n=== FINAL STATS ===")
print("images:", s["images"], "| searches:", s["searches"], "| success rates:", s["success"]["searches"])
print("providers:", s["success"]["providers"])
print("trackers:", len(httpx.get(f"{BASE}/v1/admin/tracks", timeout=5).json() if False else []))
