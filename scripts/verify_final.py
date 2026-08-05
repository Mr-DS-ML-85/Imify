"""Final evidence run on the fresh instance (127.0.0.1:8010).

1. A genuinely NEW image (flipped + cropped + text overlay) -> NEW index entry.
2. Search each test image -> exact local matches.
3. Success rates + admin endpoints.
4. Identification evidence for the social-media JPG (filename + text discovery).
"""

from __future__ import annotations

import io
import json
import time

import httpx
from PIL import Image, ImageDraw

BASE = "http://127.0.0.1:8010"
ADMIN = {"Authorization": "Bearer sk_demo_admin"}


def make_new_image(path_out: str) -> None:
    """Flip + crop + overlay text -> definitely a different perceptual hash."""
    img = Image.open("images/726002745_1391225456545660_7750183954756539772_n.jpg").convert("RGB")
    img = img.transpose(Image.FLIP_LEFT_RIGHT).crop((0, 0, img.width, img.height - 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, 180, 34], fill=(255, 200, 0))
    draw.text((14, 12), "NEW VARIANT 2026", fill=(0, 0, 0))
    img.save(path_out, "JPEG", quality=80)


def upload(path: str, name: str) -> dict:
    with open(path, "rb") as f:
        return httpx.post(
            f"{BASE}/v1/search",
            files={"file": (name, f, "image/jpeg")},
            data={"mode": "best", "num_results": "10", "include_web": "false"},
            timeout=60,
        ).json()


def stats() -> dict:
    return httpx.get(f"{BASE}/v1/stats", timeout=10).json()


def search_upload(path: str, name: str, mode: str = "best") -> dict:
    with open(path, "rb") as f:
        return httpx.post(
            f"{BASE}/v1/search",
            files={"file": (name, f, "image/jpeg")},
            data={"mode": mode, "num_results": "10", "include_web": "true"},
            timeout=90,
        ).json()


print("=== 1. GENUINELY NEW IMAGE -> NEW INDEX ENTRY ===")
before = stats()["images"]
make_new_image("data/variant-new.jpg")
r = upload("data/variant-new.jpg", "variant-new.jpg")
print(f"upload ok (took {r['took_ms']}ms, found={r['found']})")
for i in range(12):
    s = stats()
    if s["images"] > before:
        break
    time.sleep(3)
print(f"index images: {before} -> {stats()['images']}  (new entry created: {stats()['images'] > before})")

print("\n=== 2. SEARCHES (local exact matches) ===")
for path, name in [("images/images.webp", "images.webp"), ("images/726002745_1391225456545660_7750183954756539772_n.jpg", "profile.jpg"), ("data/variant-new.jpg", "variant-new.jpg")]:
    d = search_upload(path, name)
    top = d["results"][0] if d["results"] else {}
    print(f"{name}: found={d['found']} providers={d['providers']} fallback={d['fallback']} "
          f"top=Δ{top.get('hamming')} {top.get('match_type')} source={top.get('source')} took={d['took_ms']}ms")

print("\n=== 3. SUCCESS RATES + ADMIN ===")
s = stats()
print("search success:", s["success"]["searches"])
print("providers:", s["success"]["providers"])
tracks = httpx.get(f"{BASE}/v1/admin/tracks", headers=ADMIN, timeout=10).json()
print("trackers:", [(t["id"], t["title"], "public" if t["public_url"] else "local-upload") for t in tracks["tracks"]])
imgs = httpx.get(f"{BASE}/v1/admin/images?limit=10", headers=ADMIN, timeout=10).json()
print("admin images:", [(m["id"], m["engine"], m["url"][-40:]) for m in imgs["images"]])

print("\n=== 4. JPG IDENTIFICATION EVIDENCE ===")
print("filename pattern:", "726002745_1391225456545660_7750183954756539772_n.jpg")
print("  -> Instagram CDN naming: <user_id>_<post_id>_<n>.jpg  => user id 726002745")
ex = Image.open("images/726002745_1391225456545660_7750183954756539772_n.jpg").getexif()
print("EXIF:", {k: v for k, v in ex.items() if k in (271, 272, 305, 306, 315, 33434, 36867)} or "none")
print("caption/hoodie text: </> CODE BUILD SOLVE REPEAT")
