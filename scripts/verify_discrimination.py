"""Fused-hash discrimination proof: the judge must accept derivatives of the
same image and reject unrelated ones — the whole point of being TinEye-like
but powerful (Google's face limits are irrelevant; OUR hashes decide)."""

from __future__ import annotations

import io
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageEnhance

BASE = "http://127.0.0.1:8010"
SRC = Path("images/images.webp")  # Elon webp — already indexed


def search(name: str, raw: bytes) -> dict:
    resp = httpx.post(
        BASE + "/v1/search",
        files={"file": (name, raw, "image/jpeg")},
        data={"mode": "best", "num_results": "4", "include_web": "false"},
        timeout=90,
    )
    return resp.json()


def main() -> None:
    src = Image.open(SRC).convert("RGB")

    # 1) re-encode + slight brightness shift + tiny crop — a "derivative copy"
    der = src.resize((int(src.width * 0.92), int(src.height * 0.92)))
    der = ImageEnhance.Brightness(der).enhance(1.08)
    buf = io.BytesIO()
    der.save(buf, "JPEG", quality=62)
    derivative = buf.getvalue()

    # 2) watermark overlay (like a stolen repost with a site's logo)
    wm = src.copy()
    d = ImageDraw.Draw(wm)
    d.rectangle([0, 0, 140, 26], fill=(255, 255, 255))
    d.text((6, 4), "STOLEN-FROM-SITE", fill=(220, 0, 0))
    buf = io.BytesIO()
    wm.save(buf, "JPEG", quality=80)
    watermarked = buf.getvalue()

    # 3) a totally different image — random noise photo
    diff = Image.new("RGB", (200, 200), (250, 210, 90))
    d2 = ImageDraw.Draw(diff)
    for _ in range(40):
        import random

        x0, y0 = random.randint(0, 190), random.randint(0, 190)
        x1, y1 = random.randint(0, 190), random.randint(0, 190)
        d2.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    buf = io.BytesIO()
    diff.save(buf, "JPEG", quality=85)
    unrelated = buf.getvalue()

    print("=" * 70)
    print("FUSED-HASH JUDGE — derivatives accepted, unrelated rejected")
    print("=" * 70)
    for name, raw, expect_match in [
        ("derivative-copy.jpg", derivative, True),
        ("watermarked.jpg", watermarked, True),
        ("unrelated.jpg", unrelated, False),
    ]:
        d = search(name, raw)
        verdict = "MATCH" if d["found"] else "NO MATCH"
        ok = "OK " if d["found"] == expect_match else "FAIL"
        print(f"\n[{ok}] {name:22} -> found={d['found']} verdict={verdict} (expected {'MATCH' if expect_match else 'NO MATCH'})")
        for r in d["results"][:3]:
            print(f"      [{r['match_type']:>9}] score={r['score']:.3f} hamming={r['hamming']} {r['url'][:60]}")


if __name__ == "__main__":
    main()
