"""Live reverse proof on the real stack:

- Bing byte-upload reverse (POST base64) through the proxify tunnel
- Yandex byte-upload reverse (rpt=imageview) through the proxify tunnel
- every candidate fetched THROUGH THE TUNNEL (Facebook CDN etc. decode)
- fused-hash verdict on each candidate

The point: engines find candidates; OUR hashes judge. And candidates that
previously 403'd from this IP (lookaside.fbsbx.com) now decode through the
proxify tunnel.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from app.engines.bing_images import BingImagesReverse
from app.engines.yandex_images import YandexImagesReverse
from app.features import decode_image, extract_features
from app.matcher import Matcher

TUNNEL = "http://127.0.0.1:8888"
BING = BingImagesReverse(proxify_fetch_url="http://127.0.0.1:8081", fetch_proxy_url=TUNNEL)
YANDEX = YandexImagesReverse(proxify_fetch_url="http://127.0.0.1:8081", fetch_proxy_url=TUNNEL)

IMAGES = [
    ("images/images.webp", "Elon webp"),
    ("images/726002745_1391225456545660_7750183954756539772_n.jpg", "social-media JPG"),
]


def verdict_row(hit, qf, via_tunnel: bool):
    try:
        r = httpx.get(
            hit.img_src,
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/146.0.0.0 Safari/537.36"},
            proxy=TUNNEL if via_tunnel else None,
        )
        if r.status_code >= 400:
            return f"    [http{r.status_code}] {hit.img_src[:66]}"
        cimg = decode_image(r.content)
        if cimg is None:
            return f"    [undecodable] {hit.img_src[:66]}"
        feats = extract_features(cimg, r.content)
        fused, dp, _, _ = Matcher._fused_from_hashes(qf, int(feats["phash"]), int(feats["dhash"]), int(feats["ahash"]))
        v = "MATCH" if fused <= 0.22 else "rejected"
        return f"    [{v:>8}] fused={fused:.3f} ham={dp:>2} {hit.img_src[:60]}"
    except Exception as exc:  # noqa: BLE001
        return f"    [fetch-fail {type(exc).__name__}] {hit.img_src[:50]}"


def run(label, raw, provider, via_tunnel: bool):
    qf = extract_features(decode_image(raw), raw)
    print(f"\n=== {label} via {provider.__class__.__name__} ({'tunnel' if via_tunnel else 'direct'}) ===")
    try:
        hits = provider.search_upload(raw, filename="q.jpg")
    except Exception as exc:  # noqa: BLE001
        print(f"  upload FAILED: {exc}")
        return
    print(f"  {len(hits)} candidates")
    accepted = rejected = 0
    for hit in hits[:12]:
        line = verdict_row(hit, qf, via_tunnel)
        if "MATCH" in line:
            accepted += 1
        elif "rejected" in line:
            rejected += 1
        print(line)
    print(f"  verdict: {accepted} accepted, {rejected} rejected-as-unverified")


def main() -> None:
    print("=" * 70)
    print("REVERSE LIVE — BING + YANDEX BYTE-UPLOAD THROUGH PROXIFY, HASH VERDICT")
    print("=" * 70)
    for rel, label in IMAGES:
        path = Path(rel)
        if not path.exists():
            print(f"\n[skip] {label}: missing")
            continue
        raw = path.read_bytes()
        run(label, raw, BING, via_tunnel=True)
        run(label, raw, YANDEX, via_tunnel=True)


if __name__ == "__main__":
    main()
