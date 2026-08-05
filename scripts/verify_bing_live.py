"""Live proof of the reliable reverse path on the real stack:

1. Bing byte-upload reverse (POST base64 image) — works for images with NO
   public URL (the case where URL-based reverse returns junk).
2. Every candidate is fused-hash-verified against the query — the wrong
   Pinterest-style images are rejected as unverified, real matches accepted.
"""

from __future__ import annotations

from pathlib import Path

from app.engines.bing_images import BingImagesReverse, parse_bing_results
from app.features import decode_image, extract_features
from app.matcher import Matcher

BING = BingImagesReverse(
    proxify_fetch_url="http://127.0.0.1:8081",
    fetch_proxy_url="http://127.0.0.1:8888",
)

IMAGES = [
    ("images/images.webp", "Elon webp"),
    ("images/726002745_1391225456545660_7750183954756539772_n.jpg", "social-media JPG"),
]


def main() -> None:
    print("=" * 70)
    print("BING BYTE-UPLOAD REVERSE + FUSED-HASH VERDICT (live)")
    print("=" * 70)
    for rel, label in IMAGES:
        path = Path(rel)
        if not path.exists():
            print(f"\n[skip] {label}: missing")
            continue
        raw = path.read_bytes()
        img = decode_image(raw)
        assert img is not None
        qf = extract_features(img, raw)

        print(f"\n--- {label} ({len(raw)//1024} KB, {img.size}) ---")
        try:
            hits = BING.search_upload(raw, filename=path.name)
        except Exception as exc:  # noqa: BLE001
            print(f"  bing-upload FAILED: {exc}")
            continue
        print(f"  bing returned {len(hits)} candidates")
        accepted = rejected = 0
        for hit in hits[:10]:
            # download + fused-hash verdict (same as Matcher.verify_web_hits)
            try:
                import httpx

                r = httpx.get(hit.img_src, timeout=15, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
                cimg = decode_image(r.content)
                if cimg is None:
                    print(f"    [undecodable] {hit.img_src[:70]}")
                    continue
                feats = extract_features(cimg, r.content)
                fused, dp, _, _ = Matcher._fused_from_hashes(
                    qf, int(feats["phash"]), int(feats["dhash"]), int(feats["ahash"])
                )
                verdict = "MATCH" if fused <= 0.22 else "rejected"
                if verdict == "MATCH":
                    accepted += 1
                else:
                    rejected += 1
                print(
                    f"    [{verdict:>8}] fused={fused:.3f} hamming={dp:>2}  {hit.img_src[:72]}"
                )
            except Exception:  # noqa: BLE001
                print(f"    [fetch-fail] {hit.img_src[:70]}")
        print(f"  verdict: {accepted} accepted, {rejected} rejected-as-unverified")


if __name__ == "__main__":
    main()
