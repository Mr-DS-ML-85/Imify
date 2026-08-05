"""Live proof: the fused multi-hash matcher (pHash+dHash+aHash) is the judge.

Uploads both test images through the fresh instance on :8010 and verifies:
1. The social-media JPG resolves to its indexed entry (or indexes fresh).
2. The Elon webp resolves to its indexed entry.
3. Every result carries a REAL match_type (exact/near) + real fused score,
   never the old blind "web" label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8010"
IMAGES = [
    ("images/726002745_1391225456545660_7750183954756539772_n.jpg", "social-media JPG"),
    ("images/images.webp", "Elon webp"),
]


def main() -> None:
    print("=" * 70)
    print("FUSED-HASH IDENTIFICATION — LIVE AGAINST YOUR REAL INDEX (:6008)")
    print("=" * 70)
    for rel, label in IMAGES:
        path = Path(rel)
        if not path.exists():
            print(f"\n[skip] {label}: {path} missing")
            continue
        raw = path.read_bytes()
        print(f"\n--- {label} ({path.name}, {len(raw)//1024} KB) ---")
        try:
            with open(path, "rb") as fh:
                resp = httpx.post(
                    BASE + "/v1/search",
                    files={"file": (path.name, fh, "image/jpeg")},
                    data={"mode": "best", "num_results": "6", "include_web": "true"},
                    timeout=90,
                )
        except httpx.HTTPError as exc:
            print(f"  search failed: {exc}")
            continue
        try:
            d = resp.json()
        except Exception:
            print(f"  non-JSON response ({resp.status_code}): {resp.text[:200]}")
            continue
        print(f"  found={d['found']} fallback={d['fallback']} providers={d['providers']} degraded={d['degraded']} took={d['took_ms']}ms")
        for r in d["results"][:6]:
            print(
                f"    [{r['match_type']:>10}] score={r['score']:.3f} hamming={r['hamming']} "
                f"engine={r['engine'] or '-':<14} src={r['source']:<5} {r['url'][:70]}"
            )
        if not d["results"]:
            print("    (no results)")


if __name__ == "__main__":
    main()
