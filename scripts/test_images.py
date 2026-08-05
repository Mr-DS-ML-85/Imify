"""Live test: index the attached test images through the REAL pipeline path.

This bypasses the pending queue (so a concurrently-running old indexer cannot
double-process) and calls Indexer._process_task directly — the same code the
workers run. Afterwards it runs reverse searches through the running API.

Usage:
    uv run python scripts/test_images.py          # clean + reseed + search
    uv run python scripts/test_images.py --seed   # clean + reseed only
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from app.config import get_settings
from app.indexer import Indexer
from app.repo import Repository
from app.storage import get_storage

BASE = "http://127.0.0.1:8099"
ORIGINAL = f"{BASE}/images/726002745_1391225456545660_7750183954756539772_n.jpg"
WEBP = f"{BASE}/images/images.webp"
VARIANT = f"{BASE}/data/test/variant.jpg"


def make_variant() -> None:
    from PIL import Image

    src = "images/726002745_1391225456545660_7750183954756539772_n.jpg"
    img = Image.open(src)
    small = img.resize((140, 140), Image.Resampling.LANCZOS)
    out = Path("data/test")
    out.mkdir(parents=True, exist_ok=True)
    small.convert("RGB").save(out / "variant.jpg", "JPEG", quality=55)
    print(f"[test] created near-dup variant -> {out / 'variant.jpg'} (was {img.size}, now {small.size}, q55)")


def clean(repo: Repository) -> None:
    """Remove Imify test artifacts only — never touches key:* (admin keys)."""
    prefixes = ["img:meta:", "img:url:", "img:sha:", "img:phash:", "pending:", "claim:", "hist:"]
    removed = 0
    for prefix in prefixes:
        for key in repo.db.keys(prefix):
            repo.db.delete(key)
            removed += 1
    for stat in ("images", "indexed", "fetched", "failed", "deduped", "pipeline_last_progress"):
        repo.db.delete(f"stat:{stat}")
    repo.db.delete("img:seq")
    print(f"[clean] removed {removed} Imify keys (admin keys untouched)")


def seed() -> Repository:
    settings = get_settings()
    settings.ensure_dirs()
    storage = get_storage(settings)
    if not storage.ping():
        print("FAIL: StrikeDB unreachable on", settings.strikedb_port)
        sys.exit(1)
    repo = Repository(storage)
    clean(repo)
    make_variant()
    indexer = Indexer(repo, settings)  # no threads — we drive it directly
    tasks = [
        {"url": ORIGINAL, "page_url": ORIGINAL, "title": "instagram profile pic (original)", "engine": "local-test", "query": "test"},
        {"url": WEBP, "page_url": WEBP, "title": "elon musk portrait webp", "engine": "local-test", "query": "test"},
        {"url": VARIANT, "page_url": VARIANT, "title": "resized variant of profile pic", "engine": "local-test", "query": "test"},
    ]
    for task in tasks:
        ok = indexer._process_task(task)
        print(f"[seed] {task['url'].split('/')[-1]:45s} -> {'indexed' if ok else 'FAILED'}")
    storage.close()
    return repo


def search(upload_path: str, mode: str = "best", include_web: bool = False) -> dict:
    import httpx

    with open(upload_path, "rb") as fh:
        r = httpx.post(
            "http://127.0.0.1:8000/v1/search",
            files={"file": (upload_path.rsplit("/", 1)[-1], fh, "image/jpeg")},
            data={"mode": mode, "num_results": "20", "include_web": str(include_web).lower()},
            timeout=30,
        )
    print(f"[search] {upload_path} mode={mode} include_web={include_web} -> HTTP {r.status_code}")
    return r.json()


def main() -> None:
    seed()
    time.sleep(1)
    print("\n=== REVERSE SEARCH: upload the VARIANT (near-dup of the profile pic) ===")
    r = search("data/test/variant.jpg")
    for row in r.get("results", []):
        print(f"  {row['match_type']:6s} Δ{row.get('hamming', '?'):>3} score={row['score']:.3f} src={row['source']} -> {row['url']}")
    print(f"  providers={r.get('providers')} degraded={r.get('degraded')} took_ms={r.get('took_ms')}")

    print("\n=== REVERSE SEARCH: the webp (Elon Musk portrait) ===")
    r2 = search("images/images.webp", include_web=True)
    for row in r2.get("results", [])[:5]:
        print(f"  {row['match_type']:6s} Δ{row.get('hamming', '?'):>3} src={row['source']} -> {row['url']}")
    print(f"  providers={r2.get('providers')} degraded={r2.get('degraded')} took_ms={r2.get('took_ms')}")

    print("\n=== REVERSE SEARCH: the original profile pic ===")
    r3 = search("images/726002745_1391225456545660_7750183954756539772_n.jpg")
    for row in r3.get("results", [])[:5]:
        print(f"  {row['match_type']:6s} Δ{row.get('hamming', '?'):>3} src={row['source']} -> {row['url']}")
    print(f"  providers={r3.get('providers')} degraded={r3.get('degraded')} took_ms={r3.get('took_ms')}")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        seed()
    else:
        main()
