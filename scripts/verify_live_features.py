"""Live verification of the new features against the real StrikeDB (:6008).

Reads/writes the imify namespaces but restores state afterwards (the removed
test image is re-enqueued; the demo tracker is deleted). Touches NO process.
"""

from __future__ import annotations

from app.config import get_settings
from app.engines.google_images import GoogleImagesReverse
from app.indexer import Indexer
from app.matcher import Matcher
from app.repo import Repository
from app.searx import SearxClient
from app.storage import get_storage

settings = get_settings()
settings.ensure_dirs()
storage = get_storage(settings)
repo = Repository(storage)
searx = SearxClient(settings.searxng_url, timeout=settings.searxng_timeout_seconds, min_interval=0, max_pages=1)
google = GoogleImagesReverse(settings.proxify_fetch_url, use_browser=settings.proxify_use_browser)
indexer = Indexer(repo, settings, searx, google)

print(f"backend={settings.storage_backend} searxng={settings.searxng_url} proxify={settings.proxify_fetch_url or '(not configured)'}")
print(f"images in index: {repo.image_count()}")

# ---- 1. pick the webp test image and register a demo tracker ---------------
webp = next((m for m in repo.all_images() if m.url.endswith("images.webp")), None)
if webp is None:
    print("SKIP: images.webp not in index")
else:
    print(f"\n[tracker] webp image id={webp.id} phash={webp.phash:016x}")
    settings.tracker_interval_seconds = 0  # always due for the demo round
    track_id = repo.upsert_track(webp.sha256, webp.phash, public_url="", title="demo: elon webp")
    print(f"[tracker] created track #{track_id} (public_url empty -> honest degradation)")
    processed = indexer.run_tracker_round()
    t = repo.get_track(track_id)
    print(f"[tracker] round processed={processed} last_run={t['last_run'] > 0} note={t['note']!r}")
    # now with the local URL as the target (proves the SearXNG reverse is hit)
    repo.upsert_track(webp.sha256, webp.phash, public_url="http://127.0.0.1:8099/images/images.webp", title="demo: elon webp")
    indexer.run_tracker_round()
    t = repo.get_track(track_id)
    print(f"[tracker] round with local URL -> note={t['note']!r} (SearXNG was hit; 0 hits expected, URL not public)")
    repo.delete_track(track_id)
    print(f"[tracker] demo tracker #{track_id} deleted (state restored)")

# ---- 2. removal demo + restore --------------------------------------------
variant = next((m for m in repo.all_images() if "variant" in m.url), None)
if variant is None:
    print("\nSKIP: variant test image not in index")
else:
    print(f"\n[removal] variant id={variant.id} url={variant.url}")
    ok = repo.remove_image(variant.id)
    print(f"[removal] removed={ok} images now={repo.image_count()}")
    if ok:
        # restore: re-enqueue so the running worker re-indexes it
        re_enqueued = repo.enqueue_pending(variant.url, {"page_url": variant.page_url, "title": variant.title, "engine": "local-test", "query": "test"})
        print(f"[removal] re-enqueued for restore={re_enqueued}")

# ---- 3. purge demo (dry: count only) --------------------------------------
hits = [m for m in repo.all_images() if "8099" in m.url]
print(f"\n[purge] would remove {len(hits)} images matching '8099' — NOT executed (dry run)")

# ---- 4. google provider availability --------------------------------------
print(f"\n[google] provider available={google.available} (needs PROXIFY_FETCH_URL to be usable)")
print("DONE")
