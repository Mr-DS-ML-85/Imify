#!/usr/bin/env python3
"""Full live e2e — the REAL Facebook resolution chain through Matcher.verify_web_hits:

   lookaside.fbsbx.com stub
     -> location.href -> photo.php (comet shell)
     -> proxify gui_chrome render -> signed scontent URL
     -> fetch with cookies through tunnel -> pixels
     -> fused-hash verdict against the query.

Uses the REAL running stack: proxify (:8081 fetch, :8888 tunnel), real
gui-cookies.txt session, and one of the actual lookaside media_ids Bing found.
"""
import os
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from app.config import Settings
from app.features import decode_image, extract_features
from app.matcher import Matcher
from app.models import ImageHit
from app.repo import Repository
from app.storage import SqliteStorage

settings = Settings(
    storage_backend="sqlite",
    sqlite_path="data/fb-chain-proof.db",
    data_dir="data",
    cookie_file="gui-cookies.txt",
    proxify_fetch_url="http://127.0.0.1:8081",
    fetch_proxy_url="http://127.0.0.1:8888",
    web_verify_candidates=True,
    web_verify_concurrency=2,
)
settings.ensure_dirs()
repo = Repository(SqliteStorage(settings.sqlite_path))
matcher = Matcher(repo, settings)

media_id = os.environ.get("FB_MEDIA_ID", "2753647544906463")
url = f"https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id={media_id}"
print(f"query candidate: {url}")

# The Bing-found media_ids belong to a random FB photo, not to our local test
# images — so the HASH VERDICT must be 'unverified' (correct!). What we're
# proving is that the CHAIN resolves to real pixels instead of dying at the
# stub. Use the real photo page's own image as query to prove the accept path.
photo_query = None
for cand in ("data/fb-real-image.bin", "data/fb-graph-2753647544906463.bin"):
    if os.path.exists(cand) and os.path.getsize(cand) > 5000:
        photo_query = cand
        break

if photo_query:
    qraw = open(photo_query, "rb").read()
    qimg = decode_image(qraw)
    print(f"query = the FB photo itself ({photo_query}, {qimg.size}) -> expect MATCH")
    qf = extract_features(qimg, qraw)
else:
    qraw = open("images/images.webp", "rb").read()
    qimg = decode_image(qraw)
    qf = extract_features(qimg, qraw)
    print(f"query = local test image ({qimg.size}) -> expect unverified (Bing's media_id is a different FB photo)")

hit = ImageHit(title="facebook", url="https://www.facebook.com/", img_src=url, engine="bing")
t0 = time.time()
results = matcher.verify_web_hits(qf, [hit])
dt = time.time() - t0
r = results[0] if results else None
print(f"\nverdict ({dt:.1f}s): {r.match_type} score={r.score} hamming={r.hamming}" if r else "no result")
if photo_query:
    ok = r is not None and r.match_type in ("exact", "near")
    print("ACCEPT PATH PROVEN" if ok else "accept path failed")
else:
    ok = r is not None and r.match_type == "unverified"
    print("REJECT PATH CORRECT" if ok else "reject path misbehaved")
