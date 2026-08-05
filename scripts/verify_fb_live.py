#!/usr/bin/env python3
"""Live proof of the cookie-aware Facebook resolution + full verify round.

1. Take a real Bing reverse result that pointed at lookaside.fbsbx.com and run
   it through Matcher.verify_web_hits with the real gui-cookies.txt session.
2. Show the candidate resolves through the location.href stub -> photo.php hop
   with cookies attached (if the target image is still live).
3. Run the full Bing upload -> verify chain and count verified vs rejected.
"""
import os
import re
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import httpx

from app.config import Settings
from app.features import decode_image, extract_features
from app.matcher import Matcher
from app.models import ImageHit
from app.repo import Repository
from app.storage import SqliteStorage

settings = Settings(
    storage_backend="sqlite",
    sqlite_path="data/live-proof.db",
    data_dir="data",
    cookie_file="gui-cookies.txt",
    web_verify_candidates=True,
    web_verify_concurrency=6,
)
settings.ensure_dirs()
repo = Repository(SqliteStorage(settings.sqlite_path))
matcher = Matcher(repo, settings)

# ---------- 1) real FB candidate from the saved Bing upload page ----------
raw = open("images/images.webp", "rb").read()
img = decode_image(raw)
qf = extract_features(img, raw)
print("query:", img.size, "phash:", hex(qf["phash"]))

text = open("data/bing-upload.html", encoding="utf-8", errors="replace").read()
fb_urls = re.findall(r"https://lookaside\.fbsbx\.com/[^&\s\"'<>]+", text)
fb_urls = [u.replace("&quot;", "").rstrip("&,") for u in fb_urls]
print("\n=== 1) Facebook candidates from Bing (", len(set(fb_urls)), "unique) ===")
for u in sorted(set(fb_urls))[:2]:
    hit = ImageHit(title="facebook", url="https://www.facebook.com/", img_src=u, engine="bing")
    t0 = time.time()
    results = matcher.verify_web_hits(qf, [hit])
    dt = time.time() - t0
    r = results[0] if results else None
    verdict = f"{r.match_type} score={r.score} hamming={r.hamming}" if r else "no result"
    print(f"  {u[:80]}\n    -> {verdict}  ({dt:.1f}s)")

# ---------- 2) full Bing upload -> verify round ----------
print("\n=== 2) Bing byte-upload -> verify round ===")
from app.engines.bing_images import BingImagesReverse

bing = BingImagesReverse(proxify_fetch_url="http://127.0.0.1:8081", fetch_proxy_url="http://127.0.0.1:8888")
hits = bing.search_upload(raw, filename="q.webp")
print("bing upload candidates:", len(hits))
results = matcher.verify_web_hits(qf, hits)
verified = [r for r in results if r.match_type != "unverified"]
rejected = [r for r in results if r.match_type == "unverified"]
print(f"verified: {len(verified)}  unverified/rejected: {len(rejected)}")
for r in verified[:5]:
    print(f"  MATCH {r.match_type} h={r.hamming} {r.url[:80]}")
for r in rejected[:3]:
    print(f"  reject h={r.hamming} {r.url[:70]}")

print("\nDONE")
