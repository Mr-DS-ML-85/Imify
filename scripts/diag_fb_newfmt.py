#!/usr/bin/env python3
"""Diagnose: (1) the new FB photo/?fbid= URL format decode, (2) the local upload path leak."""
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import httpx

from app.config import Settings
from app.cookies import CookieFile
from app.features import decode_image, extract_features
from app.matcher import Matcher
from app.models import ImageHit
from app.repo import Repository
from app.storage import SqliteStorage
from app.searx import DEFAULT_UA

settings = Settings(
    storage_backend="sqlite",
    sqlite_path="data/fb-newfmt.db",
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
cf = CookieFile("gui-cookies.txt")

url = "https://www.facebook.com/photo/?fbid=1391220289879510&set=a.646142547720625"
print("=== 1) new FB format direct fetch ===")
r = httpx.get(url, headers={"User-Agent": DEFAULT_UA, "Cookie": cf.header_for("www.facebook.com"),
                            "Referer": "https://www.facebook.com/"}, follow_redirects=True,
              proxy=settings.fetch_proxy_url, timeout=25)
print(f"status={r.status_code} len={len(r.content)} ct={r.headers.get('content-type','')[:26]}")

import re
body = r.text
for marker in ("location.href", "og:image", "scontent", "captcha", "checkpoint", "login"):
    print(f"  marker '{marker}':", body.count(marker))
m = re.search(r'location\.href\s*=\s*["\']([^"\']+)["\']', body)
print("  location.href:", m.group(1)[:80] if m else None)

# --- try the FULL matcher verify path with the query being the same FB image ---
print("\n=== 2) matcher verify on the photo page as candidate ===")
# query: use a downloaded FB image if present, else the test webp
qraw = None
for cand in ("data/fb-real-image.bin", "images/726002745_1391225456545660_7750183954756539772_n.jpg", "images/images.webp"):
    if os.path.exists(cand):
        qraw = open(cand, "rb").read()
        qimg = decode_image(qraw)
        if qimg:
            print("query:", cand, qimg.size)
            break
if qraw:
    qf = extract_features(qimg, qraw)
    hit = ImageHit(title="fb", url=url, img_src=url, engine="bing")
    results = matcher.verify_web_hits(qf, [hit])
    r0 = results[0] if results else None
    print("  verdict:", r0.match_type if r0 else "none", "| score:", r0.score if r0 else None, "| hamming:", r0.hamming if r0 else None)
