#!/usr/bin/env python3
"""Verify: (1) does the rendered FB page contain stale/cached scontent URLs,
(2) which URL patterns actually fetch (real photo vs preview/sticker)."""
import os
import re
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import httpx

from app.cookies import CookieFile

cf = CookieFile("gui-cookies.txt")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
TUNNEL = "http://127.0.0.1:8888"

# 1) fresh render with bypass_cache to see ALL scontent URLs
url = "https://www.facebook.com/photo.php?fbid=2753647544906463&set=a.503814651757085&type=3&from_lookaside=1"
print("=== fresh render (bypass_cache) ===")
t0 = time.time()
r = httpx.post(
    "http://127.0.0.1:8081/fetch",
    json={"url": url, "method": "GET", "force_strategy": "gui_chrome", "timeout": 45, "bypass_cache": True},
    timeout=100,
)
print(f"render {time.time()-t0:.1f}s status={r.status_code}")
d = r.json()
html = d.get("html") or ""
print("html len:", len(html), "scontent:", html.count("scontent"))

# extract ALL image-ish scontent URLs
imgs = re.findall(r"https://scontent[^\"'\\ <]+?\.(?:jpe?g|png|webp)(?:\?[^\"'\\ <]*)?", html)
uniq = []
for i in imgs:
    if i not in uniq:
        uniq.append(i)
print("unique image urls:", len(uniq))
for i, u in enumerate(uniq[:8]):
    print(f"  [{i}] {u[:100]}")

# 2) fetch each and report which works
print("\n=== fetch each URL ===")
for i, u in enumerate(uniq[:8]):
    if "emoji" in u or ".mp4" in u:
        continue
    try:
        resp = httpx.get(u, headers={"User-Agent": UA, "Cookie": cf.header_for("scontent.fdac165-1.fna.fbcdn.net"),
                                     "Referer": "https://www.facebook.com/"}, follow_redirects=True, proxy=TUNNEL, timeout=20)
        good = resp.status_code == 200 and len(resp.content) > 5000 and "image" in (resp.headers.get("content-type") or "")
        print(f"  [{i}] {resp.status_code} {len(resp.content)}b ct={(resp.headers.get('content-type') or '')[:20]} {'OK' if good else 'FAIL'}")
        if good and i == 0:
            open("data/fb-fresh.bin", "wb").write(resp.content)
    except Exception as e:
        print(f"  [{i}] ERR {type(e).__name__} {str(e)[:60]}")

# 3) the user's exact URL pattern
print("\n=== user-reported pattern (t39.30808-6) ===")
for u in uniq:
    if "t39.30808" in u:
        print("  found in render:", u[:110])
        try:
            resp = httpx.get(u, headers={"User-Agent": UA, "Cookie": cf.header_for("scontent.fdac165-1.fna.fbcdn.net"),
                                         "Referer": "https://www.facebook.com/"}, follow_redirects=True, proxy=TUNNEL, timeout=20)
            print("  fetch:", resp.status_code, len(resp.content), "b", "| body:", resp.text[:120])
        except Exception as e:
            print("  fetch ERR:", str(e)[:80])
        break
else:
    print("  no t39.30808 URL in fresh render")
