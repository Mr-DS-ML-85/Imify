#!/usr/bin/env python3
"""Compare unescaped vs raw scontent URL fetch + read the 403 body."""
import html
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import httpx

from app.cookies import CookieFile

cf = CookieFile("gui-cookies.txt")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
TUNNEL = "http://127.0.0.1:8888"

rendered = open("data/fb-rendered.html", encoding="utf-8", errors="replace").read()
raw_urls = re.findall(r"https://scontent[^\"'\\ <]+?\.(?:jpe?g|png|webp)(?:\?[^\"'\\ <]*)?", rendered)
print("raw urls:", len(raw_urls))
for i, u in enumerate(raw_urls[:3]):
    unesc = html.unescape(u)
    print(f"\n[{i}] raw:    {u[:80]}")
    print(f"    unescaped: {unesc[:80]}")
    print(f"    differs: {u != unesc}")
    # fetch both
    for label, url in (("raw", u), ("unescaped", unesc)):
        try:
            resp = httpx.get(url, headers={"User-Agent": UA, "Cookie": cf.header_for("scontent.fdac165-1.fna.fbcdn.net"),
                                           "Referer": "https://www.facebook.com/"}, follow_redirects=True, proxy=TUNNEL, timeout=20)
            print(f"    [{label}] {resp.status_code} {len(resp.content)}b ct={(resp.headers.get('content-type') or '')[:18]} body={resp.text[:60]!r}")
        except Exception as e:
            print(f"    [{label}] ERR {type(e).__name__} {str(e)[:60]}")
