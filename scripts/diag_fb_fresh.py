#!/usr/bin/env python3
"""Fresh render (bypass_cache) — which scontent URL is the main photo of fbid=1391220289879510?"""
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import httpx

url = "https://www.facebook.com/photo/?fbid=1391220289879510&set=a.646142547720625"
r = httpx.post("http://127.0.0.1:8081/fetch",
               json={"url": url, "method": "GET", "force_strategy": "gui_chrome", "timeout": 45, "bypass_cache": True},
               timeout=120)
d = r.json()
print("success:", d.get("success"), "strategy:", d.get("strategy_used"), "status:", d.get("status_code"),
      "cached:", d.get("cached"))
html = d.get("html") or ""
print("html len:", len(html))
open("data/fb-newfmt-fresh.html", "w", encoding="utf-8").write(html)

imgs = re.findall(r"https://scontent[^\"'\\ <]+", html)
print("\nscontent urls:", len(imgs))
seen = set()
for u in imgs:
    key = u.split("?")[0][-45:]
    if key in seen:
        continue
    seen.add(key)
    print("  ", u[:130])
    # context
    idx = html.find(u)
    ctx = re.sub(r"\s+", " ", html[max(0, idx - 130):idx + 10])[-150:]
    print("      ctx:", ctx[:140])

# does the fbid appear in any image url?
print("\nfbid in any url:", any("1391220289879510" in u for u in imgs))
# any alt text mentioning photo?
alts = re.findall(r'<img[^>]+alt="([^"]{3,120})"[^>]+src="(https://scontent[^"]+)"', html)
print("img+alt pairs:", len(alts))
for alt, src in alts[:6]:
    print("  alt:", alt[:60], "|", src[:80])
