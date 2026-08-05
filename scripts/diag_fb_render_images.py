#!/usr/bin/env python3
"""Render the new-format FB photo page and dump ALL image candidates with context."""
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import httpx

url = "https://www.facebook.com/photo/?fbid=1391220289879510&set=a.646142547720625"
r = httpx.post("http://127.0.0.1:8081/fetch",
               json={"url": url, "method": "GET", "force_strategy": "gui_chrome", "timeout": 45},
               timeout=100)
d = r.json()
print("success:", d.get("success"), "strategy:", d.get("strategy_used"), "status:", d.get("status_code"))
html = d.get("html") or ""
print("html len:", len(html))
open("data/fb-newfmt-rendered.html", "w", encoding="utf-8").write(html)

# all scontent image urls with the surrounding 120 chars to understand what each is
imgs = re.findall(r"https://scontent[^\"'\\ <]+", html)
print("\nall scontent urls:", len(imgs))
seen = set()
for u in imgs:
    key = u.split("?")[0][-40:]
    if key in seen:
        continue
    seen.add(key)
    print("  ", u[:120])
    idx = html.find(u)
    ctx = re.sub(r"\s+", " ", html[max(0, idx - 150):idx + 20])[-170:]
    print("      ctx:", ctx[:160])
