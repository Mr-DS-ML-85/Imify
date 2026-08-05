#!/usr/bin/env python3
"""Try different FB URL formats + proxify options to get a fully-hydrated render."""
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import httpx

FBID = "1391220289879510"
urls = {
    "new format": f"https://www.facebook.com/photo/?fbid={FBID}&set=a.646142547720625",
    "old format": f"https://www.facebook.com/photo.php?fbid={FBID}&set=a.646142547720625",
    "photo.php bare": f"https://www.facebook.com/photo.php?fbid={FBID}",
    "photo/? bare": f"https://www.facebook.com/photo/?fbid={FBID}",
    "mbasic": f"https://mbasic.facebook.com/photo.php?fbid={FBID}",
}

for name, url in urls.items():
    for extra in ({"bypass_cache": True}, {"bypass_cache": True, "timeout": 60}, {}):
        payload = {"url": url, "method": "GET", "force_strategy": "gui_chrome", **extra}
        try:
            r = httpx.post("http://127.0.0.1:8081/fetch", json=payload, timeout=110)
            d = r.json()
            html = d.get("html") or ""
            sc = len(re.findall(r"https://scontent[^\"'\\ <]+", html))
            cap = html.count("checkpoint") + html.count("captcha")
            print(f"[{name}][{list(extra.keys())}] s={d.get('status_code')} cached={d.get('cached')} "
                  f"html={len(html)} scontent={sc} wall={cap}")
            if sc and name == "old format":
                open("data/fb-oldfmt-rendered.html", "w", encoding="utf-8").write(html)
        except Exception as e:
            print(f"[{name}] ERR {type(e).__name__} {str(e)[:60]}")
