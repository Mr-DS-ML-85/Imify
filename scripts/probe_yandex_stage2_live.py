#!/usr/bin/env python3
"""Probe Yandex modern stage-2 flows for the upload shell.

Three paths:
A) apphost image-details API using the parent-reqid from the upload shell
B) gui_chrome render of the upload URL (GET-only — probably no results)
C) upload with browser session cookies via tunnel (yandex_cbir cookie?)
"""
import os
import re
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import httpx

from app.engines.yandex_images import SEARCH_URL, DESKTOP_UA, parse_yandex_results

raw = open("images/images.webp", "rb").read()
TUNNEL = "http://127.0.0.1:8888"

# ---------- A) upload, extract reqid + cookies ----------
client = httpx.Client(proxy=TUNNEL, timeout=60, follow_redirects=True,
                      headers={"User-Agent": DESKTOP_UA, "Accept-Language": "en-US,en;q=0.9"})
r = client.post(SEARCH_URL, data={"rpt": "imageview", "format": "json"},
                files={"upfile": ("q.webp", raw)})
print(f"A) upload -> {r.status_code} len={len(r.text)}")
reqids = re.findall(r'"(?:parent-)?reqid"\s*:\s*"([^"]+)"', r.text)
print("   reqids:", [x[:50] for x in reqids[:3]])
cbir = re.findall(r'cbir[_A-Za-z]*["\':= ]{1,3}["\']?([A-Za-z0-9_\-\.]{10,80})', r.text)
print("   cbir refs:", cbir[:3])
# session cookies
print("   cookies:", [c.split(";")[0] for c in r.headers.get_list("set-cookie")][:4])

# ---------- B) apphost API with parent-reqid ----------
if reqids:
    for api in (
        f"https://yandex.com/images-apphost/image-details?parent-reqid={reqids[0]}",
        f"https://yandex.com/images-apphost/images?parent-reqid={reqids[0]}",
    ):
        try:
            r2 = client.get(api, headers={"Referer": "https://yandex.com/images/search"})
            print(f"B) {api[:70]}... -> {r2.status_code} len={len(r2.text)} ct={r2.headers.get('content-type','')[:24]}")
            if "json" in (r2.headers.get("content-type") or ""):
                import json
                d = r2.json()
                print("   keys:", list(d.keys())[:8] if isinstance(d, dict) else type(d))
                s = json.dumps(d)[:400]
                print("   sample:", s)
        except Exception as e:
            print(f"B) apphost err: {type(e).__name__} {str(e)[:80]}")

# ---------- C) parse what we got ----------
hits = parse_yandex_results(r.text)
print(f"C) parsed from shell: {len(hits)} hits")
for h in hits[:3]:
    print("   -", h.img_src[:80])

# ---------- D) gui_chrome render of the imageview page (no upload, just render) ----------
print("\nD) gui_chrome render of yandex imageview (public-url form of our upload server)")
# our private box has no public URL; render anyway to see the JS flow
try:
    rr = httpx.post("http://127.0.0.1:8081/fetch",
                    json={"url": "https://yandex.com/images/search?rpt=imageview&url=https%3A%2F%2Fwww.python.org%2Fstatic%2Fimg%2Fpython-logo.png",
                          "method": "GET", "force_strategy": "gui_chrome", "timeout": 50}, timeout=90)
    d = rr.json()
    html = d.get("html") or ""
    print(f"   render -> success={d.get('success')} strategy={d.get('strategy_used')} len={len(html)}")
    print("   markers: cbir-page:", html.count("cbir-page"), "img_href:", html.count("img_href"),
          "avatar:", html.count("avatars.mds"), "sizes:", html.count('"sizes"'))
    hits2 = parse_yandex_results(html)
    print(f"   parsed from render: {len(hits2)} hits")
    for h in hits2[:3]:
        print("   -", h.img_src[:80])
except Exception as e:
    print(f"D) render err: {type(e).__name__} {str(e)[:120]}")
