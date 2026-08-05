#!/usr/bin/env python3
"""Modern Yandex: GET the imageview page first (establishes yandex_cbir cookie),
then POST the upload — check whether the POST returns a cbir_id."""
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import httpx

from app.engines.yandex_images import SEARCH_URL, DESKTOP_UA, parse_yandex_results

raw = open("images/images.webp", "rb").read()
TUNNEL = "http://127.0.0.1:8888"

client = httpx.Client(proxy=TUNNEL, timeout=60, follow_redirects=True,
                      headers={"User-Agent": DESKTOP_UA, "Accept-Language": "en-US,en;q=0.9"})

# 1) GET imageview page -> cookies (yandex_cbir?)
r0 = client.get(f"{SEARCH_URL}?rpt=imageview")
print("GET imageview:", r0.status_code, "len:", len(r0.text))
print("  cookies now:", [c.name for c in client.cookies.jar][:8] if hasattr(client.cookies, 'jar') else [c for c in client.cookies][:8])
# look for yandex_cbir in the cookie jar
found_cbir = False
for c in client.cookies.jar:
    if "cbir" in c.name.lower():
        print("  CBIR COOKIE:", c.name, "=", c.value[:40])
        found_cbir = True
if not found_cbir:
    print("  no cbir cookie after GET")

# 2) POST upload with the session cookies
r = client.post(SEARCH_URL, data={"rpt": "imageview", "format": "json"},
                files={"upfile": ("q.webp", raw)})
print("\nPOST upload:", r.status_code, "len:", len(r.text))
# where did we end up?
print("  final url:", r.url)
# cbir in body?
cbir = re.findall(r'cbir[_A-Za-z]*["\':= ]{1,3}["\']?([A-Za-z0-9_\-\.]{10,80})', r.text)
print("  cbir refs:", cbir[:3])
# redirects?
print("  redirect history:", [str(h.url)[:70] for h in r.history][:3])
hits = parse_yandex_results(r.text)
print("  parsed hits:", len(hits))

# 3) also try with format param variations
for fmt in ({}, {"format": "json"}, {"rpt": "imageview"}):
    r2 = client.post(SEARCH_URL, data=fmt, files={"upfile": ("q.webp", raw)})
    cbir2 = re.findall(r'cbir[_A-Za-z]*["\':= ]{1,3}["\']?([A-Za-z0-9_\-\.]{10,80})', r2.text)
    print(f"\nPOST {fmt} -> {r2.status_code} len={len(r2.text)} cbir: {cbir2[:2]}")

# 4) the apphost 'image-details' with the reqid from THIS session
reqids = re.findall(r'"(?:parent-)?reqid"\s*:\s*"([^"]+)"', r.text)
if reqids:
    api = f"https://yandex.com/images-apphost/image-details?parent-reqid={reqids[0]}&rpt=imageview"
    r3 = client.get(api, headers={"Referer": f"{SEARCH_URL}?rpt=imageview", "X-Requested-With": "XMLHttpRequest"})
    print("\napphost w/ reqid:", r3.status_code, "len:", len(r3.text), "ct:", r3.headers.get("content-type", "")[:24])
    print("  body sample:", r3.text[:200])
