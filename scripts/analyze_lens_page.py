"""Inspect the real Google Lens results page to build a correct parser."""

from __future__ import annotations

import json
import re

page = open("data/lens-direct.html", encoding="utf-8", errors="replace").read()
print("size:", len(page))

markers = {
    "AF_initDataCallback": page.count("AF_initDataCallback"),
    "imgres": page.count("imgres"),
    '": "ou"': page.count('"ou"'),
    "gstatic": page.count("gstatic.com"),
    "udm=26": page.count("udm=26"),
    "visual matches": page.count("visual"),
    "lens": page.count("lens.google"),
    "data:image": page.count("data:image"),
    "search?vsrid": page.count("vsrid"),
}
print("markers:", markers)

# 1) imgres links (page_url + imgurl)
imgres = re.findall(r"https://www\.google\.com/imgres\?[^\"\\]+", page)
print("\nimgres links found:", len(imgres))
for u in imgres[:3]:
    print("   ", u[:160])

# 2) any 'ou' / 'ru' style JSON fields in init data
for m in re.finditer(r'"ou":"([^"]{10,120})"', page):
    print("ou field:", m.group(1)[:120])
    if m.start() > 50000:
        break

# 3) look at a chunk of the AF_initDataCallback content
idx = page.find("AF_initDataCallback")
if idx != -1:
    chunk = page[idx : idx + 3000]
    print("\n=== AF_initDataCallback chunk (first 1500) ===")
    print(chunk[:1500].replace("\\n", "\n"))

# 4) search for result-like JSON arrays: look for common Lens item keys
for key in ("af_visual", "b3fe", "actualUrl", "imageUrl", "pageTitle", "resultUrl", '"title"'):
    print(f"key {key!r}: {page.count(key)}")
