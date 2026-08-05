#!/usr/bin/env python3
"""Is the fresh render a checkpoint wall? What photo data is really in there?"""
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

html = open("data/fb-newfmt-fresh.html", encoding="utf-8", errors="replace").read()
print("len:", len(html))

for marker in ("checkpoint", "captcha", "Confirm your identity", "log in", "Log in",
               "Sorry", "temporarily", "photo.php", "photo/?fbid", "viewer",
               "PhotoViewer", "comet-photo", "image", "profile picture", "Profile"):
    print(f"  '{marker}':", html.count(marker))

# title
m = re.search(r"<title>([^<]*)</title>", html)
print("title:", m.group(1)[:80] if m else None)

# meta og tags present?
ogs = re.findall(r'<meta[^>]+(?:property|name)="og:([^"]+)"[^>]*>', html)
print("og tags:", ogs[:10])

# any fbcdn lookaside / video / non-scontent image hosts
for host in ("lookaside", "fbcdn.net/v/t51", "fbcdn.net/v/t1", "fbcdn.net/v/t39"):
    print(f"  urls with '{host}':", len(re.findall(host, html)))

# what text is visible? first 400 chars of body text
body = re.sub(r"<script[\s\S]*?</script>", "", html)
body = re.sub(r"<style[\s\S]*?</style>", "", body)
body = re.sub(r"<[^>]+>", " ", body)
body = re.sub(r"\s+", " ", body).strip()
print("\nbody text:", body[:400])
