"""Dump what the Lens upload response actually contains."""

from __future__ import annotations

import re

page = open("data/lens-direct.html", encoding="utf-8", errors="replace").read()

# title + first text
title = re.search(r"<title[^>]*>(.*?)</title>", page, re.DOTALL)
print("title:", (title.group(1).strip() if title else "none")[:120])

# visible text (strip tags)
text = re.sub(r"<script.*?</script>", " ", page, flags=re.DOTALL)
text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL)
text = re.sub(r"<[^>]+>", " ", text)
text = re.sub(r"\s+", " ", text).strip()
print("\ntext length:", len(text))
print("text head:", text[:600])

# what scripts/urls are referenced?
refs = re.findall(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', page)
print("\nresource refs:", [r[:80] for r in refs[:12]])

# meta refresh / canonical
for m in re.finditer(r'<meta[^>]+(?:http-equiv="refresh"|content="[^"]*url=)[^>]*>', page, re.I):
    print("meta:", m.group(0)[:200])
canon = re.search(r'<link rel="canonical" href="([^"]+)"', page)
print("canonical:", canon.group(1)[:120] if canon else None)
