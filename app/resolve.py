"""Resolve a *page* URL (e.g. a Facebook photo page) to actual image bytes.

The ``/v1/search image_url`` path needs the raw pixels of an image. Direct
image URLs are trivial, but social photo pages are HTML:

- most sites: ``og:image`` meta tag;
- Facebook comet photo pages: no ``og:image`` — the photo URL lives inside
  JS JSON as *escaped* scontent URLs (``https:\\/\\/scontent...``). With the
  real session cookies (``CookieFile``) those signed URLs resolve to actual
  pixels;
- last resort: render the page through proxify's gui_chrome browser to reveal
  the signed URL, then fetch it.

Every step runs through the configured CONNECT tunnel (``fetch_proxy_url``)
and attaches the per-host session cookies, mirroring ``Matcher.fetch_bytes``.
Returns ``(bytes | None, resolved_url)`` and never raises — a page that can't
be resolved degrades the search instead of crashing it.
"""

from __future__ import annotations

import html as html_mod
import re
from urllib.parse import urlparse

import httpx

from app.cookies import CookieFile
from app.searx import DEFAULT_UA

# Terminates a candidate URL: quote, whitespace or an angle bracket.
_END = re.compile(r'["\'\s<>]')


def _photo_urls_from_dom(html: str) -> list[str]:
    """All candidate FB CDN image URLs in a page/DOM, HTML-entity and
    JSON-escaped-slash unescaped (the ``_nc_*``/``oe`` signature params are the
    auth — a raw ``&amp;`` breaks the hash and Facebook answers ``Bad URL
    hash``). Handles both rendered DOM (plain ``https://scontent...``) and
    JS-embedded JSON (``https:\\/\\/scontent...`` — every slash escaped).
    Prefers real photo variants over stickers/previews/reactions."""
    plain = re.findall(r"https://scontent[^\"'\s<>]+?\.(?:jpe?g|png|webp|gif)(?:\?[^\"'\s<>]*)?", html)
    # JSON-escaped form: https:\/\/scontent...\/v\/...jpg — the class must
    # ALLOW backslashes (the escaped URLs are full of \/) and terminate only
    # at quotes / whitespace / angle brackets.
    escaped = re.findall(r"https:\\/\\/scontent[^\"'\s<>]+?\.(?:jpe?g|png|webp|gif)(?:\\?[^\"'\s<>]*)?", html)
    urls = [html_mod.unescape(u) for u in plain]
    urls += [html_mod.unescape(u).replace("\\/", "/") for u in escaped]
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        low = u.lower()
        if "emoji" in low or ".mp4" in low or "/m1/v/" in low or "t42" in low or "t39.1997" in low:
            continue  # stickers / videos / reaction assets — never the photo
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def resolve_image_bytes(
    url: str,
    cookie_file: str = "",
    proxy: str = "",
    proxify_fetch: str = "",
    timeout: float = 20.0,
    max_bytes: int = 20 * 1024 * 1024,
) -> tuple[bytes | None, str]:
    """Resolve ``url`` to image bytes. Returns (content, resolved_url)."""
    cf = CookieFile(cookie_file) if cookie_file else None
    proxify = proxify_fetch.rstrip("/") if proxify_fetch else ""
    hops: list[str] = []

    def _fetch(target: str, referer: str | None = None, depth: int = 0):
        nonlocal hops
        if depth > 3:
            return None, target
        try:
            host = (urlparse(target).hostname or "").lower()
            headers = {"User-Agent": DEFAULT_UA}
            if referer:
                headers["Referer"] = referer
            if cf and host:
                cookie = cf.header_for(host)
                if cookie:
                    headers["Cookie"] = cookie
                    hops.append(f"cookies:{host}")
            hops.append("direct")
            resp = httpx.get(
                target,
                timeout=timeout,
                headers=headers,
                follow_redirects=True,
                proxy=proxy or None,
            )
        except Exception:  # noqa: BLE001
            return None, target
        if resp.status_code >= 400 or not resp.content:
            return None, target
        ctype = resp.headers.get("content-type", "").lower()
        if ctype.startswith("image/"):
            if len(resp.content) > max_bytes:
                return None, target
            return resp.content, str(getattr(resp, "url", "") or target)
        if "html" in ctype or (len(resp.content) < 8192 and resp.content.lstrip().startswith(b"<")):
            text = resp.text
            # og:image (most social/photo pages)
            m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', text)
            if not m:
                m = re.search(r'<meta[^>]+content=["\']([^"\']+)["\']\s+property=["\']og:image["\']', text)
            if m:
                hops.append("og:image")
                return _fetch(m.group(1), referer=target, depth=depth + 1)
            # Facebook: escaped scontent URLs inside the JS JSON
            cands = _photo_urls_from_dom(text)
            if cands:
                for c in cands:
                    content, _final = _fetch(c, referer=target, depth=depth + 1)
                    if content:
                        return content, c
            # Facebook comet page: render through gui_chrome to reveal the URL
            if proxify and ("facebook.com" in host or "fbsbx.com" in host):
                try:
                    r = httpx.post(
                        proxify + "/fetch",
                        json={
                            "url": target,
                            "method": "GET",
                            "force_strategy": "gui_chrome",
                            "timeout": 45,
                            "bypass_cache": True,
                        },
                        timeout=90,
                    )
                    rendered = (r.json().get("html") or "") if getattr(r, "status_code", 200) < 400 else ""
                    hops.append("render:gui_chrome")
                except Exception:  # noqa: BLE001
                    rendered = ""
                for c in _photo_urls_from_dom(rendered):
                    content, _final = _fetch(c, referer=target, depth=depth + 1)
                    if content:
                        return content, c
        return None, target

    return _fetch(url)
