"""Yandex Images reverse image search provider.

Why Yandex: it is the least privacy-censored of the big reverse engines.
Google deliberately suppresses face results and Bing is US-centric, but
Yandex's imageview still returns exact and derivative matches for people —
the ideal fallback after SearXNG and Bing.

How it works:

Stage 1 (byte-upload, ``rpt=imageview`` POST):
    the raw pixels are POSTed to https://yandex.com/images/search — the only
    honest path for images with NO public URL. Routed through the proxify
    CONNECT tunnel first, then proxify's /fetch (POST body now supported),
    then direct. Classic responses embed a ``cbir-page`` JSON with
    img_href (image) + original_url/link (source page).

Stage 2 (browser render, when cbir-page is absent):
    Modern Yandex is JS-gated — the upload returns a comet shell whose
    results only exist after JS executes, and the byte-upload often never
    registers a ``cbirId`` at all. When stage 1 yields no hits, the imageview
    URL is rendered through proxify's real browser (gui_chrome, carrying the
    session cookies) and the rendered DOM is parsed. This is the path that
    actually returns results in 2026 (proven: 18 real hits for a public URL).

As with every engine, results are only candidates — the fused-hash matcher is
the judge.
"""

from __future__ import annotations

import base64
import html as html_mod
import json
import re
import urllib.parse

import httpx

from app.engines import EngineError
from app.logging import child_logger
from app.models import ImageHit

log = child_logger("imify.yandex", engine="yandex_images")

SEARCH_URL = "https://yandex.com/images/search"
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)


class YandexImagesReverse:
    def __init__(self, proxify_fetch_url: str = "", fetch_proxy_url: str = ""):
        self.proxify_fetch_url = proxify_fetch_url.rstrip("/")
        self.fetch_proxy_url = fetch_proxy_url

    @property
    def available(self) -> bool:
        return bool(self.proxify_fetch_url or self.fetch_proxy_url)

    # ------------------------------------------------------------- upload
    # Modern Yandex (2026) is JS-gated: a plain HTTP POST (``simple``) returns
    # a shell whose results only exist after JS executes. The browser POST
    # (gui_chrome, carrying the session cookies) registers the upload and the
    # rendered DOM carries the results — so the browser route is tried FIRST.
    def search_upload(self, image_bytes: bytes, filename: str = "image.jpg") -> list[ImageHit]:
        """Byte-upload reverse search (circle-to-search style, no public URL
        needed). Stage 1: proxify /fetch browser POST (gui_chrome + wait for
        the results container to render) — the proven path. Stage 2: proxify
        CONNECT tunnel -> direct as fallbacks."""
        errors: list[str] = []
        # The browser route's rendered page is the authoritative verdict (it
        # can honestly say "no matching results") — only fall through on real
        # errors (blocked/timeouts), never on an empty-but-rendered page.
        if self.proxify_fetch_url:
            try:
                page = self._upload_via_fetch(image_bytes, filename)
                return self._finish(page, "proxify-fetch-browser")
            except EngineError as exc:
                errors.append(f"proxify-fetch: {exc}")
        if self.fetch_proxy_url:
            try:
                page = self._upload_proxy(image_bytes, filename, proxy=self.fetch_proxy_url)
                return self._finish(page, "proxify-tunnel")
            except EngineError as exc:
                errors.append(f"proxify: {exc}")
        try:
            page = self._upload_proxy(image_bytes, filename, proxy=None)
            return self._finish(page, "direct")
        except EngineError as exc:
            errors.append(f"direct: {exc}")
        raise EngineError("yandex-upload failed via all paths: " + "; ".join(errors))

    def _upload_proxy(self, image_bytes: bytes, filename: str, proxy: str | None) -> str:
        try:
            resp = httpx.post(
                SEARCH_URL,
                data={"rpt": "imageview", "format": "json"},
                files={"upfile": (filename, image_bytes)},
                timeout=45,
                follow_redirects=True,
                proxy=proxy,
                headers={"User-Agent": DESKTOP_UA, "Accept-Language": "en-US,en;q=0.9"},
            )
            resp.raise_for_status()
            if resp.status_code in (403, 429):
                raise EngineError(f"yandex blocked the upload (status={resp.status_code})")
            return resp.text
        except httpx.HTTPError as exc:
            raise EngineError(f"yandex-upload failed (proxy={proxy or 'direct'}): {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — any transport failure degrades
            raise EngineError(f"yandex-upload failed (proxy={proxy or 'direct'}): {exc}") from exc

    def _upload_via_fetch(self, image_bytes: bytes, filename: str) -> str:
        """Binary upload through proxify /fetch rendered in the REAL browser
        (gui_chrome — session cookies + TLS persona + JS execution). The page
        is returned only after the results container (.CbirSites/.SerpList)
        renders — or after the wait times out ("no matching results" page),
        which is an honest outcome, not a pipeline failure."""
        boundary = f"----imify{id(image_bytes):x}"
        parts = [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="upfile"; filename="{filename}"\r\n'.encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            image_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        try:
            resp = httpx.post(
                self.proxify_fetch_url + "/fetch",
                json={
                    "url": SEARCH_URL,
                    "method": "POST",
                    "body_b64": base64.b64encode(b"".join(parts)).decode("ascii"),
                    "headers": {
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://yandex.com/",
                    },
                    "timeout": 45,
                    "use_browser": True,
                    "force_strategy": "gui_chrome",
                    # Yandex renders results into .serp-item / .CbirSites nodes
                    # asynchronously after the upload registers. CRITICAL: wait
                    # for actual RESULT items — the empty-page shell already
                    # contains .SerpList, so waiting on that returns the empty
                    # page (0 hits) before results ever render.
                    "wait_selector": '.serp-item, .CbirSites, .ImagesContentSearchItem, [class*="serp-item"]',
                    "bypass_cache": True,
                },
                timeout=80,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"proxify /fetch unreachable: {exc}") from exc
        except ValueError as exc:
            raise EngineError(f"proxify returned non-JSON: {exc}") from exc
        if data.get("success") is False:
            raise EngineError(f"proxify upload failed: {data.get('error', data.get('status_code', 'unknown'))}")
        return data.get("html") or ""

    def _finish(self, page: str, via: str) -> list[ImageHit]:
        hits = parse_yandex_results(page)
        log.info(
            "yandex-upload reverse",
            extra={"event": "yandex_upload_reverse", "via": via, "hits": len(hits)},
        )
        return hits

    # --------------------------------------------------------------- url
    def search(self, image_url: str) -> list[ImageHit]:
        """URL-based imageview. Stage 1: plain proxify fetch + parse. Stage 2:
        when the page is a JS shell (no cbir-page — Yandex 2026), render the
        imageview URL through proxify's real browser (gui_chrome) and parse the
        rendered DOM — the flow that actually returns hits today."""
        target = f"{SEARCH_URL}?rpt=imageview&url={urllib.parse.quote(image_url, safe='')}"
        html = self._fetch(target)
        if html:
            hits = parse_yandex_results(html)
            if hits:
                log.info(
                    "yandex reverse",
                    extra={"event": "yandex_reverse", "image_url": image_url, "hits": len(hits), "stage": 1},
                )
                return hits
        # Stage 2: JS-gated -> real browser render (proven path)
        if self.proxify_fetch_url:
            rendered = self._render(target)
            if rendered:
                hits = parse_yandex_results(rendered)
                log.info(
                    "yandex reverse",
                    extra={"event": "yandex_reverse", "image_url": image_url, "hits": len(hits), "stage": 2},
                )
                return hits
        return []

    def _fetch(self, target: str) -> str:
        if not self.proxify_fetch_url:
            raise EngineError("yandex url-reverse requires PROXIFY_FETCH_URL (private-box URL reverse returns junk)")
        try:
            resp = httpx.post(
                self.proxify_fetch_url + "/fetch",
                json={"url": target, "timeout": 45, "use_browser": False},
                timeout=70,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"proxify unreachable: {exc}") from exc
        except ValueError as exc:
            raise EngineError(f"proxify returned non-JSON: {exc}") from exc
        if data.get("success") is False:
            raise EngineError(f"proxify fetch failed: {data.get('error', data.get('status_code', 'unknown'))}")
        return data.get("html") or data.get("markdown") or ""

    def _render(self, target: str) -> str:
        """Render the imageview URL through proxify's real browser (gui_chrome,
        which carries the imported session cookies). Bypasses proxify's cache so
        URL signatures are fresh."""
        try:
            resp = httpx.post(
                self.proxify_fetch_url + "/fetch",
                json={
                    "url": target,
                    "method": "GET",
                    "force_strategy": "gui_chrome",
                    "timeout": 50,
                    "bypass_cache": True,
                },
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # noqa: BLE001 — rendering is best-effort
            return ""
        if data.get("success") is False:
            return ""
        return data.get("html") or ""


# ------------------------------------------------------------------ parse
def _decode(s: str) -> str:
    try:
        return html_mod.unescape(s)
    except Exception:  # noqa: BLE001
        return s


def parse_yandex_results(page: str) -> list[ImageHit]:
    """Parse Yandex imageview results.

    Yandex embeds the results as JSON inside the page. Two known shapes:

    1. cbir-page JSON with items carrying img_href (image) and original_url
       or link (source page);
    2. plain <img>/<a> markup.

    Results are deduped by image URL; Yandex's own CDN/proxy URLs are filtered.
    """
    hits: list[ImageHit] = []
    seen: set[str] = set()

    # Yandex UI/support pages are NOT results — a modern imageview response is
    # JS-gated and the shell contains auth/tune/support links. Emitting them as
    # "hits" would pollute the chain with junk that can never hash-match, so
    # they are filtered hard here.
    JUNK_HOSTS = (
        "passport.yandex",
        "yandex.com/support",
        "yandex.com/tune",
        "yandex.ru/support",
        "yandex.ru/tune",
        "yandex.com/company",
        "yandex.com/legal",
        "yandex.com/images/",
        "yandex.ru/images/",
    )

    def junk(url: str) -> bool:
        low = url.lower()
        if any(bad in low for bad in JUNK_HOSTS):
            return True
        # auth pages: match the path segment, not a raw substring (so a legit
        # image under /author/ or /authenticate/ is never dropped)
        try:
            path = urllib.parse.urlsplit(url).path
        except ValueError:
            path = ""
        return path.startswith("/auth")

    def add(img: str, page_url: str, title: str = "") -> None:
        img = _decode(img).strip()
        if not img or img.startswith("data:") or img in seen:
            return
        if junk(img):
            return
        if any(bad in img for bad in ("favicon", "yandex.net/")):
            return
        # The modern rendered DOM serves thumbnails from avatars.mds.yandex.net.
        # When the anchor carries a real SOURCE page, the CDN url is the actual
        # image bytes (perfect for hash verification) — keep it. Only drop bare
        # Yandex CDN noise when there is no source page context at all.
        if "avatars.mds.yandex.net" in img and not _decode(page_url).strip():
            return
        seen.add(img)
        hits.append(
            ImageHit(
                title=title or page_url or img,
                url=_decode(page_url).strip() or img,
                img_src=img,
                engine="yandex images",
            )
        )

    # 1) cbir-page JSON: { "cbir-page": { "content": [ { "img_href": .., "original_url": .., "link": .. } ] } }
    # Balanced-brace scan so the full object is captured (non-greedy regex
    # would stop at the first inner comma).
    for m in re.finditer(r'"cbir-page"\s*:\s*\{', page):
        start = m.end() - 1  # position of the '{'
        depth = 0
        i = start
        while i < len(page):
            if page[i] == "{":
                depth += 1
            elif page[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            continue
        try:
            blob = json.loads(page[start : i + 1])
        except json.JSONDecodeError:
            continue
        items = blob.get("content") or blob.get("items") or []
        for it in items if isinstance(items, list) else []:
            img = it.get("img_href") or it.get("image_href") or ""
            purl = it.get("original_url") or it.get("link") or ""
            if img:
                add(img, purl, it.get("title", ""))

    # 2) rendered-DOM result pairs: <a href="SOURCE"><img src="CDN thumb">
    #    (modern Yandex 2026 — the anchor carries the real source page, the
    #    img is the actual bytes on Yandex's CDN)
    for m in re.finditer(r'<a[^>]+href="(https?://[^"]+)"[^>]*>\s*<img[^>]+src="(https?://[^"]+?)"', page):
        add(m.group(2), m.group(1))
    for m in re.finditer(r'<img[^>]+src="(https?://[^"]+?)"[^>]*>\s*</a>', page):
        add(m.group(1), "")
    # 3) plain img/a markup
    for m in re.finditer(r'<img[^>]+src="(https?://[^"]+?)"[^>]*>', page):
        add(m.group(1), "")
    for m in re.finditer(r'<a[^>]+href="(https?://[^"]+)"[^>]*>', page):
        add(m.group(1), "")

    # 4) generic image URLs (deduped) — only when the page has NO ui junk,
    # i.e. it is a real (old-style) results page, never a JS shell.
    if not hits and not any(j in page for j in ("passport.yandex", "yandex.com/support", "cbirPreview")):
        for m in re.finditer(r'https?://[^\s"\'<>]+?\.(?:jpe?g|png|webp|gif)(?:\?[^\s"\'<>]*)?', page):
            url = m.group(0).rstrip(",.;)")
            if "yandex" not in url:
                add(url, "")
    return hits
