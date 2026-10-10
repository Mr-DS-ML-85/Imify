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
    # Modern Yandex (2026) is a TWO-STEP flow, not a single POST:
    #   1. register the raw bytes via POST images-apphost/image-download →
    #      a JSON response whose "url" encodes the cbir_id (the raw bytes are
    #      already in memory — no intermediate download needed).
    #   2. GET /images/search?cbir_id=<id>&rpt=imageview&cbir_page=sites →
    #      the real results page (HTML with .serp-item / cbir-page JSON).
    # A single POST to /images/search returns an empty shell (cbirId="",
    # "No matching results.") even for the most-indexed image — proven. So the
    # browser route (gui_chrome carrying session cookies) does both steps.
    def search_upload(self, image_bytes: bytes, filename: str = "image.jpg") -> list[ImageHit]:
        """Byte-upload reverse search (circle-to-search style, no public URL
        needed). Uses the true TWO-STEP flow: POST image-download (register the
        raw bytes, read the cbir_id) then GET the /images/search?cbir_id= URL.

        Carrier order (live-proven 2026): the bridge 2-step upload (POST
        image-download -> GET /images/search?cbir_id=) via the proxify CONNECT
        tunnel and direct both work and are fast; the /fetch browser path is the
        last resort. Only the first route that yields a real results page wins."""
        errors: list[str] = []
        if self.fetch_proxy_url:
            try:
                page = self._upload_proxy(image_bytes, filename, proxy=self.fetch_proxy_url)
                hits = self._finish(page, "proxify-tunnel")
                if hits:
                    return hits
            except EngineError as exc:
                errors.append(f"proxify: {exc}")
        try:
            page = self._upload_proxy(image_bytes, filename, proxy=None)
            hits = self._finish(page, "direct")
            if hits:
                return hits
        except EngineError as exc:
            errors.append(f"direct: {exc}")
        if self.proxify_fetch_url:
            try:
                page = self._upload_via_fetch(image_bytes, filename)
                hits = self._finish(page, "proxify-fetch-browser")
                if hits:
                    return hits
            except EngineError as exc:
                errors.append(f"proxify-fetch: {exc}")
        raise EngineError("yandex-upload failed via all paths: " + "; ".join(errors))

    # ------------------------------------------------------------------ 2-step
    def _image_download_url(self) -> str:
        # Endpoint the real front-end uses to register a byte-upload.
        return (
            "https://yandex.com/images-apphost/image-download"
            "?cbird=37&images_avatars_size=preview&images_avatars_namespace=images-cbir"
        )

    def _cbir_id_from_upload(self, page: str) -> str:
        """Extract cbir_id from the image-download JSON response. Live shape
        (2026)::

            {"cbir_id":"4761143/<token>","namespace":"images-cbir",
             "sizes":{"orig":{"path":"https://avatars.mds.yandex.net/get-images-cbir/4761143/<token>/orig"},
                      "preview":{"path":"..."}}}

        Also tolerates the older ``{"url":".../get-images-cbir/<id>/preview"}``
        shape. Returns the cbir_id (or "" if the upload did not register)."""
        m = re.search(r'"cbir_id"\s*:\s*"([^"]+)"', page)
        if m:
            return m.group(1)
        m = re.search(r'"url"\s*:\s*"([^"]+)"', page)
        if m:
            idm = re.search(r"get-images-cbir/([^/\"]+)/", m.group(1))
            if idm:
                return idm.group(1)
        for key in ("cbirId", "requestId", "idimg", "id"):
            m = re.search(r'"%s"\s*:\s*"([^"]+)"' % key, page)
            if m:
                return m.group(1)
        return ""

    def _orig_url_from_upload(self, page: str, cbir_id: str) -> str:
        """Extract the orig (full-size) CDN URL of the uploaded image from the
        image-download response, used as the ``url`` param in the follow-up GET.
        Falls back to the cbir_id-encoded path."""
        m = re.search(r'"orig"\s*:\s*\{[^}]*?"path"\s*:\s*"([^"]+)"', page)
        if m:
            return m.group(1)
        return f"https://avatars.mds.yandex.net/get-images-cbir/{cbir_id}/orig"

    def _upload_proxy(self, image_bytes: bytes, filename: str, proxy: str | None) -> str:
        """Plain-HTTP 2-step upload (tunnel/direct): register via
        image-download, then GET the cbir search page."""
        cookies = httpx.Cookies()

        def _get_session() -> httpx.Client:
            return httpx.Client(
                timeout=45,
                follow_redirects=True,
                proxy=proxy,
                cookies=cookies,
                headers={
                    "User-Agent": DESKTOP_UA,
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )

        try:
            with _get_session() as client:
                # -- step 1: register the raw bytes (in-memory; no disk download)
                up = client.post(
                    self._image_download_url(),
                    content=image_bytes,
                    headers={
                        "Content-Type": "image/jpeg",
                        "Origin": "https://yandex.com",
                        "Referer": "https://yandex.com/images/search",
                        "Accept": "application/json",
                    },
                )
                up.raise_for_status()
                if up.status_code in (403, 429):
                    raise EngineError(f"yandex blocked the upload (status={up.status_code})")
                cbir_id = self._cbir_id_from_upload(up.text)
                if not cbir_id:
                    raise EngineError("yandex image-download returned no cbir id")
                orig_url = self._orig_url_from_upload(up.text, cbir_id)
                # -- step 2: fetch the results page for that cbir id
                target = (
                    f"{SEARCH_URL}?rpt=imageview&url={urllib.parse.quote(orig_url, safe='')}"
                    f"&cbir_id={urllib.parse.quote(cbir_id)}"
                    "&cbir_page=sites&family=0&pornowhitelist=1&ipnd=1"
                )
                resp = client.get(target)
                resp.raise_for_status()
                return resp.text
        except httpx.HTTPError as exc:
            raise EngineError(f"yandex-upload failed (proxy={proxy or 'direct'}): {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — any transport failure degrades
            raise EngineError(f"yandex-upload failed (proxy={proxy or 'direct'}): {exc}") from exc

    def _fetch_via_proxify(self, url: str, method: str = "GET", body: bytes | None = None,
                           content_type: str | None = None, strategy: str | None = None,
                           wait_selector: str | None = None) -> str:
        """proxify /fetch for the 2-step upload. Step 1 (POST image-download)
        is a fast JSON exchange — curl_cffi_plus carries it in ~2s. Step 2
        (GET the results page) is JS-gated — gui_chrome + wait_selector. The
        same proxify persona (session cookies) applies to both."""
        payload: dict = {
            "url": url,
            "method": method,
            "timeout": 45,
            "bypass_cache": True,
        }
        if strategy:
            payload["force_strategy"] = strategy
            payload["use_browser"] = strategy == "gui_chrome"
        if wait_selector:
            payload["wait_selector"] = wait_selector
        headers = {
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://yandex.com/images/search",
        }
        if body is not None:
            payload["body_b64"] = base64.b64encode(body).decode("ascii")
            if content_type:
                headers["Content-Type"] = content_type
                headers["Accept"] = "application/json"
        payload["headers"] = headers
        try:
            resp = httpx.post(self.proxify_fetch_url + "/fetch", json=payload, timeout=80)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"proxify /fetch unreachable: {exc}") from exc
        except ValueError as exc:
            raise EngineError(f"proxify returned non-JSON: {exc}") from exc
        if data.get("success") is False and data.get("status_code") not in (200,):
            raise EngineError(f"proxify fetch failed: {data.get('error', data.get('status_code', 'unknown'))}")
        return data.get("html") or ""

    def _upload_via_fetch(self, image_bytes: bytes, filename: str) -> str:
        """Two-step upload through proxify /fetch:
          1. POST image-download with the raw bytes (already in memory) via the
             fast curl_cffi persona → cbir_id (this is a JSON exchange).
          2. GET /images/search?cbir_id=... through gui_chrome — the JS-gated
             results page carrying .serp-item / cbir-page JSON."""
        # ----- step 1: register the bytes and read the cbir id
        up_page = ""
        try:
            up_page = self._fetch_via_proxify(
                self._image_download_url(),
                method="POST",
                body=image_bytes,
                content_type="image/jpeg",
                strategy="curl_cffi_plus",
            )
        except EngineError:
            up_page = ""
        cbir_id = self._cbir_id_from_upload(up_page) if up_page else ""
        if not cbir_id:
            raise EngineError("yandex image-download returned no cbir id (upload not registered)")
        orig_url = self._orig_url_from_upload(up_page, cbir_id)

        # ----- step 2: fetch the results page for that cbir id
        target = (
            f"{SEARCH_URL}?rpt=imageview&url={urllib.parse.quote(orig_url, safe='')}"
            f"&cbir_id={urllib.parse.quote(cbir_id)}"
            "&cbir_page=sites&family=0&pornowhitelist=1&ipnd=1"
        )
        try:
            page = self._fetch_via_proxify(
                target,
                method="GET",
                strategy="gui_chrome",
                wait_selector=".serp-item, .CbirSites, .ImagesContentSearchItem, [class*=\"serp-item\"]",
            )
        except EngineError:
            page = ""
        if not page:
            raise EngineError("yandex cbir results fetch failed")
        return page

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
        # Yandex appends organic-tracking params to result URLs — strip them so
        # dedup and the URL-based matcher see clean, comparable URLs.
        img = re.sub(r"(?i)[?&]utm_(medium|source|campaign|campaignid|content|term)=[^&]+", "", img).rstrip("?&")
        page_url = re.sub(r"(?i)[?&]utm_(medium|source|campaign|campaignid|content|term)=[^&]+", "", page_url).rstrip("?&")
        if not img or img.startswith("data:") or img in seen:
            return
        if junk(img):
            return
        if any(bad in img for bad in ("favicon", "yandex.net/")):
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
