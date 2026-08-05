"""Bing Visual Search reverse image search provider.

Why Bing, and why byte-upload:

- Google's ``searchbyimage``/Lens needs a **publicly reachable** image URL and
  even then answers with captchas (``/sorry``) to datacenter IPs — proxify's
  browser gets captcha'd too. On a private/LAN box a URL-based reverse engine
  can't fetch ``127.0.0.1`` URLs at all and falls back to visually-similar
  JUNK (Bing does this: unreachable query -> random Pinterest pins).
- Bing's **byte-upload** endpoint (``POST /images/search`` with the image as
  base64 in ``image_url``) sends the actual pixels, so it works for images
  with no public URL. Results come back as HTML with ``murl`` (image) +
  ``purl`` (page) pairs.
- The ``imgurl:`` URL trick is kept as a fallback for genuinely public URLs.

Routing: proxify CONNECT tunnel first (consistent cookies/persona), then
direct as last resort. Per the project rule, every candidate returned here is
only a *candidate* — the fused-hash matcher judges whether it actually matches
the query (see ``Matcher.verify_web_hits``); unrelated results like the random
Pinterest pins are rejected as ``unverified``.
"""

from __future__ import annotations

import base64
import html as html_mod
import re

import httpx

from app.engines import EngineError
from app.logging import child_logger
from app.models import ImageHit

log = child_logger("imify.bing", engine="bing_images")

SEARCH_URL = "https://www.bing.com/images/search"
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)


class BingImagesReverse:
    def __init__(self, proxify_fetch_url: str = "", fetch_proxy_url: str = ""):
        # REST /fetch endpoint (text strategies; for the URL-based imgurl path).
        self.proxify_fetch_url = proxify_fetch_url.rstrip("/")
        # CONNECT tunnel — raw binary passes through; used for byte-upload.
        self.fetch_proxy_url = fetch_proxy_url

    @property
    def available(self) -> bool:
        return bool(self.proxify_fetch_url or self.fetch_proxy_url)

    # ------------------------------------------------------------- upload
    def search_upload(self, image_bytes: bytes, filename: str = "image.jpg") -> list[ImageHit]:
        """Byte-upload reverse search — works for images with NO public URL.

        POSTs the image as base64 ``image_url`` form data (the modern Bing
        visual-search endpoint). Route order: proxify /fetch browser POST
        (gui_chrome — session cookies, JS; fixes the plain-HTTP 400) -> proxify
        CONNECT tunnel -> direct. Parses ``murl``/``purl`` pairs from HTML.
        """
        b64 = base64.b64encode(image_bytes).decode("ascii")
        errors: list[str] = []
        # Route order: CONNECT tunnel first (proven to return visual-similar
        # candidates), then direct, then the browser POST as last resort.
        # Only the FIRST route that yields hits wins; empty results fall
        # through to the next (Bing's browser POST returns a 404 shell for
        # some images — never treat a shell as an authoritative verdict).
        if self.fetch_proxy_url:
            try:
                page = self._upload_proxy(b64, proxy=self.fetch_proxy_url)
                hits = self._finish(page, "proxify-tunnel")
                if hits:
                    return hits
            except EngineError as exc:
                errors.append(f"proxify: {exc}")
        try:
            page = self._upload_proxy(b64, proxy=None)
            hits = self._finish(page, "direct")
            if hits:
                return hits
        except EngineError as exc:
            errors.append(f"direct: {exc}")
        if self.proxify_fetch_url:
            try:
                page = self._upload_via_fetch(b64)
                hits = self._finish(page, "proxify-fetch-browser")
                if hits:
                    return hits
            except EngineError as exc:
                errors.append(f"proxify-fetch: {exc}")
        raise EngineError("bing-upload failed via all paths: " + "; ".join(errors))

    def _upload_via_fetch(self, b64: str) -> str:
        """Byte-upload through proxify /fetch rendered in the real browser
        (gui_chrome — session cookies + JS). The plain-HTTP POST gets a 400
        from modern Bing; the browser POST carries the same form body but with
        a real session, which Bing accepts."""
        import urllib.parse

        form = urllib.parse.urlencode({"image_url": b64, "q": "", "view": "detailv2", "iss": "sbi"})
        try:
            resp = httpx.post(
                self.proxify_fetch_url + "/fetch",
                json={
                    "url": SEARCH_URL,
                    "method": "POST",
                    "body_b64": base64.b64encode(form.encode("ascii")).decode("ascii"),
                    "headers": {
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://www.bing.com/",
                    },
                    "timeout": 45,
                    "use_browser": True,
                    "force_strategy": "gui_chrome",
                    # Bing renders visual results into .dgControl asynchronously
                    "wait_selector": '.dgControl, #b_results, [class*="dgControl"]',
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

    def _upload_proxy(self, b64: str, proxy: str | None) -> str:
        try:
            resp = httpx.post(
                SEARCH_URL,
                data={"image_url": b64, "q": "", "view": "detailv2", "iss": "sbi"},
                timeout=45,
                follow_redirects=True,
                proxy=proxy,
                headers={
                    "User-Agent": DESKTOP_UA,
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.bing.com/",
                },
            )
            resp.raise_for_status()
            if resp.status_code in (403, 429):
                raise EngineError(f"bing blocked the upload (status={resp.status_code})")
            return resp.text
        except httpx.HTTPError as exc:
            raise EngineError(f"bing-upload failed (proxy={proxy or 'direct'}): {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — any transport failure degrades, never escapes
            raise EngineError(f"bing-upload failed (proxy={proxy or 'direct'}): {exc}") from exc

    def _finish(self, page: str, via: str) -> list[ImageHit]:
        hits = parse_bing_results(page)
        log.info(
            "bing-upload reverse",
            extra={"event": "bing_upload_reverse", "via": via, "hits": len(hits)},
        )
        return hits

    # --------------------------------------------------------------- url
    def search(self, image_url: str) -> list[ImageHit]:
        """URL-based reverse via the ``imgurl:`` trick. ONLY correct when the
        image URL is publicly reachable — on a private box the engines return
        visually-similar junk, so prefer ``search_upload``."""
        import urllib.parse

        target = f"{SEARCH_URL}?q=imgurl:{urllib.parse.quote(image_url, safe='')}&view=detailv2&iss=sbi"
        html = self._fetch(target)
        if not html:
            return []
        hits = parse_bing_results(html)
        log.info(
            "bing reverse",
            extra={"event": "bing_reverse", "image_url": image_url, "hits": len(hits)},
        )
        return hits

    def _fetch(self, target: str) -> str:
        if not self.proxify_fetch_url:
            # degrade instead of a doomed direct fetch that returns junk
            raise EngineError("bing url-reverse requires PROXIFY_FETCH_URL (private-box URL reverse returns junk)")
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


# ------------------------------------------------------------------ parse
def _decode(s: str) -> str:
    try:
        return html_mod.unescape(s)
    except Exception:  # noqa: BLE001
        return s


def parse_bing_results(page: str) -> list[ImageHit]:
    """Parse Bing Visual Search results.

    Modern Bing embeds result JSON HTML-escaped inside attributes:
    ``murl&quot;:&quot;<image>&quot;,...&quot;purl&quot;:&quot;<page>&quot;``.
    We extract (murl, purl) pairs; turl (thumbnail) is optional. Results are
    deduped by image URL.
    """
    hits: list[ImageHit] = []
    seen: set[str] = set()

    def add(murl: str, purl: str, title: str = "") -> None:
        murl = _decode(murl).strip()
        purl = _decode(purl).strip()
        if not murl or murl.startswith("data:") or murl in seen:
            return
        seen.add(murl)
        hits.append(
            ImageHit(
                title=title or purl or murl,
                url=purl or murl,
                img_src=murl,
                engine="bing",
            )
        )

    # primary: murl/purl pairs (HTML-escaped JSON)
    for m in re.finditer(r'murl&quot;:&quot;([^&]{8,}?)&quot;(.{0,600}?)purl&quot;:&quot;([^&]{8,}?)&quot;', page):
        add(m.group(1), m.group(3))
    # titles (t&quot;:&quot;...&quot; near the pair) — best-effort pairing
    for m in re.finditer(r'murl&quot;:&quot;([^&]{8,}?)&quot;.{0,700}?t&quot;:&quot;([^"\\]{4,160})', page):
        murl = _decode(m.group(1))
        title = _decode(m.group(2))
        for hit in hits:
            if hit.img_src == murl and not hit.title.startswith("http"):
                hit.title = title
                break
    # fallback: plain JSON murl/purl
    if not hits:
        for m in re.finditer(r'"murl":"([^"]{8,}?)".{0,400}?"purl":"([^"]{8,}?)"', page):
            add(m.group(1), m.group(2))
    # final fallback: standalone image URLs (deduped, non-bing)
    if not hits:
        for m in re.finditer(r'https://[^\s"\'<>]+?\.(?:jpe?g|png|webp|gif)(?:\?[^\s"\'<>]*)?', page):
            url = m.group(0).rstrip(",.;)")
            if "bing.com" not in url and "microsoft.com" not in url and "akamai" not in url:
                add(url, "")
    return hits
