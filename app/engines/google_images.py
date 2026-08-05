"""Google Images reverse image search (``searchbyimage?image_url=``) — proxify-only.

Google's legacy reverse-search endpoint accepts a publicly reachable image URL
via GET and returns a results page. Plain requests are almost always answered
with Google's ``sorry`` block — so this provider routes **exclusively** through
your proxify anti-bot gateway (real browser session cookies + pinned Chrome 146
persona, optionally forcing the GUI-Chrome strategy). When no proxify endpoint
is configured the provider degrades instead of fetching directly — a direct
fetch would just get blocked and burn a request.
"""

from __future__ import annotations

import base64
import html as html_mod
import re
import urllib.parse

import httpx

from app.engines import EngineError
from app.logging import child_logger
from app.models import ImageHit

log = child_logger("imify.google", engine="google_images")

SEARCH_URL = "https://www.google.com/searchbyimage"
# Google retired the legacy /searchbyimage/upload endpoint (404 since 2026);
# the live byte-upload path is Google Lens: POST multipart -> 303 redirect to
# a /search?udm=26 visual-results page carrying the session token.
UPLOAD_URL = "https://lens.google.com/upload"
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)


def _build_multipart(image_bytes: bytes, filename: str, content_type: str) -> tuple[bytes, str]:
    """Build a Google-compatible multipart body with the image as encoded_image."""
    boundary = f"----imify{id(image_bytes):x}"
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="encoded_image"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        image_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), boundary


class GoogleImagesReverse:
    def __init__(
        self,
        proxify_fetch_url: str = "",
        fetch_proxy_url: str = "",
        use_browser: bool = True,
        direct_upload: bool = True,
    ):
        # REST /fetch endpoint (for URL-based reverse) — UTF-8 text only.
        self.proxify_fetch_url = proxify_fetch_url.rstrip("/")
        # CONNECT tunnel (for byte-upload reverse) — raw binary passes through.
        self.fetch_proxy_url = fetch_proxy_url
        self.use_browser = use_browser
        # last-resort direct upload (works from residential IPs only).
        self.direct_upload = direct_upload

    @property
    def available(self) -> bool:
        """Google is only reachable through proxify — never direct."""
        return bool(self.proxify_fetch_url)

    def search_upload(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        content_type: str = "image/jpeg",
        direct_upload: bool = True,
    ) -> list[ImageHit]:
        """Circle-to-Search: reverse-search by uploading the raw bytes to Google Lens.

        The one path that works for images with NO publicly reachable URL.
        Routing, in order:

        1. proxify /fetch (base64 body) — full strategy pipeline: pinned Chrome
           persona, your real session cookies, TLS impersonation;
        2. proxify CONNECT tunnel — raw bytes;
        3. direct — only when no proxify is configured (works from residential
           IPs; Google sorry-blocks datacenter IPs — degraded honestly).

        Returns parsed visual-similar results from the Lens /search?udm=26 page.
        """
        body, boundary = _build_multipart(image_bytes, filename, content_type)
        errors: list[str] = []
        if self.proxify_fetch_url:
            try:
                page = self._upload_via_fetch(body, boundary)
                return self._finish_upload(page, len(image_bytes), via="proxify")
            except EngineError as exc:
                errors.append(f"proxify: {exc}")
        if self.fetch_proxy_url:
            try:
                page = self._upload_via_tunnel(body, boundary, proxy=self.fetch_proxy_url)
                return self._finish_upload(page, len(image_bytes), via="tunnel")
            except EngineError as exc:
                errors.append(f"tunnel: {exc}")
        if direct_upload:
            try:
                page = self._upload_via_tunnel(body, boundary, proxy=None)
                return self._finish_upload(page, len(image_bytes), via="direct")
            except EngineError as exc:
                errors.append(f"direct: {exc}")
        raise EngineError("google-upload failed via all paths: " + "; ".join(errors))

    def _finish_upload(self, page: str, size: int, via: str) -> list[ImageHit]:
        hits = parse_google_results(page)
        log.info(
            "google-upload reverse",
            extra={"event": "google_upload_reverse", "via": via, "bytes": size, "hits": len(hits)},
        )
        return hits

    def _upload_via_fetch(self, body: bytes, boundary: str) -> str:
        """Binary upload through proxify /fetch (cookies + TLS impersonation)."""
        try:
            resp = httpx.post(
                self.proxify_fetch_url + "/fetch",
                json={
                    "url": UPLOAD_URL,
                    "method": "POST",
                    "body_b64": base64.b64encode(body).decode("ascii"),
                    "headers": {
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://www.google.com/",
                    },
                    # Google Lens is fully JS-gated — the upload POST returns a
                    # shell and the visual-similar results only exist after JS
                    # renders them. Force the real browser (gui_chrome, session
                    # cookies) and wait for the results grid so the returned
                    # page actually carries candidates.
                    "timeout": 45,
                    "use_browser": True,
                    "force_strategy": "gui_chrome",
                    "wait_selector": '.islrc, [class*="islrc"], #yDmH0d img, a[href*="/imgres"]',
                    "bypass_cache": True,
                },
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"proxify /fetch unreachable: {exc}") from exc
        except ValueError as exc:
            raise EngineError(f"proxify returned a non-JSON response: {exc}") from exc
        if data.get("success") is False:
            raise EngineError(f"proxify upload failed: {data.get('error', data.get('status_code', 'unknown'))}")
        return data.get("html") or data.get("markdown") or ""

    def _upload_via_tunnel(self, body: bytes, boundary: str, proxy: str | None) -> str:
        """Raw binary upload through the proxify CONNECT tunnel (or direct)."""
        try:
            resp = httpx.post(
                UPLOAD_URL,
                content=body,
                timeout=60,
                follow_redirects=True,
                proxy=proxy,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": DESKTOP_UA,
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.google.com/",
                },
            )
            resp.raise_for_status()
            if resp.status_code in (403, 429) or (getattr(resp, "url", None) and "sorry" in str(resp.url)):
                raise EngineError(f"google blocked the upload (status={resp.status_code})")
            return resp.text
        except httpx.HTTPError as exc:
            raise EngineError(f"google-upload failed (proxy={proxy or 'direct'}): {exc}") from exc

    def search(self, image_url: str) -> list[ImageHit]:
        params = urllib.parse.urlencode({"image_url": image_url, "hl": "en"})
        target = f"{SEARCH_URL}?{params}"
        html = self._fetch(target)
        if not html:
            return []
        hits = parse_google_results(html)
        log.info(
            "google reverse",
            extra={"event": "google_reverse", "image_url": image_url, "hits": len(hits)},
        )
        return hits

    def _fetch(self, target: str) -> str:
        if not self.proxify_fetch_url:
            # The user's rule: never hit Google directly — it blocks datacenter
            # IPs. Degrade (reported as provider:google) instead of a doomed request.
            raise EngineError(
                "google provider requires PROXIFY_FETCH_URL (direct fetches get blocked by Google)"
            )
        try:
            resp = httpx.post(
                self.proxify_fetch_url + "/fetch",
                json={"url": target, "timeout": 45, "use_browser": self.use_browser},
                timeout=70,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise EngineError(f"proxify unreachable: {exc}") from exc
        try:
            data = resp.json()
        except ValueError as exc:
            # proxify answered 200 with a non-JSON body (error page) — degrade.
            raise EngineError(f"proxify returned a non-JSON response: {exc}") from exc
        if data.get("success") is False:
            raise EngineError(f"proxify fetch failed: {data.get('error', data.get('status_code', 'unknown'))}")
        return data.get("html") or data.get("markdown") or ""


def _decode(s: str) -> str:
    try:
        return html_mod.unescape(urllib.parse.unquote(s))
    except Exception:  # noqa: BLE001
        return s


def parse_google_results(page: str) -> list[ImageHit]:
    """Tolerant parser for the Google Images reverse-search page.

    Modern Google embeds results as JSON (``\"ou\":\"original\",\"ru\":\"page\"``);
    older variants use ``imgres?imgurl=..&imgrefurl=..`` pairs. Both are
    handled, deduped by original URL. Google's OWN assets (the logo, gstatic
    placeholders, the captcha/blocked page) are NOT results — they are filtered
    hard so a blocked or logo-only page can never be parsed into candidates.
    """
    hits: list[ImageHit] = []
    seen: set[str] = set()

    def add(img: str, page_url: str, title: str, engine: str = "google images") -> None:
        if not img or img.startswith("data:"):
            return
        if img in seen:
            return
        low = img.lower()
        # Google's own branding/placeholder assets — never results. A blocked
        # page (captcha / logo shell) otherwise leaks these as fake candidates
        # that can never hash-match and pollute the chain with junk.
        if any(bad in low for bad in ("gstatic.com", "googlelogo", "google.com/images", ".google.com/", "googleusercontent.com/images")):
            return
        seen.add(img)
        hits.append(
            ImageHit(
                title=title or img,
                url=page_url or img,
                img_src=img,
                engine=engine,
            )
        )

    # 1) embedded JSON fields: "ou" original, "ru" referrer, "pt" title
    for m in re.finditer(r'"ou"\s*:\s*"([^"]+)".*?"ru"\s*:\s*"([^"]*)"', page, re.DOTALL):
        img = _decode(m.group(1))
        ref = _decode(m.group(2))
        add(img, ref, "")
    for m in re.finditer(r'"ou"\s*:\s*"([^"]+)"', page):
        img = _decode(m.group(1))
        if img not in seen:
            add(img, "", "")

    # 2) classic imgres pairs
    for m in re.finditer(r"imgres\?imgurl=([^&\"' ]+)(?:&[^\"']*?)?imgrefurl=([^&\"' ]+)", page):
        add(_decode(m.group(1)), _decode(m.group(2)), "")

    # 3) standalone image URLs from the result grid (fallback)
    for m in re.finditer(r'https://[^\s"\'<>]+?\.(?:jpe?g|png|webp|gif|avif)(?:\?[^\s"\'<>]*)?', page):
        url = m.group(0).rstrip(",.;)")
        add(url, "", "")

    # 4) titles from JSON "pt" fields (best-effort pairing)
    titles = re.findall(r'"pt"\s*:\s*"([^"]+)"', page)
    for i, hit in enumerate(hits):
        if i < len(titles) and titles[i]:
            hit.title = _decode(titles[i])[:160]

    return hits
