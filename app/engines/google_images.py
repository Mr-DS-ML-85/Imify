"""Google Images reverse image search (``searchbyimage?image_url=``).

Google's legacy reverse-search endpoint accepts a publicly reachable image URL
via GET and returns a results page. Plain httpx requests get ``sorry``-blocked
at the TLS level on datacenter IPs — so this provider defaults to **curl_cffi
chrome146 impersonation** (a cloned real-Chrome ClientHello), optionally routed
through your proxify anti-bot gateway (CONNECT tunnel or REST /fetch). When
nothing is configured the provider degrades instead of fetching directly.
"""

from __future__ import annotations

import base64
import html as html_mod
import re
import urllib.parse
from typing import Any

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


def _impersonating_session(proxy: str | None) -> Any:
    """Session that mimics a real Chrome 146 TLS fingerprint (curl_cffi).

    Google's anti-bot blocks datacenter IPs on the *TLS* level — httpx's
    OpenSSL ClientHello is a dead giveaway. curl_cffi clones the real Chrome
    146 handshake, which is the difference between a SERP and a /sorry page.
    Falls back to plain httpx if curl_cffi isn't installed (rare).
    """
    try:
        from curl_cffi import requests as cf
    except Exception:  # noqa: BLE001
        return None
    try:
        return cf.Session(
            impersonate="chrome146",
            proxies={"https": proxy, "http": proxy} if proxy else None,
            timeout=40,
        )
    except Exception:  # noqa: BLE001
        return None


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
        return bool(self.proxify_fetch_url or self.fetch_proxy_url or self.direct_upload)

    def search_upload(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        content_type: str = "image/jpeg",
        direct_upload: bool | None = None,
    ) -> list[ImageHit]:
        """Circle-to-Search: reverse-search by uploading the raw bytes to Google Lens.

        The one path that works for images with NO publicly reachable URL.
        Routing, in order:

        1. curl_cffi chrome146 impersonation — via the proxify CONNECT tunnel
           when configured, else direct. This is the path that beats Google's
           TLS-level datacenter block (a real chrome146 ClientHello is the
           difference between a SERP and a /sorry page);
        2. proxify /fetch (base64 body) — full strategy pipeline: pinned
           Chrome persona, your real session cookies, browser render;
        3. direct httpx — only when impersonation is unavailable.

        Returns parsed visual-similar results from the Lens /search?udm=26 page.
        """
        body, boundary = _build_multipart(image_bytes, filename, content_type)
        errors: list[str] = []
        direct = self.direct_upload if direct_upload is None else direct_upload
        if self.fetch_proxy_url or (direct and _impersonating_session(None) is not None):
            try:
                page = self._upload_via_tunnel(body, boundary, proxy=self.fetch_proxy_url or None)
                return self._finish_upload(page, len(image_bytes), via="impersonated-tunnel" if self.fetch_proxy_url else "impersonated-direct")
            except EngineError as exc:
                errors.append(f"impersonation: {exc}")
        if self.proxify_fetch_url:
            try:
                page = self._upload_via_fetch(body, boundary)
                return self._finish_upload(page, len(image_bytes), via="proxify")
            except EngineError as exc:
                errors.append(f"proxify: {exc}")
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
        sess = _impersonating_session(proxy)
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": DESKTOP_UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
        try:
            if sess is not None:
                resp = sess.post(UPLOAD_URL, data=body, headers=headers, allow_redirects=True)
            else:
                resp = httpx.post(
                    UPLOAD_URL,
                    content=body,
                    timeout=60,
                    follow_redirects=True,
                    proxy=proxy,
                    headers=headers,
                )
            status = int(getattr(resp, "status_code", 0) or 0)
            final_url = str(getattr(resp, "url", ""))
            if status in (403, 429) or "sorry" in final_url:
                raise EngineError(f"google blocked the upload (status={status})")
            return str(getattr(resp, "text", "") or "")
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001
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
        sess = _impersonating_session(None)
        if self.proxify_fetch_url:
            payload: dict = {"url": target, "timeout": 45, "use_browser": self.use_browser}
            # gui_chrome is the strategy that beats Google; /fetch falls back to
            # curl_cffi_plus when the GUI browser isn't actually up, so forcing
            # it explicitly (vs bare use_browser=true -> "No strategies
            # available") always yields a live strategy.
            if self.use_browser:
                payload["force_strategy"] = "gui_chrome"
            try:
                resp = httpx.post(
                    self.proxify_fetch_url + "/fetch",
                    json=payload,
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
        if sess is not None:
            try:
                resp = sess.get(target, params={"hl": "en"})
                status = int(getattr(resp, "status_code", 0) or 0)
                if status in (403, 429) or "sorry" in str(getattr(resp, "url", "")):
                    raise EngineError(f"google blocked the fetch (status={status})")
                return str(getattr(resp, "text", "") or "")
            except EngineError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise EngineError(f"google fetch failed: {exc}") from exc
        raise EngineError(
            "google provider requires curl_cffi or PROXIFY_FETCH_URL "
            "(direct fetches get blocked by Google)"
        )


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
