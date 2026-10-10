"""Thin client for proxify's ``/fetch`` — raw bytes for images, PDFs, files.

proxify returns ``body_b64`` + ``content_type`` for binary responses (and
for any request with ``raw: true``). Imify uses it as the fallback
downloader when a plain fetch is refused: hotlink-protected CDNs (403),
rate limits (429) and anti-bot fronts that only let a real browser
session through.
"""

from __future__ import annotations

import base64

import httpx

from app.logging import child_logger

log = child_logger("imify.proxify")


def fetch_raw(
    proxify_fetch_url: str,
    url: str,
    referer: str = "",
    timeout: float = 40.0,
    browser: bool = False,
) -> tuple[bytes | None, str, str]:
    """Fetch ``url`` through proxify and return ``(bytes, content_type,
    final_url)``; ``(None, "", url)`` when proxify is unset or it fails.

    ``browser=True`` forces the real GUI browser (slow; for pages that need
    JavaScript or a trusted session). By default proxify picks the cheapest
    strategy that works (TLS-impersonating HTTP first).
    """
    base = (proxify_fetch_url or "").rstrip("/")
    if not base or not url.startswith(("http://", "https://")):
        return None, "", url
    payload: dict = {"url": url, "method": "GET", "raw": True, "timeout": timeout}
    if referer:
        payload["headers"] = {"Referer": referer}
    if browser:
        payload["use_browser"] = True
    try:
        resp = httpx.post(base + "/fetch", json=payload, timeout=timeout + 30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — fallback path: never raise
        log.debug("proxify raw fetch failed", extra={"url": url, "error": str(exc)})
        return None, "", url
    if not data.get("success") or not data.get("body_b64"):
        return None, str(data.get("content_type") or ""), url
    try:
        content = base64.b64decode(data["body_b64"])
    except Exception:  # noqa: BLE001
        return None, "", url
    return content, str(data.get("content_type") or ""), str(data.get("final_url") or url)
