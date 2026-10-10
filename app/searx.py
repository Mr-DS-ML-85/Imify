"""SearXNG client — the free, self-hosted discovery layer.

Two capabilities:

1. ``search_images`` — keyword discovery in the ``images`` category, used by
   the continuous indexer to fill the own index.
2. ``reverse_search`` — live web reverse image search via the ``image_url``
   parameter (engines such as Bing Images accept it). Results are merged into
   search responses AND saved into the own index so the system learns.

All calls go through the JSON API (``format=json``). Politeness: a minimum
interval between queries and a circuit breaker so a throttled SearXNG never
wedges the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from app.models import ImageHit

DEFAULT_UA = "ImifyBot/0.1 (+self-hosted image search; polite crawler)"


@dataclass
class SearxError(RuntimeError):
    message: str


class SearxClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 12.0,
        min_interval: float = 5.0,
        max_pages: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_pages = max_pages
        self._last_request = 0.0
        self.failures = 0  # consecutive failures (circuit breaker)
        self.total_requests = 0

    @property
    def available(self) -> bool:
        return self.failures < 5

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def _get_json(self, params: dict) -> dict | None:
        if not self.available:
            raise SearxError("circuit open: too many consecutive SearXNG failures")
        self._throttle()
        try:
            resp = httpx.get(
                f"{self.base_url}/search",
                params={**params, "format": "json"},
                timeout=self.timeout,
                headers={"User-Agent": DEFAULT_UA},
                follow_redirects=True,
            )
            self.total_requests += 1
            if resp.status_code == 429:
                self.failures += 1
                raise SearxError("SearXNG rate-limited (429)")
            resp.raise_for_status()
            self.failures = 0
            return resp.json()
        except httpx.HTTPError as exc:
            self.failures += 1
            raise SearxError(f"SearXNG unreachable: {exc}") from exc

    @staticmethod
    def _parse_results(data: dict | None) -> list[ImageHit]:
        hits: list[ImageHit] = []
        if not data:
            return hits
        for r in data.get("results", []):
            img_src = r.get("img_src") or ""
            if not img_src or img_src.startswith("data:"):
                continue
            hits.append(
                ImageHit(
                    title=(r.get("title") or "").strip(),
                    url=(r.get("url") or "").strip(),
                    img_src=img_src,
                    thumbnail_src=r.get("thumbnail_src") or "",
                    engine=r.get("engine") or "",
                    width=int(r.get("width") or 0),
                    height=int(r.get("height") or 0),
                    img_format=r.get("img_format") or "",
                )
            )
        return hits

    def search_images(self, query: str, pageno: int = 1, safesearch: int = 0) -> list[ImageHit]:
        data = self._get_json(
            {"q": query, "categories": "images", "pageno": pageno, "safesearch": safesearch}
        )
        return self._parse_results(data)

    def reverse_search(self, image_url: str, safesearch: int = 0) -> list[ImageHit]:
        """Live web reverse image search for a publicly reachable image URL.

        Only engines that honor ``image_url`` answer. When SearXNG rejects
        the request (some builds refuse an empty ``q``) this DEGRADES — it
        must never fall back to a text query: the old fallback searched the
        literal words "reverse image search" and fed pages ABOUT reverse
        image search tools into every result set as candidates.
        """
        data = self._get_json({"q": "", "image_url": image_url, "categories": "images", "safesearch": safesearch})
        return self._parse_results(data)

    def discover(self, query: str, safesearch: int = 0) -> list[ImageHit]:
        """Paginated discovery across the configured max pages."""
        hits: list[ImageHit] = []
        seen: set[str] = set()
        for page in range(1, self.max_pages + 1):
            try:
                page_hits = self.search_images(query, pageno=page, safesearch=safesearch)
            except SearxError:
                break
            for hit in page_hits:
                if hit.img_src not in seen:
                    seen.add(hit.img_src)
                    hits.append(hit)
            if not page_hits:
                break
        return hits
