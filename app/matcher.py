"""The matching engine — how Imify finds "where does this image appear".

Layers, in order:

1. **ANN near-duplicate** — the query pHash (encoded as ±1 floats) is pushed
   through the backend's ``VSEARCH``. For ±1 vectors cosine distance is a
   monotone function of Hamming distance (dist == hamming/32), so this is a
   sub-millisecond Hamming ANN.
2. **Exact rerank** — candidate image records are loaded and the true Hamming
   distance is computed from stored hashes (the f32-rerank step, matching the
   pattern used by production vector DBs).
3. **Exact scan** — for small indexes (or "exact" mode) a full brute-force
   Hamming scan guarantees recall.
4. **Color match** — cosine similarity over 64-dim HSV histograms for the
   "match color" mode.
5. **Fusion + rerank** — local matches and live web results are deduped by
   normalized URL and ranked (SearchOS-style RRF philosophy).
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Iterable
from urllib.parse import urlparse

import httpx
import numpy as np

from app.config import Settings
from app.features import encode_hash_vector, hamming
from app.models import ImageMeta, SearchResult
from app.repo import Repository


class Matcher:
    def __init__(self, repo: Repository, settings: Settings):
        self.repo = repo
        self.settings = settings
        self._histogram_cache: dict[int, np.ndarray] = {}
        self._cache_loaded = False

    # ------------------------------------------------------------------ local
    def _load_histogram_cache(self) -> None:
        if self._cache_loaded:
            return
        self._histogram_cache = {}
        for meta in self.repo.all_images():
            if meta.color:
                self._histogram_cache[meta.id] = np.asarray(meta.color, dtype=np.float32)
        self._cache_loaded = True

    def invalidate_histogram_cache(self) -> None:
        self._cache_loaded = False
        self._histogram_cache = {}

    def search_local(self, query_features: dict, mode: str, limit: int) -> list[SearchResult]:
        # Small index -> exact scan (guaranteed recall, still fast).
        # Large index -> ANN via VSEARCH (sub-ms Hamming search).
        index_size = self.repo.image_count()
        if index_size <= self.settings.exact_scan_max_images:
            return self._exact_scan(query_features, mode, limit)
        return self._ann_search(query_features, mode, limit)

    def _ann_search(self, query_features: dict, mode: str, limit: int) -> list[SearchResult]:
        phash = int(query_features.get("phash", 0))
        query_vec = encode_hash_vector(phash)
        hits = self.repo.db.vsearch(self.settings.vsearch_k, query_vec)
        results = self._rerank_hits(hits, query_features, mode)
        return results[:limit]

    def _exact_scan(self, query_features: dict, mode: str, limit: int) -> list[SearchResult]:
        ids = self.repo.all_image_ids()
        if not ids:
            return []
        metas = [m for m in self.repo.get_images(ids) if m is not None]
        scored: list[tuple[float, ImageMeta]] = []
        fused_map: dict[int, tuple[float, int, int, int]] = {}
        for meta in metas:
            fused, dp, dd, da = self._fused_distance(query_features, meta)
            scored.append((fused, meta))
            fused_map[meta.id] = (fused, dp, dd, da)
        scored.sort(key=lambda t: t[0])
        hits = [(meta.id, 0.0) for _, meta in scored[: self.settings.vsearch_k]]
        results = self._rerank_hits(hits, query_features, mode, exact_fused=fused_map)
        return results[:limit]

    def _rerank_hits(
        self,
        hits: list[tuple[int, float]],
        query_features: dict,
        mode: str,
        exact_fused: dict[int, tuple[float, int, int, int]] | None = None,
    ) -> list[SearchResult]:
        ids = [hid for hid, _ in hits]
        metas = self.repo.get_images(ids)
        results: list[SearchResult] = []
        fused_exact = self.settings.fused_exact_threshold
        fused_near = self.settings.fused_near_threshold
        for (image_id, _dist), meta in zip(hits, metas):
            if meta is None:
                continue
            if exact_fused is not None and image_id in exact_fused:
                fused, dp, dd, da = exact_fused[image_id]
            else:
                fused, dp, dd, da = self._fused_distance(query_features, meta)
            if mode in ("best", "exact", "biggest", "newest") and fused > fused_near:
                continue  # only strong near-duplicates for these modes
            if mode == "changed" and not (fused_exact < fused <= 0.5):
                continue  # the "most changed" band: beyond exact, still related
            results.append(
                SearchResult(
                    image_id=meta.id,
                    url=meta.url,
                    page_url=meta.page_url,
                    title=meta.title or meta.url,
                    engine=meta.engine,
                    source="local",
                    hamming=dp,
                    match_type=self._match_type(fused),
                    score=round(1.0 - fused, 4),
                    width=meta.width,
                    height=meta.height,
                    thumb=f"/img/{meta.id}.{meta.thumb_ext}",
                    first_seen=meta.first_seen,
                    last_seen=meta.last_seen,
                    source_count=meta.source_count,
                    source_urls=list(meta.source_urls or []),
                )
            )
        # post-rerank: boost exact matches, bigger images, fresher results
        for r in results:
            bonus = 0.0
            if r.match_type == "exact":
                bonus += 0.3
            if r.width and r.height:
                bonus += min(0.15, np.log10(r.width * r.height) / 40)
            r.score += bonus
        results.sort(key=lambda r: (-r.score, r.hamming))
        return self._sort_mode(results, mode)

    def _match_type(self, fused: float) -> str:
        if fused <= self.settings.fused_exact_threshold:
            return "exact"
        if fused <= self.settings.fused_near_threshold:
            return "near"
        return "changed"

    @staticmethod
    def _fused_distance(query_features: dict, meta: ImageMeta) -> tuple[float, int, int, int]:
        """Weighted fusion of pHash/dHash/aHash (0-1). pHash is the primary
        perceptual signal; gradient + average hashes add precision so crops,
        watermarks and re-encodes of the SAME image stay close while unrelated
        images separate. Falls back to pHash-only for legacy rows.

        Returns (fused, phash_hamming, dhash_hamming, ahash_hamming).
        """
        return Matcher._fused_from_hashes(query_features, meta.phash, meta.dhash, meta.ahash)

    @staticmethod
    def _fused_from_hashes(query_features: dict, phash: int, dhash: int, ahash: int) -> tuple[float, int, int, int]:
        """Fused distance against raw hashes (used for query vs. candidate)."""
        qp = int(query_features.get("phash") or 0)
        dp = hamming(qp, phash)
        dd = da = -1
        if "dhash" in query_features and "ahash" in query_features:
            dd = hamming(int(query_features["dhash"]), dhash)
            da = hamming(int(query_features["ahash"]), ahash)
        if dd >= 0 and da >= 0:
            fused = 0.5 * (dp / 64.0) + 0.3 * (dd / 64.0) + 0.2 * (da / 64.0)
        else:
            fused = dp / 64.0
        return round(fused, 4), dp, dd, da

    @staticmethod
    def _sort_mode(results: list[SearchResult], mode: str) -> list[SearchResult]:
        if mode == "biggest":
            results.sort(key=lambda r: ((r.width or 0) * (r.height or 0)), reverse=True)
        elif mode == "newest":
            results.sort(key=lambda r: r.last_seen, reverse=True)
        elif mode == "changed":
            results.sort(key=lambda r: -r.hamming)
        elif mode == "color":
            results.sort(key=lambda r: r.score, reverse=True)
        return results

    # --------------------------------------------------- web verification
    def verify_web_hits(
        self,
        query_features: dict,
        hits: list,
        limit: int = 24,
        budget_seconds: float | None = None,
    ) -> list[SearchResult]:
        """Download every web candidate and judge it with OUR hashes — the
        TinEye model. Engines only find URLs; this method decides which ones
        actually contain the query image.

        Each candidate is fetched (through the proxify tunnel when configured),
        feature-extracted, and its fused pHash/dHash/aHash distance to the
        query is computed. Results:

        - fused <= near threshold  -> ``match_type`` exact/near + real score
          ("this site has your image" — safe to report for removals);
        - fetch failed / not a match -> ``match_type="unverified"``, kept at
          low score so nothing is silently dropped, but never presented as a
          confirmed match.

        Runs concurrently (bounded) with a short per-fetch timeout so a slow
        candidate never stalls the whole search. ``budget_seconds`` hard-caps
        the total verification phase — when the budget is spent the remaining
        candidates are reported ``unverified`` (timeout) so the response stays
        real-time and never blocks on stragglers.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import html as html_mod

        from app.cookies import CookieFile
        from app.features import decode_image, extract_features
        from app.models import ImageHit
        from app.searx import DEFAULT_UA

        candidates = list(hits)[: min(limit, self.settings.web_verify_max_candidates)]
        if not candidates:
            return []
        proxy = self.settings.fetch_proxy_url or None
        timeout = self.settings.web_verify_timeout_seconds
        max_bytes = self.settings.image_max_bytes
        min_dim = self.settings.image_min_dimension
        fused_near = self.settings.fused_near_threshold
        cookie_file = CookieFile(self.settings.cookie_file) if self.settings.cookie_file else None
        proxify_fetch = self.settings.proxify_fetch_url.rstrip("/") if self.settings.proxify_fetch_url else ""

        render_budget = self.settings.web_verify_render_max
        renders_used = 0

        def _photo_urls_from_dom(html: str) -> list[str]:
            """All candidate FB CDN image URLs in a rendered page, HTML-entity
            unescaped (the ``_nc_*``/``oe`` signature params are the auth — a
            raw ``&amp;`` breaks the hash and Facebook answers ``Bad URL hash``).
            Prefers real photo variants (``t1.6435-9``/``t51.2885-15``/``t39.30808``)
            over stickers/previews (``m1/v/t6``, ``t42`` video, ``t39.1997``
            reaction assets)."""
            urls = re.findall(r"https://scontent[^\"'\\ <]+?\.(?:jpe?g|png|webp|gif)(?:\?[^\"'\\ <]*)?", html)
            urls = [html_mod.unescape(u) for u in urls]
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

        def render_photo_page(page_url: str, depth: int) -> tuple[bytes | None, str]:
            """Facebook photo pages are JS-rendered comet shells — the real
            scontent URL only exists in the DOM. Render through proxify's real
            browser (gui_chrome, which carries the imported session cookies),
            extract the signed scontent image URL from the rendered HTML and
            fetch it through the tunnel. (The signed ``_nc_*``/``oe`` query
            params are the auth — no cookie match exists for ``scontent.*.fbcdn``
            hosts, and none is needed.) The render bypasses proxify's cache so
            the URL signatures are FRESH (they expire in minutes — a cached
            page returns ``Bad URL hash`` 403s). Returns (content, fresh_url)."""
            nonlocal renders_used
            if not proxify_fetch:
                return None, ""
            if renders_used >= render_budget:
                return None, ""
            renders_used += 1
            try:
                resp = httpx.post(
                    proxify_fetch + "/fetch",
                    json={
                        "url": page_url,
                        "method": "GET",
                        "force_strategy": "gui_chrome",
                        "timeout": 45,
                        "bypass_cache": True,
                    },
                    timeout=90,
                )
                resp.raise_for_status()
                html = resp.json().get("html") or ""
            except Exception:  # noqa: BLE001 — rendering is best-effort
                return None, ""
            for img_url in _photo_urls_from_dom(html):
                content, _resolved, _hops = fetch_bytes(img_url, referer=page_url, depth=depth + 1)
                if content:
                    return content, img_url
            return None, ""

        def fetch_bytes(
            url: str,
            referer: str | None = None,
            depth: int = 0,
            hops: list[str] | None = None,
        ) -> tuple[bytes | None, str, list[str]]:
            """Fetch raw bytes and return ``(content, resolved_url, hops)``.

            ``resolved_url`` is the URL the bytes ACTUALLY came from — for
            Facebook that is the fresh render-discovered scontent URL, which
            differs from the (possibly expired) engine candidate. ``hops`` is
            the resolution trail shown in the UI (direct / cookies:<host> /
            redirect-stub / og:image / render:gui_chrome / fb-signature-retry).

            Walks the tunnel with the real session cookies for the host, and
            follows client-side redirect stubs (Facebook's ``location.href`` /
            og:image pattern) so hotlink-protected CDN candidates resolve to
            actual pixels. When a signed FB CDN URL answers 403 "Bad URL hash"
            (signatures expire in minutes) the engine's PAGE url is re-walked
            to obtain a fresh signature instead of giving up."""
            hops = hops or []
            if depth > 3:
                return None, url, hops + ["depth-limit"]
            try:
                host = (urlparse(url).hostname or "").lower()
                headers = {"User-Agent": DEFAULT_UA}
                if referer:
                    headers["Referer"] = referer
                if cookie_file and host:
                    cookie = cookie_file.header_for(host)
                    if cookie:
                        headers["Cookie"] = cookie
                        hops = hops + [f"cookies:{host}"]
                hops = hops + ["direct"]
                resp = httpx.get(
                    url,
                    timeout=timeout,
                    headers=headers,
                    follow_redirects=True,
                    proxy=proxy,
                )
            except Exception:  # noqa: BLE001
                return None, url, hops + ["fetch-error"]
            if resp.status_code >= 400 or not resp.content:
                # FB CDN signatures expire in minutes -> "Bad URL hash" 403s.
                # If the engine ALSO gave us the page (photo.php), walk it to
                # mint a fresh signed URL instead of declaring the candidate dead.
                if "fbcdn" in host and referer and referer != url:
                    ref_host = (urlparse(referer).hostname or "").lower()
                    if "facebook.com" in ref_host or "fbsbx.com" in ref_host:
                        return fetch_bytes(
                            referer,
                            referer="https://www.facebook.com/",
                            depth=depth + 1,
                            hops=hops + [f"fb-signature-{resp.status_code}→page"],
                        )
                return None, url, hops + [f"http-{resp.status_code}"]
            ctype = resp.headers.get("content-type", "").lower()
            if ctype.startswith("image/"):
                if len(resp.content) > max_bytes:
                    return None, url, hops + ["too-large"]
                # the URL the bytes actually came from (after HTTP redirects)
                final_url = str(getattr(resp, "url", "") or url)
                return resp.content, final_url, hops
            # HTML: look for a client-side redirect stub or og:image.
            if "html" in ctype or (len(resp.content) < 4096 and resp.content.lstrip().startswith(b"<")):
                text = resp.text
                # Facebook lookaside stubs: <script>location.href = "..."</script>
                # (the href is JSON-escaped: https:\/\/www.facebook.com\/photo.php...)
                m = re.search(r'location\.href\s*=\s*["\']([^"\']+)["\']', text)
                if m:
                    target = m.group(1).replace("\\/", "/")
                    return fetch_bytes(
                        target,
                        referer="https://www.facebook.com/",
                        depth=depth + 1,
                        hops=hops + ["redirect-stub"],
                    )
                m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', text)
                if not m:
                    m = re.search(r'<meta[^>]+content=["\']([^"\']+)["\']\s+property=["\']og:image["\']', text)
                if m:
                    return fetch_bytes(m.group(1), referer=url, depth=depth + 1, hops=hops + ["og:image"])
                # Facebook comet photo pages: JS shell — render via proxify browser.
                if "facebook.com" in host or "fbsbx.com" in host:
                    content, fresh_url = render_photo_page(url, depth)
                    if content:
                        return content, fresh_url or url, hops + ["render:gui_chrome"]
                    return None, url, hops + ["render-failed"]
            return None, url, hops + ["not-an-image"]

        def verify_one(hit: ImageHit) -> SearchResult:
            try:
                content, resolved_url, hops = fetch_bytes(hit.img_src, referer=hit.url or hit.img_src)
                if not content:
                    return self._unverified(hit, note="; ".join(hops) or "unreachable", resolution=hops)
                img = decode_image(content)
                if img is None:
                    return self._unverified(hit, note="payload undecodable", resolution=hops)
                if min(img.size) < min_dim:
                    return self._unverified(hit, note=f"too small ({img.size[0]}x{img.size[1]})", resolution=hops)
                feats = extract_features(img, content)
                fused, dp, _, _ = self._fused_from_hashes(
                    query_features,
                    int(feats["phash"]),
                    int(feats["dhash"]),
                    int(feats["ahash"]),
                )
            except Exception:  # noqa: BLE001 — a bad candidate must never fail the search
                return self._unverified(hit, note="verification error", resolution=hops)
            if fused > fused_near:
                return self._unverified(
                    hit, hamming=dp, fused=fused, note=f"hash mismatch Δ{dp}", resolution=hops
                )
            # resolved_url is the URL the verified pixels actually came from —
            # for FB that is the FRESH render-discovered signed URL (the engine
            # candidate may already be expired -> "Bad URL hash"). The page
            # link stays the stable human reference.
            final_url = resolved_url or hit.img_src
            note = ""
            if "fbcdn.net" in (urlparse(final_url).hostname or ""):
                note = "Facebook signed CDN links expire in minutes — use the page link for the stable source."
            return SearchResult(
                url=final_url,
                page_url=hit.url,
                title=hit.title or final_url,
                engine=hit.engine,
                source="web",
                match_type=self._match_type(fused),
                score=round(1.0 - fused, 4),
                hamming=dp,
                width=feats["width"],
                height=feats["height"],
                last_seen=time.time(),
                first_seen=time.time(),
                resolution=hops,
                note=note,
            )

        deadline = None
        if budget_seconds is not None:
            deadline = time.monotonic() + budget_seconds

        results: list[SearchResult] = []
        pending_hits = list(candidates)
        with ThreadPoolExecutor(max_workers=self.settings.web_verify_concurrency) as pool:
            futures = [pool.submit(verify_one, h) for h in candidates]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception:  # noqa: BLE001
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    break
        # anything still unverified when the budget ran out is reported as a
        # timeout-unverified candidate — never silently dropped, never a match
        if deadline is not None and len(results) < len(candidates):
            done = {r.url for r in results}
            for hit in pending_hits:
                if hit.img_src not in done:
                    results.append(self._unverified(hit, note="verification budget exceeded", resolution=["budget-timeout"]))
        # verified matches first (highest score), unverified last
        results.sort(key=lambda r: (0 if r.match_type != "unverified" else 1, -r.score))
        return results

    @staticmethod
    def _unverified(
        hit,
        hamming: int | None = None,
        fused: float | None = None,
        note: str = "",
        resolution: list[str] | None = None,
    ) -> SearchResult:
        """A candidate that could not be fetched or did not hash-match.
        Score stays LOW regardless of how close it seemed, so an unverified
        row can never look like a strong confirmed match. ``note`` is the
        honest reason shown in the UI (fetch 403 / hash mismatch / ...)."""
        return SearchResult(
            url=hit.img_src,
            page_url=hit.url,
            title=hit.title or hit.img_src,
            engine=hit.engine,
            source="web",
            match_type="unverified",
            score=round(max(0.0, 1.0 - (fused or 1.0)) * 0.2, 4) if hamming is not None else 0.05,
            hamming=hamming,
            last_seen=time.time(),
            first_seen=time.time(),
            resolution=resolution or [],
            note=note,
        )

    def search_color(self, query_features: dict, limit: int) -> list[SearchResult]:
        """Exact cosine over the histogram cache (color match mode)."""
        self._load_histogram_cache()
        q = np.asarray(query_features.get("color") or [], dtype=np.float32)
        if q.size == 0 or not self._histogram_cache:
            return []
        ids = np.asarray(list(self._histogram_cache.keys()), dtype=np.int64)
        mat = np.stack(list(self._histogram_cache.values()))
        sims = mat @ q
        dists = np.clip(1.0 - sims, 0.0, 2.0)
        order = np.argsort(dists)[:limit]
        out: list[SearchResult] = []
        for idx in order:
            meta = self.repo.get_image(int(ids[idx]))
            if not meta:
                continue
            out.append(
                SearchResult(
                    image_id=meta.id,
                    url=meta.url,
                    page_url=meta.page_url,
                    title=meta.title or meta.url,
                    engine=meta.engine,
                    source="local",
                    score=max(0.0, 1.0 - float(dists[idx])),
                    match_type="color",
                    width=meta.width,
                    height=meta.height,
                    thumb=f"/img/{meta.id}.{meta.thumb_ext}",
                    first_seen=meta.first_seen,
                    last_seen=meta.last_seen,
                    source_count=meta.source_count,
                )
            )
        out.sort(key=lambda r: -r.score)
        return out[:limit]


# ------------------------------------------------------------------ fusion
def normalize_url(url: str) -> str:
    """Normalize a URL for dedup: strip scheme/www, fragments, tracking params."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{host}{path}".lower()


def fuse_results(local: list[SearchResult], web: list[SearchResult], k: int = 60) -> list[SearchResult]:
    """Merge local + web results, dedupe by normalized URL, prefer indexed copies.

    RRF-style: each list contributes rank-based points; the indexed copy of a
    URL is preferred over the web snippet (same philosophy as SearchOS's
    "indexed copies preferred over web snippets").
    """
    buckets: dict[str, list[SearchResult]] = defaultdict(list)
    for i, r in enumerate(local + web):
        buckets[normalize_url(r.url)].append(r)

    local_rank = {id(r): i for i, r in enumerate(local)}
    web_rank = {id(r): i for i, r in enumerate(web)}
    fused: list[SearchResult] = []
    for group in buckets.values():
        best = min(group, key=lambda r: 0 if r.source == "local" else 1)
        rrf = 0.0
        for r in group:
            if id(r) in local_rank:
                rank = local_rank[id(r)]
            elif id(r) in web_rank:
                rank = web_rank[id(r)]
            else:
                rank = k
            rrf += 1.0 / (k + rank + 1)
        merged = SearchResult(
            image_id=best.image_id,
            url=best.url,
            page_url=best.page_url or best.url,
            title=best.title,
            engine=best.engine,
            source="both" if any(r.source == "local" for r in group) and any(r.source == "web" for r in group) else best.source,
            score=best.score + rrf * 0.5,
            hamming=best.hamming,
            # verified web results carry their real match type (exact/near) —
            # the old blind "web" label no longer exists anywhere
            match_type=best.match_type,
            width=best.width,
            height=best.height,
            thumb=best.thumb,
            first_seen=best.first_seen,
            last_seen=best.last_seen,
            source_count=best.source_count,
            resolution=best.resolution,
            note=best.note,
        )
        fused.append(merged)

    fused.sort(key=lambda r: (-r.score, r.hamming if r.hamming is not None else 99))
    return fused
