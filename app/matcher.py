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

import hashlib
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import threading

import httpx
import numpy as np

from app.config import Settings
from app.features import encode_hash_vector, hamming

_WEB_PREVIEW_SUBDIR = "web-preview"


_preview_writes = 0


def _prune_previews(d: Path, max_files: int) -> None:
    """Keep the preview cache bounded: drop the oldest files past the cap."""
    try:
        files = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    excess = len(files) - max_files
    for p in files[: max(0, excess)]:
        try:
            p.unlink()
        except OSError:
            pass


def _cache_web_preview(content: bytes, settings: Settings) -> str | None:
    """Persist a verified web candidate's pixels locally so the UI loads them
    from our static mount instead of the raw (often hotlink-protected /
    rate-limited) Yandex/Bing CDN URL. Returns the local /uploads/... path."""
    try:
        d = Path(settings.uploads_dir) / _WEB_PREVIEW_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha1(content).hexdigest() + ".jpg"
        p = d / name
        if not p.exists():
            p.write_bytes(content)
            global _preview_writes
            _preview_writes += 1
            if _preview_writes % 500 == 0:
                _prune_previews(d, settings.web_preview_max_files)
        return f"/uploads/{_WEB_PREVIEW_SUBDIR}/{name}"
    except Exception:  # noqa: BLE001 — preview caching is best-effort
        return ""
from app.models import MATCH_FAMILY_RANK, ImageMeta, SearchResult
from app.proxify import fetch_raw
from app.repo import Repository


class Matcher:
    def __init__(self, repo: Repository, settings: Settings):
        self.repo = repo
        self.settings = settings
        self.clip = None  # optional Clipper, injected late (avoids import cycle)
        self.face = None  # optional FaceId, injected late
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

    def search_local(
        self,
        query_features: dict,
        mode: str,
        limit: int,
        query_clip: list[float] | None = None,
    ) -> list[SearchResult]:
        # Small index -> exact scan (guaranteed recall, still fast).
        # Large index -> ANN via VSEARCH (sub-ms Hamming search).
        index_size = self.repo.image_count()
        if index_size <= self.settings.exact_scan_max_images:
            results = self._exact_scan(query_features, mode, limit)
        else:
            results = self._ann_search(query_features, mode, limit)
        return self._semantic_gate(results, query_clip, mode)

    def _semantic_gate(self, results: list[SearchResult], query_clip: list[float] | None, mode: str) -> list[SearchResult]:
        """Drop hash matches whose CONTENT disagrees (CLIP below the verify
        floor): flat logos, dark frames and textures can hash close to
        anything. Same rule as web verification. Skipped for the deliberate
        "changed"/"color" modes and for rows without a stored embedding."""
        floor = self.settings.clip_verify_min_sim
        if not query_clip or floor <= 0 or mode in ("changed", "color") or self.clip is None:
            return results
        kept: list[SearchResult] = []
        for r in results:
            if r.clip_embedding:
                sim = self.clip.match(query_clip, r.clip_embedding)
                if sim < floor:
                    continue
                r.clip_sim = round(sim, 4)
                r.ai_modalities = sorted(set(r.ai_modalities or []) | {"hash", "clip"})
            kept.append(r)
        return kept

    def search_local_geometric(
        self,
        query_img,
        query_clip: list[float] | None,
        exclude: set[int],
        limit: int = 20,
    ) -> list[SearchResult]:
        """TinEye-style crops/edits inside the OWN index: hashes miss a
        cropped or bordered copy, keypoints don't. Candidates are the top
        ``limit`` entries by CLIP similarity above the geometry prefilter
        (an unrelated picture cannot be a crop), checked with SIFT + RANSAC
        against the best local pixels (original upload, else thumbnail)."""
        from app import geometry
        from app.indexer import local_pixels

        st = self.settings
        if not st.geometric_verify_enabled or query_img is None or not query_clip or self.clip is None:
            return []
        scored = []
        for meta in self.repo.all_images():
            if meta.id in exclude or not meta.clip_embedding:
                continue
            sim = self.clip.match(query_clip, meta.clip_embedding)
            if sim >= st.geometric_clip_prefilter:
                scored.append((sim, meta))
        if not scored:
            return []
        scored.sort(key=lambda t: -t[0])
        q_kp = geometry.Keypoints.from_image(query_img)
        q_kp_mirror = geometry.Keypoints.from_image(geometry.mirror(query_img))
        out: list[SearchResult] = []
        for sim, meta in scored[:limit]:
            img = local_pixels(st, meta)
            if img is None:
                continue
            gm = geometry.verify(q_kp, q_kp_mirror, img)
            if gm is None:
                continue
            strong = gm.inliers >= st.geometric_strong_inliers and gm.ratio >= 0.3
            medium = (
                gm.inliers >= st.geometric_min_inliers
                and gm.ratio >= st.geometric_min_inlier_ratio
                and sim >= st.clip_verify_min_sim
            )
            if not (strong or medium):
                continue
            strength = min(1.0, gm.inliers / 150.0)
            mods = ["geometry", "clip"] + (["mirrored"] if gm.mirrored else [])
            out.append(SearchResult(
                image_id=meta.id,
                url=meta.url,
                page_url=meta.page_url,
                title=meta.title or meta.url,
                engine=meta.engine,
                source="local",
                match_type="modified",
                score=round(0.6 + 0.25 * strength, 4),
                confidence=int(round(70 + 27 * strength)),
                category="duplicate",
                width=meta.width,
                height=meta.height,
                thumb=f"/img/{meta.id}.{meta.thumb_ext}",
                first_seen=meta.first_seen,
                last_seen=meta.last_seen,
                source_count=meta.source_count,
                source_urls=list(meta.source_urls or []),
                caption=meta.caption or "",
                content_phash=meta.phash,
                clip_sim=round(sim, 4),
                geo_inliers=gm.inliers,
                ai_modalities=mods,
                note="cropped / edited copy of your image" + ("; mirrored" if gm.mirrored else ""),
                clip_embedding=list(meta.clip_embedding or []),
            ))
        return out

    def _ann_search(self, query_features: dict, mode: str, limit: int) -> list[SearchResult]:
        phashes = [int(query_features.get("phash", 0))]
        phashes += [int(v["phash"]) for v in query_features.get("variants") or ()]
        hits: list[tuple[int, float]] = []
        seen: set[int] = set()
        for phash in phashes:  # rotated/mirrored copies live elsewhere in Hamming space
            for hid, dist in self.repo.db.vsearch(self.settings.vsearch_k, encode_hash_vector(phash)):
                if hid not in seen:
                    seen.add(hid)
                    hits.append((hid, dist))
        results = self._rerank_hits(hits, query_features, mode)
        return results[:limit]

    def _exact_scan(self, query_features: dict, mode: str, limit: int) -> list[SearchResult]:
        ids = self.repo.all_image_ids()
        if not ids:
            return []
        metas = [m for m in self.repo.get_images(ids) if m is not None]
        scored: list[tuple[float, ImageMeta]] = []
        fused_map: dict[int, tuple[float, int, int, int, str]] = {}
        for meta in metas:
            fused, dp, dd, da, label = self._best_fused(query_features, meta.phash, meta.dhash, meta.ahash, self.settings.rotation_variant_max_fused)
            scored.append((fused, meta))
            fused_map[meta.id] = (fused, dp, dd, da, label)
        scored.sort(key=lambda t: t[0])
        hits = [(meta.id, 0.0) for _, meta in scored[: self.settings.vsearch_k]]
        results = self._rerank_hits(hits, query_features, mode, exact_fused=fused_map)
        return results[:limit]

    def _rerank_hits(
        self,
        hits: list[tuple[int, float]],
        query_features: dict,
        mode: str,
        exact_fused: dict[int, tuple[float, int, int, int, str]] | None = None,
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
                fused, dp, dd, da, label = exact_fused[image_id]
            else:
                fused, dp, dd, da, label = self._best_fused(query_features, meta.phash, meta.dhash, meta.ahash, self.settings.rotation_variant_max_fused)
            if mode in ("best", "exact", "biggest", "newest") and fused > fused_near:
                continue  # only strong near-duplicates for these modes
            if mode == "changed" and not (fused_exact < fused <= 0.5):
                continue  # the "most changed" band: beyond exact, still related
            mtype = self._match_type(fused)
            conf, cat = self._conf_and_category(fused, mtype)
            results.append(
                SearchResult(
                    image_id=meta.id,
                    url=meta.url,
                    page_url=meta.page_url,
                    title=meta.title or meta.url,
                    engine=meta.engine,
                    source="local",
                    hamming=dp,
                    content_phash=meta.phash,
                    match_type=mtype,
                    score=round(1.0 - fused, 4),
                    width=meta.width,
                    height=meta.height,
                    thumb=f"/img/{meta.id}.{meta.thumb_ext}",
                    confidence=conf,
                    category=cat,
                    first_seen=meta.first_seen,
                    last_seen=meta.last_seen,
                    source_count=meta.source_count,
                    source_urls=list(meta.source_urls or []),
                    caption=meta.caption or "",
                    clip_embedding=list(meta.clip_embedding or []),
                    ai_modalities=["hash"] + ([label.split(" ")[0]] if label else []),
                    note=label,
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
    def _conf_and_category(
        fused: float,
        match_type: str,
    ) -> tuple[int, str]:
        """FaceCheck-style confidence + Reversely-style bucket from fused distance.

        fused 0 (identical) -> confidence 100, category "duplicate".
        fused <= near      -> a real near-duplicate copy ("duplicate").
        fused <= changed   -> a genuinely similar variant/crop ("similar").
        beyond             -> not confirmed cleanly ("similar"/"candidate").
        Percent drops with distance so the grid sorts by how close each is.
        """
        conf = int(round(max(0.0, min(1.0, 1.0 - fused)) * 100))
        if match_type == "exact":
            cat = "duplicate"
        elif match_type == "near":
            cat = "duplicate"
        elif match_type == "changed":
            cat = "similar"
        else:
            cat = "candidate"
        return conf, cat

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
    def _best_fused(
        query_features: dict, phash: int, dhash: int, ahash: int, variant_max: float = 0.20
    ) -> tuple[float, int, int, int, str]:
        """Fused distance against the query in its best orientation
        (identity + the rotation/mirror ``variants`` when present). Returns
        (fused, phash_hamming, dhash_hamming, ahash_hamming, variant_label).

        A rotated/mirrored orientation only counts when it is a tight match
        (<= ``variant_max``): a real rotated copy scores ~0, while trying 8
        orientations gives low-detail images 8x the chances to collide."""
        fused, dp, dd, da = Matcher._fused_from_hashes(query_features, phash, dhash, ahash)
        label = ""
        for v in query_features.get("variants") or ():
            vf, vp, vd, va = Matcher._fused_from_hashes(v, phash, dhash, ahash)
            if vf < fused and vf <= variant_max:
                fused, dp, dd, da, label = vf, vp, vd, va, v.get("label", "rotated")
        return fused, dp, dd, da, label

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
        query_clip: list[float] | None = None,
        query_img=None,
        query_faces: list[list[float]] | None = None,
    ) -> list[SearchResult]:
        """Download every web candidate and judge it ourselves — the TinEye
        model. Engines only find URLs; this method decides which ones
        actually contain the query image (or the query person).

        Verdict ladder per candidate (first that holds wins):

        1. **hash** — fused pHash/dHash/aHash (also against the MIRRORED
           query) within the near threshold, and CLIP agrees it is the same
           content -> ``exact`` / ``near``;
        2. **geometry** — SIFT + RANSAC finds the query inside the candidate
           (crop, border, caption bar, rotation, flip) -> ``modified``;
        3. **face** — the candidate shows the same person as the query
           (ArcFace) -> ``face`` (FaceCheck: a different photo of you);
        4. **clip** — semantically close -> ``similar`` (Google-Lens style
           lead, never counted as found);
        5. otherwise ``unverified`` with the honest reason.

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
        from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

        import html as html_mod

        from app.cookies import CookieFile
        from app.features import decode_image, extract_features
        from app.models import ImageHit
        from app.searx import DEFAULT_UA

        # Bare CDN image copies have NO article page — the engine only found the
        # raw pixels (preview.redd.it/...png, i.imgur.com/...png). Such a URL is
        # not a removable site nor an openable article; clicking "open post"
        # just re-opens the image. Drop them before spending verify budget on a
        # fetch (they can never produce a usable page_url anyway).
        def _is_bare_image_url(u: str) -> bool:
            try:
                path = urlparse(u).path.lower()
            except ValueError:
                path = ""
            return path.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp"))

        def _has_article_page(hit: ImageHit) -> bool:
            page = (hit.url or "").strip()
            if not page:
                return False
            if _is_bare_image_url(page):
                return False
            return True

        candidates = [h for h in hits if _has_article_page(h)][: min(limit, self.settings.web_verify_max_candidates)]
        if not candidates:
            return []
        proxy = self.settings.fetch_proxy_url or None
        timeout = self.settings.web_verify_timeout_seconds
        max_bytes = self.settings.image_max_bytes
        min_dim = self.settings.image_min_dimension
        fused_near = self.settings.fused_near_threshold
        cookie_file = CookieFile(self.settings.cookie_file) if self.settings.cookie_file else None
        proxify_fetch = self.settings.proxify_fetch_url.rstrip("/") if self.settings.proxify_fetch_url else ""

        fb_enabled = self.settings.facebook_resolution_enabled
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

        proxify_raw_left = [self.settings.web_verify_proxify_max]
        proxify_lock = threading.Lock()

        def via_proxify(url: str, referer: str | None) -> tuple[bytes, str] | None:
            """Refused directly -> retry through proxify's raw-bytes mode
            (its TLS-impersonating client / real browser gets past hotlink
            and anti-bot fronts). Bounded per search."""
            if not proxify_fetch:
                return None
            with proxify_lock:
                if proxify_raw_left[0] <= 0:
                    return None
                proxify_raw_left[0] -= 1
            content, ctype, final = fetch_raw(proxify_fetch, url, referer=referer or "", timeout=timeout)
            if content and ctype.lower().startswith("image/") and len(content) <= max_bytes:
                return content, final
            return None

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
                via = via_proxify(url, referer)
                if via is not None:
                    return via[0], via[1], hops + ["fetch-error", "proxify"]
                return None, url, hops + ["fetch-error"]
            if resp.status_code >= 400 or not resp.content:
                # FB CDN signatures expire in minutes -> "Bad URL hash" 403s.
                # If the engine ALSO gave us the page (photo.php), walk it to
                # mint a fresh signed URL instead of declaring the candidate dead.
                if "fbcdn" in host and referer and referer != url and fb_enabled:
                    ref_host = (urlparse(referer).hostname or "").lower()
                    if "facebook.com" in ref_host or "fbsbx.com" in ref_host:
                        return fetch_bytes(
                            referer,
                            referer="https://www.facebook.com/",
                            depth=depth + 1,
                            hops=hops + [f"fb-signature-{resp.status_code}→page"],
                        )
                if resp.status_code in (401, 403, 405, 429, 451, 503):
                    via = via_proxify(url, referer)
                    if via is not None:
                        return via[0], via[1], hops + [f"http-{resp.status_code}", "proxify"]
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
                if m and fb_enabled:
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
                if ("facebook.com" in host or "fbsbx.com" in host) and fb_enabled:
                    content, fresh_url = render_photo_page(url, depth)
                    self.repo.record_provider("facebook-render", ok=bool(content))
                    if content:
                        return content, fresh_url or url, hops + ["render:gui_chrome"]
                    return None, url, hops + ["render-failed"]
            return None, url, hops + ["not-an-image"]

        # ---- query-side profile, computed once for every candidate --------
        from app import geometry

        clip = self.clip if (self.clip is not None and self.clip.available) else None
        face = self.face if (self.face is not None and self.face.available and query_faces) else None
        q_kp = q_kp_mirror = None
        if query_img is not None:
            if not query_features.get("variants"):
                from app.features import hash_variants

                query_features = {**query_features, "variants": hash_variants(query_img)}
            if self.settings.geometric_verify_enabled:
                q_kp = geometry.Keypoints.from_image(query_img)
                q_kp_mirror = geometry.Keypoints.from_image(geometry.mirror(query_img))
        st = self.settings
        clip_floor = st.clip_verify_min_sim

        cache_ttl = st.web_candidate_cache_ttl_seconds
        preview_dir = Path(st.uploads_dir) / _WEB_PREVIEW_SUBDIR

        def cached_fetch(url: str, referer: str | None) -> tuple[bytes | None, str, list[str]]:
            """fetch_bytes behind a per-URL byte cache: a candidate seen in
            the last ``web_candidate_cache_ttl_seconds`` is read from disk,
            never downloaded again (bandwidth guard)."""
            key = "vc:" + hashlib.sha1(url.encode()).hexdigest()
            if cache_ttl > 0:
                try:
                    entry = self.repo.db.get_json(key)
                except Exception:  # noqa: BLE001
                    entry = None
                if entry and time.time() - float(entry.get("ts", 0)) < cache_ttl:
                    path = preview_dir / str(entry.get("file", ""))
                    if path.is_file():
                        return path.read_bytes(), entry.get("resolved") or url, list(entry.get("hops") or []) + ["cache"]
            content, resolved, hops = fetch_bytes(url, referer=referer)
            if content and cache_ttl > 0:
                local = _cache_web_preview(content, self.settings)
                if local:
                    try:
                        self.repo.db.set_json(key, {
                            "ts": time.time(), "file": local.rsplit("/", 1)[-1],
                            "resolved": resolved, "hops": hops,
                        })
                    except Exception:  # noqa: BLE001 — cache is best-effort
                        pass
            return content, resolved, hops

        def triage(hit: ImageHit) -> SearchResult | None:
            """Judge the engine THUMBNAIL first; return an unverified result
            when it is clearly unrelated (full image never downloaded), or
            None to proceed with the full verification."""
            thumb_url = (getattr(hit, "thumbnail_src", "") or "").strip()
            if not st.web_verify_thumbnail_triage or not thumb_url or thumb_url == hit.img_src:
                return None
            if not thumb_url.startswith(("http://", "https://")):
                return None
            tb, _r, thops = cached_fetch(thumb_url, referer=hit.url or None)
            timg = decode_image(tb) if tb else None
            if timg is None:
                return None  # thumbnail unusable -> decide on the full image
            tf = extract_features(timg)
            tfused, tdp, _, _, _ = self._best_fused(
                query_features, int(tf["phash"]), int(tf["dhash"]), int(tf["ahash"]), st.rotation_variant_max_fused
            )
            if tfused <= fused_near + 0.08:
                return None  # could be a copy
            tclip = clip.match(query_clip, clip.embed_image(timg)) if (clip and query_clip) else None
            if tclip is None:
                return None  # no semantic signal -> can't safely reject
            if tclip >= min(st.geometric_clip_prefilter, st.clip_similar_min_sim):
                return None  # could be a crop / similar content
            if face is not None:
                big = timg.resize((timg.width * 2, timg.height * 2)) if max(timg.size) < 400 else timg
                if face.detect_faces(big):
                    return None  # a person — maybe the same one
            return self._unverified(
                hit,
                hamming=tdp,
                fused=tfused,
                note=f"rejected on thumbnail (hash Δ{tdp}, clip {tclip:.2f}) — full image not downloaded",
                resolution=thops + ["thumbnail-triage"],
                thumb=_cache_web_preview(tb, self.settings) or "",
                clip_sim=tclip,
            )

        def verify_one(hit: ImageHit) -> SearchResult:
            hops: list[str] = []
            try:
                early = triage(hit)
                if early is not None:
                    return early
                content, resolved_url, hops = cached_fetch(hit.img_src, referer=hit.url or hit.img_src)
                if not content:
                    return self._unverified(hit, note="; ".join(hops) or "unreachable", resolution=hops)
                img = decode_image(content)
                if img is None:
                    return self._unverified(hit, note="payload undecodable", resolution=hops)
                if min(img.size) < min_dim:
                    return self._unverified(hit, note=f"too small ({img.size[0]}x{img.size[1]})", resolution=hops)
                feats = extract_features(img, content)
                cand_hashes = (int(feats["phash"]), int(feats["dhash"]), int(feats["ahash"]))
                fused, dp, _, _, orient = self._best_fused(query_features, *cand_hashes, self.settings.rotation_variant_max_fused)
                mirrored = bool(orient)
                # CLIP: semantic closeness. Gates hash matches (two different
                # logos can hash close), prefilters geometry, and powers the
                # "visually similar" section. The candidate embedding also
                # rides along so consolidation can merge re-encoded copies.
                cand_clip: list[float] | None = clip.embed_image(img) if clip else None
                clip_sim = clip.match(query_clip, cand_clip) if (clip and query_clip and cand_clip) else None
                # pixels decode fine — keep a LOCAL preview so the thumbnail
                # renders whatever the verdict (raw Yandex/Bing CDN URLs are
                # hotlink-protected / rate-limited).
                preview = _cache_web_preview(content, self.settings)

                mtype = ""
                modalities: list[str] = []
                note = ""
                conf = 0
                cat = ""
                score = 0.0
                geo_inliers: int | None = None
                face_sim: float | None = None
                face_tier = ""
                clip_note = ""

                # 1. hash (+ semantic agreement)
                if fused <= fused_near:
                    if clip_sim is not None and clip_floor > 0 and clip_sim < clip_floor:
                        clip_note = f"semantic mismatch (clip {clip_sim:.2f} < {clip_floor:.2f})"
                    else:
                        mtype = self._match_type(fused)
                        conf, cat = self._conf_and_category(fused, mtype)
                        score = 1.0 - fused
                        modalities = ["hash"]

                # 2. geometry: crop / border / overlay / rotation / flip
                if not mtype and q_kp is not None and (
                    clip_sim is None or clip_sim >= st.geometric_clip_prefilter
                ):
                    gm = geometry.verify(q_kp, q_kp_mirror, img)
                    if gm is not None:
                        strong = gm.inliers >= st.geometric_strong_inliers and gm.ratio >= 0.3
                        medium = (
                            gm.inliers >= st.geometric_min_inliers
                            and gm.ratio >= st.geometric_min_inlier_ratio
                            and clip_sim is not None
                            and clip_sim >= clip_floor
                        )
                        if strong or medium:
                            mtype = "modified"
                            geo_inliers = gm.inliers
                            mirrored = mirrored or gm.mirrored
                            strength = min(1.0, gm.inliers / 150.0)
                            conf = int(round(70 + 27 * strength))
                            cat = "duplicate"
                            score = 0.6 + 0.25 * strength
                            modalities = ["geometry"]
                            note = "cropped / edited copy of your image"

                # 3. face: same person, possibly a completely different photo
                if face is not None:
                    cand_faces = [f["embedding"] for f in face.embed_image(img)]
                    if cand_faces:
                        face_sim = face.best_match(query_faces, cand_faces)
                        if face.is_same_person(face_sim):
                            _fconf, face_tier = face.confidence_tier(face_sim)
                            modalities.append("face")
                            if not mtype:
                                mtype = "face"
                                conf = _fconf
                                cat = "face-" + face_tier
                                score = face_sim
                                note = "same person, different photo"

                # 4. CLIP: visually similar lead (never counted as found)
                if not mtype and not clip_note and clip_sim is not None and clip_sim >= st.clip_similar_min_sim:
                    mtype = "similar"
                    cat = "similar"
                    conf = int(round(clip_sim * 100))
                    score = 0.3 * clip_sim
                    modalities = ["clip"]
                if clip_sim is not None and "clip" not in modalities and mtype:
                    modalities.append("clip")
                if mirrored and mtype in ("exact", "near", "modified"):
                    orient = orient or "mirrored"
                    modalities.append(orient.split(" ")[0])
                    note = (note + "; " if note else "") + orient
            except Exception:  # noqa: BLE001 — a bad candidate must never fail the search
                return self._unverified(hit, note="verification error", resolution=hops)

            if not mtype:
                return self._unverified(
                    hit,
                    hamming=dp,
                    fused=fused,
                    note=clip_note or f"no match (hash Δ{dp}"
                    + (f", clip {clip_sim:.2f}" if clip_sim is not None else "")
                    + (f", face {face_sim:.2f}" if face_sim is not None else "")
                    + ")",
                    resolution=hops,
                    thumb=preview,
                    clip_sim=clip_sim,
                )
            # resolved_url is the URL the verified pixels actually came from —
            # for FB that is the FRESH render-discovered signed URL (the engine
            # candidate may already be expired -> "Bad URL hash"). The page
            # link stays the stable human reference.
            final_url = resolved_url or hit.img_src
            if "fbcdn.net" in (urlparse(final_url).hostname or ""):
                note = (note + " — " if note else "") + (
                    "Facebook signed CDN links expire in minutes — use the page link for the stable source."
                )
            # The ORIGINAL WEBSITE (hit.url) is the "found on" source that
            # matters — CDN copies are transient and useless for takedowns.
            source_urls = [hit.url] if hit.url and hit.url != final_url else [final_url]
            return SearchResult(
                url=final_url,
                page_url=hit.url,
                title=hit.title or final_url,
                engine=hit.engine,
                source="web",
                match_type=mtype,
                score=round(score, 4),
                hamming=dp,
                content_phash=int(feats["phash"]),
                width=feats["width"],
                height=feats["height"],
                thumb=preview,
                source_urls=source_urls,
                confidence=conf,
                category=cat,
                last_seen=time.time(),
                first_seen=time.time(),
                resolution=hops,
                note=note,
                ai_modalities=modalities,
                face_sim=round(face_sim, 4) if face_sim is not None else None,
                face_tier=face_tier,
                clip_sim=round(clip_sim, 4) if clip_sim is not None else None,
                geo_inliers=geo_inliers,
                clip_embedding=cand_clip or [],
            )

        # Bounded by the budget for real: the pool is shut down WITHOUT
        # waiting when time runs out (a `with` block would block on every
        # straggler, silently breaking the real-time budget).
        deadline = time.monotonic() + budget_seconds if budget_seconds is not None else None
        results: list[SearchResult] = []
        pool = ThreadPoolExecutor(max_workers=max(1, self.settings.web_verify_concurrency))
        futures = {pool.submit(verify_one, h): h for h in candidates}
        finished: set[int] = set()
        try:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            for fut in as_completed(futures, timeout=remaining):
                hit = futures[fut]
                finished.add(id(hit))
                try:
                    results.append(fut.result())
                except Exception:  # noqa: BLE001
                    results.append(self._unverified(hit, note="verification error"))
        except TimeoutError:
            pass
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        # anything not finished within the budget is reported as a
        # timeout-unverified candidate — never silently dropped, never a match
        for hit in candidates:
            if id(hit) not in finished:
                results.append(self._unverified(hit, note="verification budget exceeded", resolution=["budget-timeout"]))
        # confirmed first, then leads, then unverified; best score within
        results.sort(key=lambda r: (MATCH_FAMILY_RANK.get(r.match_type, 3), -r.score))
        return results

    @staticmethod
    def _unverified(
        hit,
        hamming: int | None = None,
        fused: float | None = None,
        note: str = "",
        resolution: list[str] | None = None,
        thumb: str = "",
        clip_sim: float | None = None,
    ) -> SearchResult:
        """A candidate that could not be fetched or did not hash-match.
        Score stays LOW regardless of how close it seemed, so an unverified
        row can never look like a strong confirmed match. ``note`` is the
        honest reason shown in the UI (fetch 403 / hash mismatch / ...).
        ``thumb`` is a locally-cached preview when the image WAS fetched but
        didn't hash-match — the thumbnail still renders in the UI even though
        it is not a confirmed copy."""
        return SearchResult(
            url=hit.img_src,
            page_url=hit.url,
            title=hit.title or hit.img_src,
            engine=hit.engine,
            source="web",
            match_type="unverified",
            score=round(max(0.0, 1.0 - (fused or 1.0)) * 0.2, 4) if hamming is not None else 0.05,
            hamming=hamming,
            # unverified candidates are always the weakest bucket; if we got
            # pixels (thumb) it's visually-similar "candidate", else just noise.
            confidence=0,
            category="candidate" if thumb else "candidate",
            last_seen=time.time(),
            first_seen=time.time(),
            resolution=resolution or [],
            note=note,
            thumb=thumb,
            source_urls=[hit.url] if hit.url and hit.url != hit.img_src else [],
            clip_sim=round(clip_sim, 4) if clip_sim is not None else None,
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
            dist = float(dists[idx])
            conf = int(round(max(0.0, min(1.0, 1.0 - dist)) * 100))
            cat = "duplicate" if dist <= 0.08 else "similar"
            out.append(
                SearchResult(
                    image_id=meta.id,
                    url=meta.url,
                    page_url=meta.page_url,
                    title=meta.title or meta.url,
                    engine=meta.engine,
                    source="local",
                    score=max(0.0, 1.0 - dist),
                    match_type="color",
                    width=meta.width,
                    height=meta.height,
                    thumb=f"/img/{meta.id}.{meta.thumb_ext}",
                    confidence=conf,
                    category=cat,
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
        # the strongest verdict wins (an exact web copy beats a local
        # "similar" row of the same URL); among equals the indexed copy wins
        best = min(group, key=lambda r: (MATCH_FAMILY_RANK.get(r.match_type, 3), 0 if r.source == "local" else 1))
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
            content_phash=best.content_phash,
            # verified web results carry their real match type (exact/near) —
            # the old blind "web" label no longer exists anywhere
            match_type=best.match_type,
            width=best.width,
            height=best.height,
            thumb=best.thumb,
            confidence=best.confidence,
            category=best.category,
            first_seen=best.first_seen,
            last_seen=best.last_seen,
            source_count=best.source_count,
            resolution=best.resolution,
            note=best.note,
            caption=best.caption or "",
            source_urls=list(best.source_urls or []),
            ai_modalities=sorted({m for r in group for m in (r.ai_modalities or [])}),
            face_sim=best.face_sim,
            face_tier=best.face_tier,
            clip_sim=best.clip_sim,
            geo_inliers=best.geo_inliers,
            clip_embedding=list(best.clip_embedding or []),
        )
        fused.append(merged)

    fused.sort(key=lambda r: (MATCH_FAMILY_RANK.get(r.match_type, 3), -r.score, r.hamming if r.hamming is not None else 99))
    return fused


def _phash_hamming(a: int | None, b: int | None) -> int | None:
    """Hamming distance between two content pHashes, None if either is missing."""
    if a is None or b is None:
        return None
    return (a ^ b).bit_count()


def consolidate_duplicates(
    results: list[SearchResult],
    threshold: int = 8,
    clip=None,
    clip_threshold: float = 0.90,
) -> list[SearchResult]:
    """Collapse results that are the SAME underlying photo into one card.

    Each result now carries ``content_phash`` — the perceptual hash of the
    bytes that were actually verified/downloaded. Multiple engines often return
    several URLs of the same photo (Logo on the company site, the same logo on
    a product page, the logo on a news article...). Without consolidation the
    grid floods with near-identical tiles of one image.

    This groups results whose content pHashes are within ``threshold`` Hamming
    bits (= same photo, possibly re-encoded / differently sized / watermarked
    copies) into a single representative whose ``source_count`` and
    ``source_urls`` carry every site the photo was found on — TinEye's
    "found on N sites". Results with no phash (unverified / noise) match no
    group and stay as-is.

    When ``clip`` (a Clipper) is live and ``clip_threshold`` > 0, a pair whose
    pHashes diverge BEYOND ``threshold`` bits (heavy re-encode / resize) can
    still be merged if their CLIP embeddings are semantically the same content
    (>= ``clip_threshold``) — this fixes the "same coffee photo, one exact one
    75%" symptom where hash-only merge left two tiles of the identical image
    with different confidences.

    Unverified candidates are never consolidated into a verified group.
    """
    if not results:
        return results
    unverified = [r for r in results if r.match_type == "unverified"]
    verified = [r for r in results if r.match_type != "unverified"]

    clip_live = bool(clip is not None and getattr(clip, "available", False) and clip_threshold > 0)

    def family(x: SearchResult) -> int:
        return MATCH_FAMILY_RANK.get(x.match_type, 3)

    groups: list[SearchResult] = []
    for r in verified:
        placed = False
        for g in groups:
            # never fold a lead (similar/ocr) or another photo of the person
            # into a confirmed-copy card, or vice versa
            if family(g) != family(r):
                continue
            d = _phash_hamming(g.content_phash, r.content_phash)
            if d is not None and d <= threshold:
                _merge_duplicate(g, r)
                placed = True
                break
            # CLIP rescue: same semantic content, hash drifted past the band.
            if (
                clip_live
                and d is not None
                and d > threshold
                and g.clip_embedding
                and r.clip_embedding
            ):
                if clip.match(g.clip_embedding, r.clip_embedding) >= clip_threshold:
                    _merge_duplicate(g, r)
                    placed = True
                    break
        if not placed:
            # seed the group with its OWN first url only when it has no
            # accumulated verified link list yet (fresh web group). Local
            # index copies already carry the canonical count/link list from
            # the repo — do not inflate it with the canonical URL again.
            if not r.source_urls:
                r.source_urls = [r.page_url or r.url]
                r.source_count = 1
            groups.append(r)

    groups.sort(key=lambda r: (family(r), -r.score, r.hamming if r.hamming is not None else 99))
    return groups + unverified


def _merge_duplicate(group: SearchResult, r: SearchResult) -> None:
    """Fuse r into group: better score wins, source URLs accumulate, engine set."""
    if r.score > group.score or (r.match_type == "exact" and group.match_type != "exact"):
        group.score = r.score
        group.match_type = r.match_type
        group.url = r.url
        group.image_id = r.image_id or group.image_id
        group.confidence = max(group.confidence, r.confidence)
    group.ai_modalities = sorted(set(group.ai_modalities or []) | set(r.ai_modalities or []))
    # engine labels are comma-joined sets — split before merging so repeated
    # merges don't grow "yandex,yandex,yandex"
    engines = {e for e in f"{group.engine},{r.engine}".split(",") if e}
    group.engine = ",".join(sorted(engines)) or group.engine
    src_set = set(group.source_urls or []) | set(r.source_urls or [])
    # The ORIGINAL WEBSITE (page_url) is the primary "found on" link — CDN
    # copies (url) are transient/hotlink-protected and useless for takedowns.
    if r.page_url:
        src_set.add(r.page_url)
    if r.url and not r.url.startswith("data:"):
        src_set.add(r.url)
    group.source_urls = sorted(s for s in src_set if s)
    group.source_count = max(group.source_count or 1, len(group.source_urls) or 1)
    group.page_url = group.page_url or r.page_url
