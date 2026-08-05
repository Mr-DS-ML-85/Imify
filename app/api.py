"""HTTP API for Imify.

Endpoints:

Public
  POST /v1/search          reverse image search (upload / url / image_id)
  GET  /v1/image/{id}      indexed image metadata
  GET  /v1/history         recent searches
  GET  /v1/stats           index statistics
  GET  /health/live|ready  probes
  GET  /img/{id}.jpg       thumbnails

Admin (API key, role=admin)
  POST   /v1/admin/ingest          add seed queries / enqueue image URLs
  GET    /v1/admin/queries         list seed queries
  PATCH  /v1/admin/queries/{id}    enable/disable
  DELETE /v1/admin/queries/{id}
  GET    /v1/admin/crawl           pipeline status
  GET    /v1/admin/images          list indexed images (filter / page)
  DELETE /v1/admin/images/{id}     remove one image from the index
  POST   /v1/admin/images/purge    bulk-remove by URL substring / query
  GET    /v1/admin/tracks          list background trackers
  PATCH  /v1/admin/tracks/{id}     enable/disable a tracker
  DELETE /v1/admin/tracks/{id}     delete a tracker
  POST   /v1/admin/tracks/run      run one tracker round now
  POST   /v1/admin/keys            create API key
  GET    /v1/admin/keys            list API keys
  DELETE /v1/admin/keys/{name}

Reverse-search ladder (SearchOS-style): local index first -> SearXNG live
reverse -> Google Images via proxify -> explicit "search not found". New
uploaded images are indexed AND tracked in the background so the index
auto-updates as matches are found across the whole internet.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from app.auth import AuthService, RateLimiter
from app.config import Settings
from app.engines import run_reverse_chain
from app.engines.bing_images import BingImagesReverse
from app.engines.google_images import GoogleImagesReverse
from app.engines.yandex_images import YandexImagesReverse
from app.features import decode_image, extract_features
from app.indexer import Indexer
from app.logging import child_logger
from app.matcher import Matcher, fuse_results
from app.models import ImageHit, SearchResponse, SearchResult
from app.repo import Repository
from app.resolve import resolve_image_bytes
from app.searx import SearxClient, DEFAULT_UA
from app.storage import StorageError


def build_api(
    settings: Settings,
    repo: Repository,
    matcher: Matcher,
    indexer: Indexer,
    auth: AuthService,
    searx: SearxClient,
    limiter: RateLimiter,
    google: GoogleImagesReverse | None = None,
    bing: BingImagesReverse | None = None,
    yandex: YandexImagesReverse | None = None,
) -> APIRouter:
    router = APIRouter()
    google = google or GoogleImagesReverse(settings.proxify_fetch_url)
    bing = bing or BingImagesReverse(
        proxify_fetch_url=settings.proxify_fetch_url,
        fetch_proxy_url=settings.fetch_proxy_url,
    )
    yandex = yandex or YandexImagesReverse(
        proxify_fetch_url=settings.proxify_fetch_url,
        fetch_proxy_url=settings.fetch_proxy_url,
    )

    # ------------------------------------------------------------- deps
    def _identity(request: Request) -> str:
        key = request.headers.get("Authorization", "").removeprefix("Bearer ").strip() or request.headers.get("X-API-Key", "")
        if key:
            record = auth.verify(key)
            if record:
                return record["name"]
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def _require_admin(authorization: Annotated[str | None, Header()] = None, x_api_key: Annotated[str | None, Header()] = None) -> dict:
        raw = (authorization or "").removeprefix("Bearer ").strip() or (x_api_key or "")
        if not raw:
            raise HTTPException(401, "admin API key required")
        record = auth.verify(raw)
        if not record or record.get("role") != "admin":
            raise HTTPException(403, "admin role required")
        return record

    def _require_user(authorization: Annotated[str | None, Header()] = None, x_api_key: Annotated[str | None, Header()] = None) -> dict | None:
        if settings.auth_mode != "key":
            return None
        raw = (authorization or "").removeprefix("Bearer ").strip() or (x_api_key or "")
        if not raw:
            raise HTTPException(401, "API key required")
        record = auth.verify(raw)
        if not record:
            raise HTTPException(401, "invalid API key")
        return record

    def _rate_limit(request: Request, identity: str) -> None:
        allowed, retry_after = limiter.check(f"search:{identity}")
        if not allowed:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(retry_after)})

    # ------------------------------------------------------------- health
    @router.get("/health/live")
    def health_live():
        return {"status": "ok"}

    @router.get("/health/ready")
    def health_ready():
        degraded: list[str] = []
        storage_ok = repo.db.ping()
        if not storage_ok:
            degraded.append("storage")
        if not searx.available:
            degraded.append("searxng")
        return {"status": "ready" if not degraded else "degraded", "degraded": degraded}

    # ------------------------------------------------------------- metrics
    @router.get("/metrics")
    def metrics():
        lines = [
            "# HELP imify_images Indexed canonical images.",
            "# TYPE imify_images gauge",
            f"imify_images {repo.image_count()}",
            "# HELP imify_searches_total Reverse searches served.",
            "# TYPE imify_searches_total counter",
            f"imify_searches_total {repo.get_stat('searches')}",
            "# HELP imify_indexed_total Images indexed since start.",
            "# TYPE imify_indexed_total counter",
            f"imify_indexed_total {indexer.status.indexed}",
            "# HELP imify_pending Pending fetch queue depth.",
            "# TYPE imify_pending gauge",
            f"imify_pending {repo.pending_count()}",
            "# HELP imify_search_found_total Searches that found at least one match.",
            "# TYPE imify_search_found_total counter",
            f"imify_search_found_total {repo.get_stat('search_found')}",
            "# HELP imify_search_notfound_total Searches that found nothing.",
            "# TYPE imify_search_notfound_total counter",
            f"imify_search_notfound_total {repo.get_stat('search_notfound')}",
        ]
        for prov, s in repo.provider_stats().items():
            lines += [
                f'# HELP imify_provider_ok_total Successful calls to reverse provider {prov}.',
                "# TYPE imify_provider_ok_total counter",
                f'imify_provider_ok_total{{provider="{prov}"}} {s["ok"]}',
                f'# HELP imify_provider_fail_total Failed calls to reverse provider {prov}.',
                "# TYPE imify_provider_fail_total counter",
                f'imify_provider_fail_total{{provider="{prov}"}} {s["fail"]}',
            ]
        # plain text, not JSON-encoded: Prometheus scrapes raw bytes
        return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    # ------------------------------------------------------------- stats
    @router.get("/v1/stats")
    def stats():
        engines: dict[str, int] = {}
        for meta in repo.all_images():
            engines[meta.engine or "unknown"] = engines.get(meta.engine or "unknown", 0) + 1
        top_engines = sorted(engines.items(), key=lambda kv: -kv[1])[:10]
        return {
            "images": repo.image_count(),
            "storage_backend": settings.storage_backend,
            "indexed_total": repo.get_stat("indexed"),
            "searches": repo.get_stat("searches"),
            "pending": repo.pending_count(),
            "queries": len(repo.list_queries()),
            "engines": dict(top_engines),
            "indexer": {
                "running": indexer.status.running,
                "indexed": indexer.status.indexed,
                "fetched": indexer.status.fetched,
                "failed": indexer.status.failed,
                "deduped": indexer.status.deduped,
                "pending": repo.pending_count(),
                "last_error": indexer.status.last_error,
            },
            "success": {
                "searches": repo.search_rate(),
                "providers": repo.provider_stats(),
                "indexer": {
                    "fetched": repo.get_stat("fetched"),
                    "failed": repo.get_stat("failed"),
                    "rate": round(
                        repo.get_stat("fetched") / (repo.get_stat("fetched") + repo.get_stat("failed")), 6
                    )
                    if (repo.get_stat("fetched") + repo.get_stat("failed")) > 0
                    else None,
                },
            },
            "uptime_s": round(time.time() - indexer.status.started) if indexer.status.started else 0,
        }

    # ------------------------------------------------------------- search
    @router.post("/v1/search")
    def search(
        request: Request,
        file: Optional[UploadFile] = File(None),
        image_url: Optional[str] = Form(None),
        image_id: Optional[int] = Form(None),
        mode: str = Form("best"),
        num_results: int = Form(40),
        include_web: bool = Form(True),
        # on-the-fly mode: default OFF — verified web matches are returned but
        # NOT persisted into the index (no storage growth). Set true to learn
        # (enqueue) verified matches so the index grows.
        learn: bool = Form(False),
        _: dict | None = Depends(_require_user),
    ):
        started = time.monotonic()
        identity = _identity(request)
        _rate_limit(request, identity)
        mode = mode if mode in ("best", "biggest", "newest", "changed", "color", "exact") else "best"
        num_results = max(1, min(num_results, settings.max_num_results))

        # ---- acquire query image -------------------------------------
        query_bytes: bytes | None = None
        query_public_url: str = ""
        if file is not None:
            query_bytes = file.file.read(settings.image_max_bytes + 1)
            if len(query_bytes) > settings.image_max_bytes:
                raise HTTPException(413, "image too large")
        elif image_url:
            if not image_url.startswith(("http://", "https://")):
                raise HTTPException(400, "image_url must be http(s)")
            # A "photo page" URL (Facebook /photo/?fbid=..., Instagram, etc.)
            # is HTML, not pixels — resolve it to the actual image bytes
            # (og:image → session-cookie FB scontent → gui_chrome render).
            resolved, resolved_url = resolve_image_bytes(
                image_url,
                cookie_file=settings.cookie_file,
                proxy=settings.fetch_proxy_url,
                proxify_fetch=settings.proxify_fetch_url,
                timeout=settings.image_fetch_timeout_seconds,
                max_bytes=settings.image_max_bytes,
            )
            if resolved is None:
                # honest fallback: plain byte fetch (direct image URLs)
                try:
                    resp = httpx.get(
                        image_url,
                        timeout=settings.image_fetch_timeout_seconds,
                        headers={"User-Agent": DEFAULT_UA},
                        follow_redirects=True,
                        proxy=settings.fetch_proxy_url or None,
                    )
                    resp.raise_for_status()
                    resolved = resp.content
                    resolved_url = image_url
                except httpx.HTTPError as exc:
                    raise HTTPException(502, f"could not fetch image_url: {exc}") from exc
            query_bytes = resolved
            query_public_url = resolved_url or image_url
        elif image_id is not None:
            meta = repo.get_image(image_id)
            if meta is None:
                raise HTTPException(404, "image_id not found")
            query_features = {
                "phash": meta.phash,
                "dhash": meta.dhash,
                "ahash": meta.ahash,
                "color": meta.color,
            }
            if meta.url.startswith(("http://", "https://")):
                query_public_url = meta.url
        else:
            raise HTTPException(400, "provide file, image_url or image_id")

        if query_bytes is not None:
            img = decode_image(query_bytes)
            if img is None:
                raise HTTPException(422, "could not decode image")
            query_features = extract_features(img, query_bytes)
            # Persist the upload NOW so the live web layer has a public URL to
            # hand to reverse engines — but LOCAL indexing is deferred until
            # the search verdict below: a failed search must never grow the
            # index with the failed image. On-the-fly matching keeps storage
            # bounded; only images verified online become index entries.
            if file is not None and settings.auto_track_searches:
                _up_sha = query_features.get("sha256") or ""
                if _up_sha and repo.sha_to_id(_up_sha) is None:
                    _name = _persist_upload(settings, query_bytes, _up_sha, file.filename or "")
                    if settings.public_base_url:
                        query_public_url = f"{settings.public_base_url.rstrip('/')}/uploads/{_name}"

        repo.incr_stat("searches")
        repo.push_history(
            {"t": time.time(), "mode": mode, "image_url": image_url or "", "image_id": image_id, "results": 0}
        )

        # ---- 1. own index first (SearchOS: "found in local -> serve local") --
        providers = ["local"]
        degraded: list[str] = []
        local_results = matcher.search_local(query_features, mode if mode != "color" else "best", num_results * 2)
        if mode == "color":
            color_results = matcher.search_color(query_features, num_results * 2)
            if color_results:
                local_results = color_results

        # ---- 2. thin/empty local coverage -> live web reverse -----------------
        # SearchOS ladder: local first -> SearXNG -> Google via proxify.
        # If every provider is down it degrades and the local index still
        # answers ("SearXNG down -> scan local only").
        web_results: list[SearchResult] = []
        verified_web: list[SearchResult] = []
        unverified_web: list[SearchResult] = []
        web_cached = False
        fallback = "local"
        # TinEye ladder: local index first (instant) — but the web layer ALWAYS
        # runs too (bounded + cached) so known images still surface remote
        # links and similar candidates, never just the local copy. min_hits=0
        # means "never skip web".
        skip_web = (
            settings.web_fallback_local_min_hits > 0
            and len(local_results) >= settings.web_fallback_local_min_hits
        )
        if settings.web_reverse_enabled and include_web and not skip_web:
            sha256 = query_features.get("sha256") or ""
            # real-time repeat search: this exact image was verified before —
            # serve the cached verified matches ("where this image lives")
            # instead of re-crawling the whole web again.
            cached_results = repo.get_web_results(sha256, settings.web_results_cache_ttl_seconds)
            if cached_results:
                web_cached = True
                web_results = [SearchResult(**r) for r in cached_results]
                verified_web = [r for r in web_results if r.match_type != "unverified"]
                unverified_web = [r for r in web_results if r.match_type == "unverified"]
                if verified_web:
                    providers.append("web")
                    fallback = "web"
            else:
                phase_start = time.monotonic()
                web_url = query_public_url
                if web_url:
                    provider_names = [p.strip() for p in settings.reverse_providers.split(",") if p.strip()]
                    hits, deg = run_reverse_chain(
                        web_url,
                        provider_names,
                        searx.reverse_search,
                        google.search,
                        bing.search,
                        yandex.search,
                        parallel=settings.web_providers_parallel,
                        budget_seconds=settings.web_phase_budget_seconds,
                    )
                    degraded.extend(deg)
                    for p in provider_names:
                        repo.record_provider(p, ok=(f"provider:{p}" not in deg))
                elif query_bytes is not None and (
                    settings.google_upload_enabled or settings.bing_upload_enabled or settings.yandex_upload_enabled
                ):
                    # Circle-to-search byte-upload path: the image has NO public
                    # URL, so the raw pixels are uploaded directly to an engine
                    # that accepts them. Yandex first (least-censored, returns
                    # face/people matches Google suppresses, browser POST proven),
                    # then Bing Visual Search, then Google Lens through proxify.
                    # URL-based reverse engines CANNOT see a private-box image —
                    # they return visually-similar junk — so byte-upload is the
                    # only honest path here.
                    up_name = (file.filename if file else None) or "image.jpg"
                    hits: list[ImageHit] = []
                    if settings.yandex_upload_enabled:
                        try:
                            hits = yandex.search_upload(query_bytes, filename=up_name)
                            repo.record_provider("yandex-upload", ok=True)
                        except Exception as exc:  # noqa: BLE001 — degrade, never fail the search
                            repo.record_provider("yandex-upload", ok=False)
                            degraded.append("provider:yandex-upload")
                            child_logger("imify.api", event="yandex_upload_degraded").warning(
                                "yandex-upload degraded", extra={"error": str(exc)}
                            )
                            hits = []
                    if not hits and settings.bing_upload_enabled:
                        try:
                            hits = bing.search_upload(query_bytes, filename=up_name)
                            repo.record_provider("bing-upload", ok=True)
                        except Exception as exc:  # noqa: BLE001 — degrade, never fail the search
                            repo.record_provider("bing-upload", ok=False)
                            degraded.append("provider:bing-upload")
                            child_logger("imify.api", event="bing_upload_degraded").warning(
                                "bing-upload degraded", extra={"error": str(exc)}
                            )
                            hits = []
                    if not hits and settings.google_upload_enabled:
                        try:
                            up_type = (file.content_type if file else None) or "image/jpeg"
                            hits = google.search_upload(query_bytes, filename=up_name, content_type=up_type)
                            repo.record_provider("google-upload", ok=True)
                        except Exception as exc:  # noqa: BLE001 — degrade, never fail the search
                            repo.record_provider("google-upload", ok=False)
                            degraded.append("provider:google-upload")
                            child_logger("imify.api", event="google_upload_degraded").warning(
                                "google-upload degraded", extra={"error": str(exc)}
                            )
                            hits = []
                else:
                    degraded.append("web_needs_public_url")
                    hits = []
                if hits:
                    # TinEye verdict: engines only FIND candidates; the fused-hash
                    # matcher JUDGES each one (downloads it, compares hashes to the
                    # query). Unverified candidates are still listed — flagged — but
                    # never presented as a confirmed site. Verification is bounded
                    # by the remaining web-phase budget so the response stays
                    # real-time even when a candidate host is slow.
                    verify_budget = max(
                        1.0,
                        settings.web_phase_budget_seconds - (time.monotonic() - phase_start),
                    )
                    web_results = matcher.verify_web_hits(
                        query_features, hits, limit=num_results * 2, budget_seconds=verify_budget
                    )
                    verified_web = [r for r in web_results if r.match_type != "unverified"]
                    unverified_web = [r for r in web_results if r.match_type == "unverified"]
                    # Cache the outcome so the next search of this exact image
                    # is served instantly (real-time). Verified matches live
                    # longer (24h, tracker refreshes them); unverified-only
                    # outcomes are cached briefly (1h) so repeat searches of
                    # "nothing found" images don't re-crawl the whole web.
                    if verified_web:
                        repo.set_web_results(
                            sha256,
                            [r.to_dict() for r in verified_web],
                            settings.web_results_cache_ttl_seconds,
                        )
                    elif unverified_web:
                        repo.set_web_results(
                            sha256,
                            [r.to_dict() for r in unverified_web],
                            min(settings.web_results_cache_ttl_seconds, 3600),
                        )
                    # learn (ONLY when requested — on-the-fly default keeps storage
                    # bounded): verified matches are recorded as SOURCE LINKS on the
                    # canonical image entry (TinEye model — one image + the list of
                    # sites carrying it), not re-fetched into new raw copies. The
                    # raw bytes are only stored once; every confirmed site becomes
                    # a link. This is what makes "which sites show my image" cheap.
                    if learn and verified_web:
                        canonical_id = _canonical_image_id(repo, query_features)
                        if canonical_id is not None:
                            for r in verified_web:
                                repo.add_source_link(canonical_id, r.url, r.page_url, r.title)
                        else:
                            for r in verified_web:
                                repo.enqueue_pending(
                                    r.url,
                                    {"page_url": r.page_url, "title": r.title, "engine": r.engine, "query": "web-reverse"},
                                )
                    if verified_web:
                        providers.append("web")
                        fallback = "web"
                    elif unverified_web:
                        # candidates existed but none hash-matched -> honest "not found"
                        degraded.append("web_candidates_unverified")
                    web_results = verified_web

        # ---- 2b. background tracker for every NEW upload ---------------------
        # The bg crawler keeps reverse-searching the upload across the whole
        # internet and ingesting NEW verified matches ("auto-update"). But a
        # failed search must NOT make the upload itself a local index entry —
        # only images verified online become searchable index copies (TinEye:
        # the index stores verified sites, not failed queries).
        if query_bytes is not None and settings.auto_track_searches:
            _maybe_track_upload(
                repo,
                settings,
                request,
                file,
                query_bytes,
                query_features,
                query_public_url,
                learn=learn,
                verified=bool(verified_web),
            )

        # ---- 3. fusion + response ("search not found" when nothing matches) ----
        fused = fuse_results(local_results, web_results) if web_results else local_results
        results = fused[:num_results]
        # unverified candidates are listed AFTER confirmed matches, clearly
        # flagged — they never count toward "found" (only hash-verified
        # matches confirm a site shows the image).
        if unverified_web:
            seen = {r.url for r in results}
            for r in unverified_web:
                if r.url not in seen:
                    results.append(r)
                    seen.add(r.url)
        found = any(r.match_type != "unverified" for r in results)
        if not found:
            fallback = "none"
        repo.incr_stat("search_results", len(results))
        repo.record_search(found)  # success-rate counter

        # update history with result count
        hist_keys = sorted(repo.db.keys("hist:"))[-1:]
        for hk in hist_keys:
            entry = repo.db.get_json(hk)
            if entry:
                entry["results"] = len(results)
                repo.db.set_json(hk, entry)

        response = SearchResponse(
            mode=mode,
            total=len(results),
            took_ms=int((time.monotonic() - started) * 1000),
            providers=providers,
            degraded=degraded,
            found=found,
            fallback=fallback,
            cached=web_cached,
            results=[r.to_dict() for r in results],
        )
        return response.to_dict()

    # ------------------------------------------------------------- images
    @router.get("/v1/image/{image_id}")
    def image_detail(image_id: int):
        meta = repo.get_image(image_id)
        if meta is None:
            raise HTTPException(404, "image not found")
        return meta.to_dict()

    @router.get("/v1/history")
    def history(limit: int = 20):
        return repo.list_history(limit)

    # ------------------------------------------------------------- admin
    @router.post("/v1/admin/ingest")
    def admin_ingest(payload: dict, admin: dict = Depends(_require_admin)):
        """Add seed queries and/or enqueue specific image URLs for indexing."""
        queries = payload.get("queries") or []
        urls = payload.get("urls") or []
        if not queries and not urls:
            raise HTTPException(400, "provide a non-empty list of queries and/or urls")
        created = []
        for q in queries:
            if isinstance(q, str):
                created.append(repo.put_query(q))
            elif isinstance(q, dict) and q.get("query"):
                created.append(repo.put_query(q["query"], enabled=q.get("enabled", True), interval_seconds=int(q.get("interval_seconds") or 0)))
        enqueued = 0
        for u in urls:
            if not isinstance(u, str) or not u.startswith(("http://", "https://", "file://")):
                continue
            if repo.enqueue_pending(u, {"query": "admin-upload", "engine": "admin"}):
                enqueued += 1
        indexer.request_discovery()  # kick off immediately, never block the request
        return {"created": created, "count": len(created), "urls_enqueued": enqueued}

    @router.get("/v1/admin/queries")
    def admin_queries(admin: dict = Depends(_require_admin)):
        return {"queries": repo.list_queries()}

    @router.patch("/v1/admin/queries/{query_id}")
    def admin_query_toggle(query_id: int, payload: dict, admin: dict = Depends(_require_admin)):
        repo.set_query_enabled(query_id, bool(payload.get("enabled")))
        return {"ok": True}

    @router.delete("/v1/admin/queries/{query_id}")
    def admin_query_delete(query_id: int, admin: dict = Depends(_require_admin)):
        repo.delete_query(query_id)
        return {"ok": True}

    @router.get("/v1/admin/crawl")
    def admin_crawl(admin: dict = Depends(_require_admin)):
        st = indexer.snapshot()
        # KV-backed totals stay consistent even when the pipeline runs in a
        # separate worker process.
        return {
            "running": st.running,
            "started": st.started,
            "fetched_total": repo.get_stat("fetched"),
            "indexed_total": repo.get_stat("indexed"),
            "failed_total": repo.get_stat("failed"),
            "deduped_total": repo.get_stat("deduped"),
            "session_indexed": st.indexed,
            "pending": st.pending,
            "workers_alive": st.workers_alive,
            "searx_requests": st.searx_requests,
            "searx_circuit_open": st.searx_circuit_open,
            "discovery_last_run": st.discovery_last_run,
            "last_progress": st.last_progress,
            "last_error": st.last_error,
        }

    # ------------------------------------------------- index removals
    @router.get("/v1/admin/images")
    def admin_images(
        limit: int = 50,
        offset: int = 0,
        q: str = "",
        admin: dict = Depends(_require_admin),
    ):
        metas = repo.all_images()
        if q:
            ql = q.lower()
            metas = [
                m
                for m in metas
                if ql in m.url.lower() or ql in (m.title or "").lower() or ql in (m.engine or "").lower()
            ]
        metas.sort(key=lambda m: m.first_seen, reverse=True)
        total = len(metas)
        page = [m.to_dict() for m in metas[offset : offset + limit]]
        # 64-bit phash as hex string — JS numbers lose precision past 2^53
        for d in page:
            d["phash"] = f"{d['phash']:016x}"
        return {"total": total, "offset": offset, "limit": limit, "images": page}

    @router.delete("/v1/admin/images/{image_id}")
    def admin_image_delete(image_id: int, admin: dict = Depends(_require_admin)):
        if not repo.remove_image(image_id):
            raise HTTPException(404, "image not found")
        matcher.invalidate_histogram_cache()
        return {"ok": True, "image_id": image_id}

    @router.post("/v1/admin/images/purge")
    def admin_images_purge(payload: dict, admin: dict = Depends(_require_admin)):
        removed = repo.purge_images(
            url_contains=str(payload.get("url_contains") or ""),
            query=str(payload.get("query") or ""),
        )
        matcher.invalidate_histogram_cache()
        return {"ok": True, "removed": removed}

    # ---------------------------------------------------- trackers
    @router.get("/v1/admin/tracks")
    def admin_tracks(admin: dict = Depends(_require_admin)):
        tracks = repo.list_tracks()
        for t in tracks:
            t["phash"] = f"{int(t.get('phash') or 0):016x}"  # hex string (see above)
        return {"tracks": tracks}

    @router.patch("/v1/admin/tracks/{track_id}")
    def admin_track_toggle(track_id: int, payload: dict, admin: dict = Depends(_require_admin)):
        repo.set_track_enabled(track_id, bool(payload.get("enabled")))
        return {"ok": True}

    @router.delete("/v1/admin/tracks/{track_id}")
    def admin_track_delete(track_id: int, admin: dict = Depends(_require_admin)):
        if not repo.delete_track(track_id):
            raise HTTPException(404, "track not found")
        return {"ok": True}

    @router.post("/v1/admin/tracks/run")
    def admin_tracks_run(admin: dict = Depends(_require_admin)):
        """Run one tracker round now (background, never blocks the request)."""
        indexer.request_tracker_round()
        return {"ok": True}

    @router.post("/v1/admin/keys")
    def admin_create_key(payload: dict, admin: dict = Depends(_require_admin)):
        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name required")
        role = payload.get("role", "user")
        if role not in ("user", "admin"):
            raise HTTPException(400, "role must be user or admin")
        raw = auth.create_key(name, role)
        return {"api_key": raw, "name": name, "role": role}

    @router.get("/v1/admin/keys")
    def admin_list_keys(admin: dict = Depends(_require_admin)):
        return {"keys": auth.list_keys()}

    @router.delete("/v1/admin/keys/{name}")
    def admin_delete_key(name: str, admin: dict = Depends(_require_admin)):
        return {"ok": auth.delete_key(name)}

    return router


# ------------------------------------------------------------------ helpers
def _canonical_image_id(repo: Repository, query_features: dict) -> int | None:
    """Find the canonical indexed image for the query (by sha256 or phash),
    or None when it isn't indexed yet — verified web hits then fall back to
    being enqueued as fresh pending fetches."""
    sha = query_features.get("sha256") or ""
    if sha:
        cid = repo.sha_to_id(sha)
        if cid is not None:
            return cid
    phash = int(query_features.get("phash") or 0)
    if phash:
        return repo.phash_to_id(phash)
    return None


def _persist_upload(settings: Settings, raw: bytes, sha256: str, original_name: str) -> str:
    """Save an uploaded image under data/uploads/<sha256><ext>; returns the name."""
    suffix = Path(original_name or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}:
        suffix = ".jpg"
    name = f"{sha256}{suffix}"
    path = Path(settings.uploads_dir) / name
    if not path.exists():
        Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return name


def _maybe_track_upload(
    repo: Repository,
    settings: Settings,
    request: Request,
    file: UploadFile | None,
    raw: bytes,
    features: dict,
    query_public_url: str,
    learn: bool = False,
    verified: bool = False,
) -> str:
    """Continuous-learning hook for a NEW uploaded image.

    A user's own upload always gets a background tracker ("search the whole
    internet for it in the bg") so NEW verified matches are ingested over
    time. But the upload itself only becomes a local INDEX entry when the
    search VERIFIED it online (``verified``) or the user opted into learning
    (``learn``) — a failed search must never grow the index with the failed
    image (the old behaviour: search fails -> image indexed -> next search
    returns the upload's local path instead of remote links).

    - persists the upload so it has a (public) URL and a byte-upload path,
    - enqueues it into the own index ONLY when verified/learn,
    - registers a background tracker that keeps reverse-searching the whole
      internet for it and ingesting new matches (auto-update of the index).

    Returns the public URL that reverse engines can reach ("" when none).
    """
    if not settings.auto_track_searches:
        return ""
    sha = features.get("sha256") or ""
    if not sha or repo.sha_to_id(sha) is not None:
        return ""  # already indexed -> already tracked
    phash = int(features.get("phash") or 0)
    # reverse_url is what external engines (SearXNG / Google via proxify) can
    # fetch. Only public_base_url uploads are reachable from the internet;
    # on a private box this stays empty and the ladder degrades honestly.
    reverse_url = query_public_url
    local_path = ""
    if file is not None:
        name = _persist_upload(settings, raw, sha, file.filename or "")
        # index the upload locally ONLY when the search verified it online
        # (or the user explicitly opted in) — the worker fetches it over
        # loopback (request.base_url), no public exposure needed.
        if verified or learn:
            internal_url = f"{str(request.base_url).rstrip('/')}/uploads/{name}"
            repo.enqueue_pending(internal_url, {"query": "user-upload", "engine": "upload", "title": file.filename or ""})
        local_path = str(Path(settings.uploads_dir) / name)
        if settings.public_base_url:
            reverse_url = f"{settings.public_base_url.rstrip('/')}/uploads/{name}"
        else:
            reverse_url = ""
    repo.upsert_track(
        sha,
        phash,
        public_url=reverse_url,
        title=file.filename if file else query_public_url,
        local_path=local_path,
        dhash=int(features.get("dhash") or 0),
        ahash=int(features.get("ahash") or 0),
    )
    return reverse_url
