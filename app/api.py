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
import io
import math

import httpx
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from app.auth import AuthService, RateLimiter
from app.ai import AiClient
from app.clip import Clipper
from app.enhance import Enhancer
from app.config import Settings
from app.cookies import cookie_health
from app.engines import run_reverse_chain
from app.engines.bing_images import BingImagesReverse
from app.engine_guard import EngineGuard
from app.engines.text_images import BROWSER_SEARCH_URLS, TextImageDiscovery, derive_queries, is_iconlike
from app.engines.google_images import GoogleImagesReverse
from app.engines.yandex_images import YandexImagesReverse
from app.features import decode_image, extract_features, hash_variants
from app.face import FaceId
from app.indexer import Indexer, local_pixels
from app.logging import child_logger
from app.matcher import Matcher, consolidate_duplicates, fuse_results
from app.models import COPY_MATCH_TYPES, MATCH_FAMILY_RANK, ImageHit, ImageMeta, SearchResponse, SearchResult
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
    ai: AiClient | None = None,
    face: FaceId | None = None,
    clip: Clipper | None = None,
    enhancer: Enhancer | None = None,
    text_discovery: TextImageDiscovery | None = None,
) -> APIRouter:
    router = APIRouter()
    ai = ai or AiClient(
        url=settings.ai_base_url or settings.ai_ollama_url,
        caption_model=settings.ai_caption_model,
        embed_model=settings.ai_embed_model,
        api_key=settings.ai_api_key,
        timeout_seconds=settings.ai_caption_timeout_seconds,
        caption_url=settings.ai_caption_url,
    )
    face = face or FaceId.from_settings(settings)
    clip = clip or Clipper.from_settings(settings)
    enhancer = enhancer or Enhancer(
        enabled=settings.enhancer_enabled,
        models_dir=str(settings.enhance_models_dir),
        blur_limit=settings.enhance_blur_limit,
    )

    # one shared model instance per layer (matcher verifies web candidates
    # with the same face/CLIP models the API and indexer use)
    if getattr(matcher, "clip", None) is None:
        matcher.clip = clip
    if getattr(matcher, "face", None) is None:
        matcher.face = face

    def _local_ai_grid(
        query_img,
        known: set[int],
        query_faces: list[list[float]] | None,
        query_clip: list[float] | None,
        limit: int = 12,
    ) -> list[SearchResult]:
        """Merge THIS search's own-AI matches (same face / similar content /
        same visible text) from the LOCAL index into the main grid.

        Excludes image_ids already served as hash copies (``known``). Pure
        local, no web calls, so it never adds latency to the web ladder.
        """
        from app.ocr import OcrExtractor as _OE
        out: list[SearchResult] = []
        seen: set[int] = set(known)
        images = [m for m in repo.all_images() if m.id not in seen]
        by_id = {m.id: m for m in images}

        def _wrap(meta: ImageMeta, score: float, cat: str, mtype: str, **extra) -> SearchResult:
            return SearchResult(
                image_id=meta.id,
                url=meta.url,
                page_url=meta.page_url,
                title=meta.title or meta.url,
                engine=meta.engine,
                source="local",
                score=round(float(score), 4),
                match_type=mtype,
                category=cat,
                width=meta.width,
                height=meta.height,
                thumb=f"/img/{meta.id}.{meta.thumb_ext}",
                first_seen=meta.first_seen,
                last_seen=meta.last_seen,
                source_count=meta.source_count,
                source_urls=list(meta.source_urls or []),
                caption=meta.caption or "",
                content_phash=meta.phash,
                **extra,
            )

        # face: same person anywhere in the library (FaceCheck-style)
        if query_faces and face.available:
            try:
                candidates = [
                    (meta.id, emb)
                    for meta in images
                    if meta.face_model == FaceId.MODEL_TAG
                    for emb in (meta.face_embeddings or [])
                ]
                best: dict[int, float] = {}
                for qemb in query_faces:
                    for sim, image_id in face.match_identity(qemb, candidates):
                        best[image_id] = max(sim, best.get(image_id, 0.0))
                for image_id, sim in sorted(best.items(), key=lambda t: -t[1])[: limit * 2]:
                    meta = by_id[image_id]
                    conf, tier = face.confidence_tier(sim)
                    out.append(_wrap(
                        meta, sim, "face-" + tier, "face",
                        confidence=conf, face_sim=round(float(sim), 4),
                        face_tier=tier, ai_modalities=["face"],
                        note="same person",
                    ))
                    seen.add(image_id)
            except Exception as exc:  # noqa: BLE001 — AI layer best-effort
                child_logger("imify.api").warning("face grid failed", extra={"error": str(exc)})

        # clip: visually similar content ("what else is like this")
        if query_clip and clip.available:
            try:
                scored = [
                    (meta, clip.match(query_clip, meta.clip_embedding))
                    for meta in images
                    if meta.id not in seen and meta.clip_embedding
                ]
                scored.sort(key=lambda t: -t[1])
                for meta, sim in scored[:limit]:
                    if sim < settings.clip_similar_min_sim:
                        break
                    out.append(_wrap(
                        meta, 0.3 * sim, "similar", "similar",
                        confidence=int(round(sim * 100)),
                        clip_sim=round(float(sim), 4), ai_modalities=["clip"],
                    ))
                    seen.add(meta.id)
            except Exception as exc:  # noqa: BLE001
                child_logger("imify.api").warning("clip grid failed", extra={"error": str(exc)})

        # ocr: match the visible text in the query against indexed OCR
        ocr = _OE(enabled=settings.ocr_enabled, lang=settings.ocr_lang)
        if ocr.available and query_img is not None:
            try:
                qtext = (ocr.extract(query_img) or "").lower()
                qtokens = sorted({t for t in qtext.split() if len(t) > 2})
                if len(qtokens) >= 2:
                    for meta in images:
                        if meta.id in seen or not meta.ocr_text:
                            continue
                        mtokens = set(meta.ocr_text.lower().split())
                        hits = sum(1 for t in qtokens if t in mtokens)
                        if hits >= max(2, (len(qtokens) + 1) // 2):
                            out.append(_wrap(
                                meta, 0.2 * hits / len(qtokens), "similar", "ocr",
                                confidence=int(round(100 * hits / len(qtokens))),
                                ai_modalities=["ocr"], note=f"{hits} words of on-image text matched",
                            ))
                            seen.add(meta.id)
            except Exception as exc:  # noqa: BLE001
                child_logger("imify.api").warning("ocr grid failed", extra={"error": str(exc)})

        out.sort(key=lambda r: (MATCH_FAMILY_RANK.get(r.match_type, 3), -r.score))
        return out[: limit * 3]

    guard = EngineGuard(repo.db)
    text_discovery = text_discovery or TextImageDiscovery(
        settings.searxng_url,
        proxify_fetch_url=settings.proxify_fetch_url,
        timeout=settings.searxng_timeout_seconds,
        engines=[e.strip() for e in settings.text_discovery_engines.split(",") if e.strip()] or None,
        browser_fallback_max=settings.text_discovery_browser_max,
        guard=guard,
    )

    def _text_discover(hits: list, img, phase_start: float) -> tuple[list, str, dict]:
        """Lens-style expansion of the reverse hits: best-guess query from
        their titles -> SearXNG image/icon engines (+ browser retry) ->
        extra candidates for the SAME verifier. Bounded so verification
        keeps its share of the web budget."""
        if not (settings.text_discovery_enabled and text_discovery.available):
            return [], "", {}
        remaining = settings.web_phase_budget_seconds - (time.monotonic() - phase_start)
        budget = min(settings.text_discovery_budget_seconds, remaining - 4.0)
        if budget < 1.5:
            return [], "", {"skipped": "no budget left"}
        queries = derive_queries(hits)
        if not queries:
            return [], "", {"skipped": "no best guess"}
        icons = settings.text_discovery_icons
        include_icons = icons == "always" or (icons == "auto" and is_iconlike(img))
        report = text_discovery.search(queries, include_icons=include_icons, budget_seconds=budget)
        for eng, n in report.engines.items():
            repo.record_provider(f"text:{eng}", ok=n > 0)
        info = {
            "queries": report.queries,
            "engines": report.engines,
            "blocked": report.blocked,
            "browser_used": report.browser_used,
            "icons": include_icons,
        }
        return report.hits, queries[0], info

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
            "face": {
                "enabled": face.available,
                "provider": face.provider,
                "images_with_faces": sum(1 for m in repo.all_images() if m.face_embeddings),
                "images": repo.image_count(),
            },
            "clip": {
                "enabled": clip.available,
                "provider": clip.provider,
                "images_embedded": sum(1 for m in repo.all_images() if m.clip_embedding),
                "images": repo.image_count(),
            },
            "cookies": cookie_health(settings.cookie_file),
            "engine_guard": guard.status(
                ["yandex-upload", "bing-upload", "google-upload"]
                + [f"browser:{e}" for e in BROWSER_SEARCH_URLS]
            ),
            "facebook_resolution": {
                "enabled": settings.facebook_resolution_enabled,
                **repo.provider_stats().get("facebook-render", {}),
            },
            "enhance": {
                "enabled": enhancer.enabled,
                "available": enhancer.available,
                "blur_limit": settings.enhance_blur_limit,
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
        query_clip: list[float] | None = None
        query_faces: list[dict] = []  # [{embedding, bbox, confidence}]
        img = None
        enhance_info: dict = {}
        _enh_img = None
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
            # AI layers (face / CLIP / geometry) need pixels: use the best
            # local copy (original upload or thumbnail) — no network.
            img = local_pixels(settings, meta)
        else:
            raise HTTPException(400, "provide file, image_url or image_id")

        if query_bytes is not None:
            img = decode_image(query_bytes)
            if img is None:
                raise HTTPException(422, "could not decode image")
            query_features = extract_features(img, query_bytes)
            # Blur restore (Opt-in): if the query is a blurry photo, sharpen it
            # with Real-ESRGAN x4 + BiRefNet BEFORE hashing and web upload so a
            # smile/blur no longer hashes as "sea". When enhance_before_web_upload
            # is set, the engines also receive the sharpened bytes.
            enhance_info: dict = {}
            _enh_img: object = None
            if enhancer and enhancer.enabled and enhancer.available and enhancer.needs_enhance(img):
                _enh, _was, _info = enhancer.enhance(img)
                if _was:
                    _buf = io.BytesIO()
                    _enh.convert("RGB").save(_buf, format="JPEG", quality=95)
                    _enhanced_bytes = _buf.getvalue()
                    _enh_img = decode_image(_enhanced_bytes) or _enh
                    query_features = extract_features(_enh_img, _enhanced_bytes)
                    if settings.enhance_before_web_upload:
                        query_bytes = _enhanced_bytes
                    enhance_info = _info
            # CLIP semantic gate: embed the query once so web candidates can be
            # content-checked during verification (two different logos that
            # happen to hash close must NOT verify as a match).
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

        # Query-side AI signals, computed ONCE and reused by the local AI grid
        # and by every web candidate's verification: CLIP (semantic gate +
        # "similar"), faces (FaceCheck identity), pixels for geometry.
        ai_img = (_enh_img or img) if enhance_info else img
        if ai_img is not None:
            # rotated / mirrored re-uploads (phone photos) hash differently —
            # match against the query in all 8 orientations
            query_features["variants"] = hash_variants(ai_img)
            if clip.available:
                query_clip = clip.embed_image(ai_img)
            if face.available:
                try:
                    query_faces = face.embed_image(ai_img)
                except Exception:  # noqa: BLE001 — face layer is best-effort
                    query_faces = []
        query_face_embs = [f["embedding"] for f in query_faces]

        repo.incr_stat("searches")
        repo.push_history(
            {"t": time.time(), "mode": mode, "image_url": image_url or "", "image_id": image_id, "results": 0}
        )

        # ---- 1. own index first (fast) -------------------------------------
        providers = ["local"]
        degraded: list[str] = []
        local_results = matcher.search_local(
            query_features, mode if mode != "color" else "best", num_results * 2, query_clip=query_clip
        )
        if mode == "color":
            color_results = matcher.search_color(query_features, num_results * 2)
            if color_results:
                local_results = color_results

        # ---- 1b. AI layers feed the SAME grid (not separate endpoints) -----
        # Face-ID / CLIP / OCR historically lived in their own /v1/search/* -
        # but a blurry person photo or a logo with text never surfaced in the
        # main grid. Merge each available layer's LOCAL matches into this same
        # ranked list (an "ai" provider) so one search answers with copies,
        # semantic-similar content AND the same face/text — never hashes only.
        if ai_img is not None and mode in ("best", "exact", "ai"):
            # crops / edits of the query inside the own index (keypoints)
            _known = {int(r.image_id) for r in local_results if r.image_id is not None}
            if image_id is not None:
                _known.add(image_id)
            _geo = matcher.search_local_geometric(ai_img, query_clip, exclude=_known)
            if _geo:
                local_results.extend(_geo)
                local_results.sort(key=lambda r: (MATCH_FAMILY_RANK.get(r.match_type, 3), -r.score))
            _ai_grid = _local_ai_grid(
                ai_img,
                known={int(r.image_id) for r in local_results if r.image_id is not None}
                | ({image_id} if image_id is not None else set()),
                query_faces=query_face_embs,
                query_clip=query_clip,
            )
            if _ai_grid:
                local_results.extend(_ai_grid)
                providers.append("ai")

        # ---- 2. thin/empty local coverage -> live web reverse -----------------
        # SearchOS ladder: local first -> SearXNG -> Google via proxify.
        # If every provider is down it degrades and the local index still
        # answers ("SearXNG down -> scan local only").
        web_results: list[SearchResult] = []
        verified_web: list[SearchResult] = []  # confirmed: copies + same face
        similar_web: list[SearchResult] = []  # CLIP leads (Google-Lens "similar")
        unverified_web: list[SearchResult] = []
        web_cached = False
        best_guess = ""
        discovery: dict = {}
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
                all_cached = [SearchResult(**r) for r in cached_results]
                verified_web = [r for r in all_cached if r.confirmed]
                similar_web = [r for r in all_cached if r.match_type == "similar"]
                unverified_web = [r for r in all_cached if r.match_type == "unverified"]
                # Cached unverified candidates must pass the SAME admission
                # gate below as fresh ones — never dumped into the grid.
                web_results = verified_web + similar_web
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
                    # URL, so the raw pixels are uploaded directly to each engine
                    # that accepts them. ALL engines are asked in PARALLEL within
                    # the shared web-phase budget (TinEye model: no single engine
                    # sees the whole web) and every candidate is merged — Bing's
                    # visually-similar junk never silences Yandex/Google. The
                    # fused-hash matcher judges each candidate afterwards.
                    up_name = (file.filename if file else None) or "image.jpg"
                    up_type = (file.content_type if file else None) or "image/jpeg"
                    up_bytes = query_bytes
                    # Engines decode only JPEG/PNG reliably — Bing silently
                    # answers a WebP/AVIF/HEIC upload with unrelated generic
                    # results. Re-encode anything else as JPEG for the engines.
                    if img is not None and (getattr(img, "format", None) or _sniff_format(query_bytes)) not in ("JPEG", "PNG"):
                        _buf = io.BytesIO()
                        img.convert("RGB").save(_buf, format="JPEG", quality=92)
                        up_bytes = _buf.getvalue()
                        up_name = Path(up_name).stem + ".jpg"
                        up_type = "image/jpeg"
                    # SearXNG is URL-based (no raw-byte upload), so hand it a
                    # copy the app itself serves back to it over loopback
                    # (SearXNG and the app both run on this box). Empty when no
                    # local upload exists (e.g. image_id replay) -> SearXNG
                    # simply degrades instead of failing the merge.
                    local_url = query_public_url
                    if not local_url and file is not None:
                        local_url = f"{str(request.base_url).rstrip('/')}/uploads/{_persist_upload(settings, query_bytes, query_features.get('sha256') or '', up_name)}"
                    hits, hit_by_engine = _byte_upload_all(
                        settings, yandex, bing, google, searx,
                        up_bytes, up_name, up_type,
                        image_url=local_url,
                        phase_start=phase_start,
                        # leave time for verification inside the same budget
                        budget=settings.web_phase_budget_seconds * 0.7,
                        face_crop=_face_crop_bytes(settings, ai_img, query_faces),
                        guard=guard,
                    )
                    # record per-provider success so success-rate metrics are honest
                    for eng, ok in hit_by_engine.items():
                        if ok is None:  # skipped by the guard (cooldown / pacing) — not tried
                            degraded.append(f"cooldown:{eng}")
                            continue
                        repo.record_provider(f"{eng}-upload", ok=ok)
                        if not ok:
                            degraded.append(f"provider:{eng}-upload")
                            child_logger("imify.api", event=f"{eng}_upload_degraded").warning(
                                f"{eng}-upload degraded", extra={"error": "no candidates"}
                            )
                else:
                    degraded.append("web_needs_public_url")
                    hits = []
                # Lens-style text discovery: best guess from the reverse hits'
                # titles -> image/icon engines -> more candidates to verify
                if hits:
                    text_hits, best_guess, discovery = _text_discover(hits, ai_img, phase_start)
                    if text_hits:
                        cap = settings.web_verify_max_candidates
                        known_src = {h.img_src for h in hits}
                        text_hits = [h for h in text_hits if h.img_src not in known_src]
                        # reverse-engine hits stay first (they carry the copies);
                        # text hits get at least a third of the verify slots
                        n_text = min(len(text_hits), max(cap // 3, cap - len(hits)))
                        hits = hits[: max(0, cap - n_text)] + text_hits[:n_text]
                        providers.append("text")
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
                        query_features,
                        hits,
                        limit=num_results * 2,
                        budget_seconds=verify_budget,
                        query_clip=query_clip,
                        query_img=ai_img,
                        query_faces=query_face_embs,
                    )
                    verified_web = [r for r in web_results if r.confirmed]
                    similar_web = [r for r in web_results if r.match_type == "similar"]
                    unverified_web = [r for r in web_results if r.match_type == "unverified"]
                    # Cache the outcome so the next search of this exact image
                    # is served instantly. Confirmed outcomes live 24h (the
                    # tracker refreshes them); lead-only outcomes 1h so repeat
                    # searches of "nothing found" images don't re-crawl.
                    if web_results:
                        repo.set_web_results(
                            sha256,
                            [r.to_dict() for r in web_results],
                            settings.web_results_cache_ttl_seconds
                            if verified_web
                            else min(settings.web_results_cache_ttl_seconds, 3600),
                        )
                    # learn (ONLY when requested — on-the-fly default keeps
                    # storage bounded). Copies of the image become SOURCE LINKS
                    # on the canonical entry (TinEye: one image + every site
                    # carrying it); other photos of the same person become new
                    # index entries (the FaceCheck asset: an index of faces).
                    if learn and verified_web:
                        canonical_id = _canonical_image_id(repo, query_features)
                        for r in verified_web:
                            if canonical_id is not None and r.match_type in COPY_MATCH_TYPES:
                                repo.add_source_link(canonical_id, r.url, r.page_url, r.title)
                            else:
                                repo.enqueue_pending(
                                    r.url,
                                    {"page_url": r.page_url, "title": r.title, "engine": r.engine, "query": "web-reverse"},
                                )
                    if verified_web:
                        providers.append("web")
                        fallback = "web"
                    elif similar_web or unverified_web:
                        # candidates existed but none was confirmed -> honest "not found"
                        degraded.append("web_candidates_unverified")
                    web_results = verified_web + similar_web

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
                verified=any(r.match_type in ("exact", "near") for r in verified_web),
                verified_web=[r for r in verified_web if r.match_type in ("exact", "near")],
            )

        # ---- 3. fusion + response ("search not found" when nothing matches) ----
        fused = fuse_results(local_results, web_results) if web_results else local_results
        # Consolidate multiple URLs/sites carrying the SAME photographic content
        # (same perceptual hash → genuine near-duplicate re-encodes/downloads)
        # into ONE result whose source list grows — "found on N sites". Different
        # images (even visually similar ones) have different phashes and stay
        # separate, so the grid never floods with 4 tiles of one image.
        fused = consolidate_duplicates(
            fused,
            settings.search_dedup_threshold,
            clip=clip,
            clip_threshold=settings.search_dedup_clip_threshold,
        )
        results = fused[:num_results]
        # Reference-RIS behavior: engines return candidate PREVIEWS, but only
        # the ones that are plausibly the same image belong in the grid. Bing
        # visual-search (and sometimes Yandex) returns visually-similar junk
        # (animal/girl previews for a company logo) that must not be shown as a
        # match. So unverified candidates are admitted ONLY if (a) the pixels
        # actually decoded (they have a thumb, so they render) and (b) their
        # perceptual distance is within a tight similarity window. Anything
        # farther is noise — dropped, not flagged. They are still listed after
        # confirmed matches and never count toward "found".
        # Reference-RIS behavior: engines return candidate PREVIEWS, but only
        # plausible ones belong in the grid. With CLIP live every candidate
        # was already judged on content — an unverified one failed hash,
        # geometry, face AND similarity, so it is noise and stays out. Without
        # CLIP, fall back to a tight perceptual window (Bing previews are
        # penalized for their junk history). Listed after everything else and
        # never counted as found.
        if unverified_web and settings.web_verify_candidates:
            seen = {r.url for r in results}
            by_sim: list[tuple[float, SearchResult]] = []
            for r in unverified_web:
                if r.url in seen or r.clip_sim is not None or r.hamming is None or not r.thumb:
                    continue
                pen = settings.web_engine_bing_penalty if r.engine and "bing" in r.engine.lower() else 1.0
                by_sim.append(((r.hamming / 64.0) * pen, r))
            by_sim.sort(key=lambda t: t[0])
            for eff, r in by_sim:
                if eff > settings.web_unverified_max_fused:
                    break  # sorted: all remaining are even less similar
                results.append(r)
                seen.add(r.url)
        found = any(r.confirmed for r in results)
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
            best_guess=best_guess,
            discovery=discovery,
            results=[r.to_dict() for r in results],
        )
        # reverseimagesearch / reversely grouping — feed the UI per-engine tabs
        # and Similar/Duplicates buckets over the final ranked list.
        rd = [r.to_dict() for r in results]
        response.by_engine = {}
        for r in rd:
            eng = (r.get("engine") or "").strip() or ("index" if r.get("source") == "local" else "web")
            response.by_engine.setdefault(eng, []).append(r)
        response.by_category = {"duplicate": [], "similar": [], "candidate": []}
        for r in rd:
            cat = r.get("category") or ("duplicate" if r.get("match_type") in ("exact", "near") else "candidate")
            if cat not in response.by_category:
                response.by_category[cat] = []
            response.by_category[cat].append(r)
        return response.to_dict()

    # --------------------------------------------------------- AI text search
    @router.post("/v1/search/text")
    def search_by_text(
        request: Request,
        query: str = Form(...),
        num_results: int = Form(40),
        _: dict | None = Depends(_require_user),
    ):
        """Natural-language search over the AI-captioned index.

        Matches the query against every indexed image's caption using embedding
        cosine similarity (self-hosted Ollama). Requires ``AI_OLLAMA_URL`` +
        an embedding model and captioned images (``AI_CAPTION_ON_INDEX``).
        Degrades honestly: 503 when the AI layer is off or the server is down.
        """
        started = time.monotonic()
        identity = _identity(request)
        _rate_limit(request, identity)
        if not ai.enabled or not ai.available:
            raise HTTPException(503, "AI layer unavailable (set AI_OLLAMA_URL and start Ollama)")
        qvec = ai.embed(text=query)
        if not qvec:
            raise HTTPException(503, "AI embedding failed (check AI_EMBED_MODEL)")
        floor = settings.ai_text_min_sim  # drop semantically-unrelated captions
        scored: list[tuple[float, dict]] = []
        for meta in repo.all_images():
            cvec = meta.caption_embedding
            # fallback for entries indexed before caption-embedding existed:
            # embed the caption once here (best-effort) rather than skip it.
            if not cvec and meta.caption:
                cvec = ai.embed(text=meta.caption)
            if not cvec:
                continue
            sim = _cosine(qvec, cvec)
            if sim < floor:
                continue
            scored.append((sim, _public_meta(meta)))
        scored.sort(key=lambda t: -t[0])
        top = scored[: max(1, min(num_results, settings.max_num_results))]
        results = []
        for sim, meta in top:
            d = dict(meta)
            d["score"] = round(float(sim), 4)
            d["thumb"] = f"/img/{meta['id']}.{meta.get('thumb_ext') or 'jpg'}"
            d["match_type"] = "text"
            d["page_url"] = meta.get("page_url") or meta.get("url", "")
            d["category"] = "similar"
            d["confidence"] = int(round(max(0.0, min(1.0, float(sim))) * 100))
            results.append(d)
        response = SearchResponse(
            mode="text",
            total=len(scored),
            took_ms=int((time.monotonic() - started) * 1000),
            providers=["ai"],
            found=bool(results),
            fallback="local",
            text_query=True,
            results=results,
        )
        return response.to_dict()

    # --------------------------------------------------------- OCR text search
    @router.post("/v1/search/ocr")
    def search_by_ocr_text(
        request: Request,
        query: str = Form(...),
        num_results: int = Form(40),
        _: dict | None = Depends(_require_user),
    ):
        """Keyword search over OCR text extracted from indexed images.

        Matches the query tokens against every indexed image's ``ocr_text``
        (visible text in logos/memes/screenshots). Simple TF-ish scoring — no
        external services, pure string matching. 503 when OCR is disabled or
        tesseract is not installed.
        """
        started = time.monotonic()
        identity = _identity(request)
        _rate_limit(request, identity)
        from app.ocr import OcrExtractor

        ocr = OcrExtractor(enabled=settings.ocr_enabled, lang=settings.ocr_lang)
        if not ocr.available:
            raise HTTPException(503, "OCR unavailable (set OCR_ENABLED=true and install tesseract)")
        tokens = [t.lower() for t in query.split() if len(t) > 1]
        if not tokens:
            raise HTTPException(400, "query too short")
        scored: list[tuple[float, dict]] = []
        for meta in repo.all_images():
            if not meta.ocr_text:
                continue
            text_l = meta.ocr_text.lower()
            hits = sum(1 for t in tokens if t in text_l)
            if hits == 0:
                continue
            score = hits / len(tokens)  # fraction of query tokens found
            scored.append((score, _public_meta(meta)))
        scored.sort(key=lambda t: -t[0])
        top = scored[: max(1, min(num_results, settings.max_num_results))]
        results = []
        for sim, meta in top:
            d = dict(meta)
            d["score"] = round(float(sim), 4)
            d["match_type"] = "ocr"
            d["thumb"] = f"/img/{meta['id']}.{meta.get('thumb_ext') or 'jpg'}"
            d["page_url"] = meta.get("page_url") or meta.get("url", "")
            d["category"] = "similar"
            d["confidence"] = int(round(max(0.0, min(1.0, float(sim))) * 100))
            results.append(d)
        response = SearchResponse(
            mode="ocr",
            total=len(scored),
            took_ms=int((time.monotonic() - started) * 1000),
            providers=["ocr"],
            found=bool(results),
            fallback="local",
            results=results,
        )
        return response.to_dict()

    # ---------------------------------------------- Full-text search
    @router.post("/v1/search/fulltext")
    def search_by_fulltext(
        request: Request,
        query: str = Form(...),
        num_results: int = Form(40),
        _: dict | None = Depends(_require_user),
    ):
        """Keyword search over EVERY text field on the index entry.

        Matches the query tokens (case-insensitive substring) against the AI
        caption, OCR text, title and clipped filename of every indexed image —
        so a search for a quoted phrase or a set of keywords hits the full
        index regardless of which layer produced the text. Pure string scoring,
        no external service (works with or without AI/OCR enabled).

        The returned score is the fraction of query tokens found across the
        entry's combined text.
        """
        started = time.monotonic()
        identity = _identity(request)
        _rate_limit(request, identity)
        # Caption boilerplate + English stopwords. Without filtering, common
        # words ("image", "with", "a", "the", "shows", "depicts") appear in
        # every AI caption and fulltext search returns the WHOLE index for
        # almost any query.
        _STOP = frozenset(
            "a an the and or but if because of in on at by for with without from to"
            " is are was were be been being has have had do does did will would shall"
            " should can could may might must not no yes i you he she it we they them"
            " this that these those there here image picture photo photograph depicts"
            " shows features displaying presents scene subject background setting showing"
            " features of front against over under between during through across into"
            " out up down off again further once also some any all both each few more"
            " most other own same so than too very s t can just don should now".split()
        )
        tokens = [t.lower() for t in query.split() if len(t) > 2 and t.lower() not in _STOP]
        if not tokens:
            raise HTTPException(400, "query has no meaningful keywords")
        min_hits = max(1, math.ceil(len(tokens) / 2))  # majority of meaningful tokens must hit
        scored: list[tuple[float, dict]] = []
        for meta in repo.all_images():
            caption = (meta.caption or "").lower()
            ocr = (meta.ocr_text or "").lower()
            title = (meta.title or "").lower()
            name = Path(meta.url or "").name.lower().rsplit(".", 1)[0].replace("-", " ").replace("_", " ")
            haystack = f" {caption} {ocr} {title} {name} "
            hits = sum(1 for t in tokens if t in haystack)
            if hits < min_hits:
                continue
            score = hits / len(tokens)
            scored.append((score, _public_meta(meta)))
        scored.sort(key=lambda t: -t[0])
        top = scored[: max(1, min(num_results, settings.max_num_results))]
        results = []
        for sim, meta in top:
            d = dict(meta)
            d["score"] = round(float(sim), 4)
            d["match_type"] = "fulltext"
            d["thumb"] = f"/img/{meta['id']}.{meta.get('thumb_ext') or 'jpg'}"
            d["page_url"] = meta.get("page_url") or meta.get("url", "")
            d["category"] = "similar"
            d["confidence"] = int(round(max(0.0, min(1.0, float(sim))) * 100))
            results.append(d)
        response = SearchResponse(
            mode="fulltext",
            total=len(scored),
            took_ms=int((time.monotonic() - started) * 1000),
            providers=["local"],
            found=bool(results),
            fallback="local",
            results=results,
        )
        return response.to_dict()

    # --------------------------------------------------------- Face-ID search
    @router.post("/v1/search/face")
    def search_by_face(
        request: Request,
        file: UploadFile = File(...),
        num_results: int = Form(40),
        _: dict | None = Depends(_require_user),
    ):
        """Match a face in the uploaded photo across the whole indexed library.

        Detects + embeds the face (ArcFace, CPU) and ranks every indexed
        image that has a detected face by embedding cosine similarity, with
        FaceCheck-style confidence tiers. Returns the matched images grouped by
        "this person appears here". 503 when Face-ID is disabled / model not
        downloaded / onnxruntime missing.
        """
        started = time.monotonic()
        identity = _identity(request)
        _rate_limit(request, identity)
        if not face.available:
            raise HTTPException(503, "Face-ID unavailable (set FACE_ID_ENABLED=true and install opencv + onnxruntime)")
        raw = file.file.read(settings.image_max_bytes + 1)
        if len(raw) > settings.image_max_bytes:
            raise HTTPException(413, "image too large")
        img = decode_image(raw)
        if img is None:
            raise HTTPException(422, "could not decode image")
        faces = face.embed_image(img)
        if not faces:
            raise HTTPException(404, "no face detected in the uploaded image")
        # every query face (group photos) against every current-pipeline
        # library face; each image keeps its best similarity
        candidates = [
            (meta.id, emb)
            for meta in repo.all_images()
            if meta.face_model == FaceId.MODEL_TAG
            for emb in (meta.face_embeddings or [])
        ]
        best: dict[int, float] = {}
        for qf in faces:
            for sim, image_id in face.match_identity(qf["embedding"], candidates):
                best[image_id] = max(sim, best.get(image_id, 0.0))
        results = []
        for image_id, sim in best.items():
            meta = repo.get_image(image_id)
            if meta is None:
                continue
            conf, tier = face.confidence_tier(sim)
            d = _public_meta(meta)
            d["confidence"] = conf
            d["face_confidence"] = conf
            d["face_tier"] = tier
            d["face_sim"] = round(sim, 4)
            d["score"] = round(float(sim), 4)
            d["match_type"] = "face"
            d["category"] = "face-" + tier
            d["thumb"] = f"/img/{meta.id}.{meta.thumb_ext}"
            d["image_id"] = meta.id
            results.append(d)
        results.sort(key=lambda r: -r["score"])
        results = results[: max(1, min(num_results, settings.max_num_results))]
        response = SearchResponse(
            mode="face",
            total=len(results),
            took_ms=int((time.monotonic() - started) * 1000),
            providers=["face"],
            found=bool(results),
            fallback="local",
            results=results,
        )
        return response.to_dict()

    # ------------------------------------------------------- CLIP content search
    @router.post("/v1/search/clip")
    def search_by_clip(
        request: Request,
        file: UploadFile = File(...),
        num_results: int = Form(40),
        _: dict | None = Depends(_require_user),
    ):
        """Semantic content search — "what else is like this scene".

        Perceptual hashes only separate copies of the SAME image; CLIP ranks
        the whole library by semantic similarity (a homelab photo matches
        another homelab photo, not a black-dog photo). Embeds the uploaded
        image (ViT-B/32 on CPU) and scores every indexed image that has a CLIP
        embedding by cosine similarity. 503 when CLIP is disabled.
        """
        started = time.monotonic()
        identity = _identity(request)
        _rate_limit(request, identity)
        if not clip.available:
            raise HTTPException(503, "CLIP unavailable (set CLIP_ENABLED=true and install onnxruntime)")
        raw = file.file.read(settings.image_max_bytes + 1)
        if len(raw) > settings.image_max_bytes:
            raise HTTPException(413, "image too large")
        img = decode_image(raw)
        if img is None:
            raise HTTPException(422, "could not decode image")
        qvec = clip.embed_image(img)
        if not qvec:
            raise HTTPException(503, "CLIP embedding failed (model did not produce a vector)")
        scored: list[tuple[float, dict]] = []
        for meta in repo.all_images():
            if not meta.clip_embedding:
                continue
            sim = clip.match(qvec, meta.clip_embedding)
            if sim >= settings.clip_similar_min_sim:
                d = _public_meta(meta)
                d["thumb"] = f"/img/{meta.id}.{meta.thumb_ext}"
                d["image_id"] = meta.id
                d["confidence"] = int(round(sim * 100))
                d["clip_sim"] = round(float(sim), 4)
                d["score"] = round(float(sim), 4)
                d["match_type"] = "similar"
                d["category"] = "duplicate" if sim >= settings.clip_same_threshold else "similar"
                scored.append((sim, d))
        scored.sort(key=lambda t: -t[0])
        top = scored[: max(1, min(num_results, settings.max_num_results))]
        response = SearchResponse(
            mode="clip",
            total=len(top),
            took_ms=int((time.monotonic() - started) * 1000),
            providers=["clip"],
            found=bool(top),
            fallback="local",
            results=[d for _, d in top],
        )
        return response.to_dict()

    # ------------------------------------------------------------- images
    @router.get("/v1/image/{image_id}")
    def image_detail(image_id: int):
        meta = repo.get_image(image_id)
        if meta is None:
            raise HTTPException(404, "image not found")
        return _public_meta(meta)
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

    @router.post("/v1/admin/face/backfill")
    def admin_face_backfill(
        limit: int = 50,
        force: bool = False,
        admin: dict = Depends(_require_admin),
    ):
        """(Re-)embed faces for index entries that have none yet or were
        embedded by an older face pipeline (``force`` re-embeds everything).

        The indexer also does this continuously in the background; this is
        the "do it now" button. Uses the original upload when available,
        else the stored thumbnail. Bounded by ``limit`` per call.
        """
        if not face.available:
            raise HTTPException(503, "Face-ID unavailable (install opencv + onnxruntime, enable FACE_ID_ENABLED)")
        done = indexer.backfill_ai(limit=limit, force_faces=force)
        return {"scanned": done["scanned"], "faces_added": done["faces"], "images_embedded": done["clip"], "provider": face.provider}

    @router.post("/v1/admin/clip/backfill")
    def admin_clip_backfill(
        limit: int = 50,
        admin: dict = Depends(_require_admin),
    ):
        """Embed CLIP vectors for index entries that lack them (also done
        continuously in the background by the indexer)."""
        if not clip.available:
            raise HTTPException(503, "CLIP unavailable (install onnxruntime, enable CLIP_ENABLED)")
        done = indexer.backfill_ai(limit=limit)
        return {"scanned": done["scanned"], "images_embedded": done["clip"], "faces_added": done["faces"], "provider": clip.provider}

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
        page = [_public_meta(m) for m in metas[offset : offset + limit]]
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
_HEAVY_FIELDS = ("face_embeddings", "clip_embedding", "caption_embedding", "color")


def _public_meta(meta: ImageMeta) -> dict:
    """ImageMeta as JSON for clients — without the raw embedding vectors
    (kilobytes per image that no client uses)."""
    d = meta.to_dict()
    for key in _HEAVY_FIELDS:
        d.pop(key, None)
    return d


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors (for AI text search)."""
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _byte_upload_all(
    settings,
    yandex,
    bing,
    google,
    searx,
    query_bytes: bytes,
    up_name: str,
    up_type: str,
    image_url: str,
    phase_start: float,
    budget: float,
    face_crop: bytes | None = None,
    guard: EngineGuard | None = None,
) -> tuple[list, dict[str, bool | None]]:
    """Run every enabled byte-upload engine in PARALLEL and merge candidates.

    TinEye model: no single engine sees the whole web. Yandex (least-censored,
    returns faces Google suppresses), Bing Visual Search, Google Lens, and
    SearXNG are all asked concurrently within the shared web-phase budget, and
    their candidates are merged (deduped by image URL). A slow or throttled
    engine degrades on its own instead of blocking the others.

    ``face_crop`` (JPEG bytes of the main face) adds a "yandex-face" task:
    FaceCheck-style people search — Yandex matches people by the face, so a
    tight crop of a group/holiday photo finds other photos of that person
    that the full frame never would. Its candidates are face-verified.

    SearXNG is URL-based (no raw-byte upload), so ``image_url`` must be a copy
    the app itself can serve back to SearXNG on loopback. Pass ``""`` when none
    exists — SearXNG is then recorded degraded instead of failing the merge.

    Returns ``(merged_hits, {engine: ok})`` where ok=False means the engine
    returned nothing, raised, or missed the budget.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

    tasks: list[tuple[str, object]] = []
    if settings.yandex_upload_enabled:
        tasks.append(("yandex", yandex))
        if face_crop:
            tasks.append(("yandex-face", yandex))
    if settings.bing_upload_enabled:
        tasks.append(("bing", bing))
    if settings.google_upload_enabled:
        tasks.append(("google", google))
    if settings.searxng_url and image_url and getattr(searx, "available", True):
        tasks.append(("searx", searx))
    if not tasks:
        return [], {}

    per_engine: dict[str, list] = {}
    skipped: dict[str, str] = {}

    def _guarded(engine: str, eng) -> list:
        """EngineGuard around one engine call: skip while it cools down after
        a block, pace it (jittered min interval), feed the outcome back."""
        if guard is None or engine == "searx":  # own SearXNG: nothing to hide from
            return _run(engine, eng)
        cool_key = "yandex-upload" if engine == "yandex-face" else f"{engine}-upload"
        allowed, why = guard.allow(cool_key)
        if not allowed:
            skipped[engine] = why
            return []
        if not guard.pace(engine if engine == "yandex-face" else cool_key):
            skipped[engine] = "paced (engine used moments ago)"
            return []
        try:
            out = _run(engine, eng)
        except Exception as exc:
            guard.report(cool_key, ok=False, text=str(exc))
            raise
        guard.report(cool_key, ok=True)
        return out

    def _run(engine: str, eng) -> list:
        if engine == "yandex":
            return eng.search_upload(query_bytes, filename=up_name)
        if engine == "yandex-face":
            return eng.search_upload(face_crop, filename="face.jpg")
        if engine == "google":
            return eng.search_upload(query_bytes, filename=up_name, content_type=up_type)
        if engine == "searx":
            return eng.reverse_search(image_url)
        return eng.search_upload(query_bytes, filename=up_name)

    # The budget is enforced for real: on timeout the pool is abandoned
    # without waiting (a `with` block would block on the slowest engine).
    pool = ThreadPoolExecutor(max_workers=len(tasks))
    futures = {pool.submit(_guarded, e, eng): e for e, eng in tasks}
    try:
        remaining = None if budget is None else max(0.5, budget - (time.monotonic() - phase_start))
        for fut in as_completed(futures, timeout=remaining):
            engine = futures[fut]
            try:
                per_engine[engine] = list(fut.result() or [])
            except Exception as exc:  # noqa: BLE001 — degrade, never fail the search
                child_logger("imify.api", event=f"{engine}_upload_errored").warning(
                    f"{engine}-upload errored", extra={"error": str(exc)}
                )
    except TimeoutError:
        child_logger("imify.api", event="upload_budget").warning(
            "byte-upload budget exhausted", extra={"pending": [e for f, e in futures.items() if not f.done()]}
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # merge all candidates, deduped by image URL, INTERLEAVING across engines
    # so one engine's flood never fills the entire verify cap. Order within
    # the response is decided by the verifier, not here.
    merged: list = []
    seen: set[str] = set()
    order = ("yandex-face", "yandex", "bing", "google", "searx")
    lists = [(e, per_engine.get(e, [])) for e in order]
    max_len = max((len(lst) for _, lst in lists), default=0)
    for i in range(max_len):
        for provider, lst in lists:
            if i >= len(lst):
                continue
            hit = lst[i]
            # SearXNG reports its internal sub-engine ("google images"); tag
            # hits with the top-level provider so a searx hit never masks a
            # genuinely failed direct engine in the per-engine tabs.
            if provider == "searx" and getattr(hit, "engine", ""):
                try:
                    hit.engine = "searx"
                except Exception:  # noqa: BLE001
                    pass
            img_src = getattr(hit, "img_src", "") or ""
            if img_src in seen:
                continue
            seen.add(img_src)
            merged.append(hit)

    # None = skipped by the guard (not a failure: the engine was not asked)
    hit_by_engine: dict[str, bool | None] = {
        e: (None if e in skipped else bool(per_engine.get(e))) for e, _ in tasks
    }
    return merged, hit_by_engine


def _sniff_format(raw: bytes | None) -> str:
    """PIL format name of raw image bytes ("" when unknown)."""
    if not raw:
        return ""
    try:
        from PIL import Image as _Image

        with _Image.open(io.BytesIO(raw)) as probe:
            return probe.format or ""
    except Exception:  # noqa: BLE001
        return ""


def _face_crop_bytes(settings: Settings, img, faces: list[dict]) -> bytes | None:
    """JPEG of the main face with context, when the photo is NOT already a
    close-up (then the full upload is the face search). None otherwise."""
    if not settings.face_crop_upload or img is None or not faces:
        return None
    x, y, w, h = faces[0]["bbox"]
    if (w * h) / float(img.width * img.height) > 0.20:
        return None  # portrait / selfie: the full image already is the face
    crop = FaceId.crop_face(img, faces[0]["bbox"])
    if min(crop.size) < 64:
        return None
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


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
    verified_web: list | None = None,
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
            # Canonical link preference: when the web layer VERIFIED this upload
            # on a real remote site, index that REMOTE source (url + page_url)
            # as the canonical — never the loopback copy. Otherwise the indexed
            # entry's url is http://127.0.0.1:8010/uploads/... and every later
            # search of the image returns our own dead loopback link instead of
            # the site that actually hosts it. The worker fetches the remote URL;
            # the local upload stays only as the on-disk thumb/upload copy.
            remote = None
            for r in verified_web or []:
                u = (r.url or "").strip()
                if u.startswith(("http://", "https://")) and "127.0.0.1" not in u and "localhost" not in u:
                    remote = r
                    break
            if remote is not None:
                repo.enqueue_pending(
                    remote.url,
                    {
                        "page_url": remote.page_url or remote.url,
                        "title": remote.title or file.filename or "",
                        "engine": "upload",
                        "query": "user-upload",
                    },
                )
            else:
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
