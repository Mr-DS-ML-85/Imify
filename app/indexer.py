"""The indexing pipeline — SearchOS philosophy applied to images.

SearchOS's core ideas, ported:

- **Search state as an asset** — every seed query, pending URL and indexed
  image lives in durable KV, so the pipeline resumes exactly where it left
  off across restarts.
- **Discovery -> fetch -> process workers** — an orchestrator schedules
  SearXNG discovery; a pool of workers fetch, validate, feature-extract,
  deduplicate and store images.
- **Coverage map** — each seed query tracks last run + counts; queries are
  re-run on a schedule so the index stays fresh.
- **Middleware safeguards** — retry with backoff, circuit breaker for a
  throttled SearXNG, stall detection, per-image fetch guards (size, dimension,
  timeout, content validation).
- **Learns from the web** — reverse-search hits are enqueued into the own
  index so repeat searches are served from memory, not from the web.

Workers run in plain threads; the FastAPI app talks to them only through the
shared Repository + an in-memory status snapshot (thread-safe).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import httpx
from PIL import Image

from app.config import Settings
from app.ai import AiClient
from app.engines import run_reverse_chain
from app.engines.bing_images import BingImagesReverse
from app.engines.google_images import GoogleImagesReverse
from app.engines.yandex_images import YandexImagesReverse
from app.clip import Clipper
from app.engine_guard import EngineGuard
from app.features import decode_image, extract_features, hamming
from app.face import FaceId
from app.logging import child_logger
from app.ocr import OcrExtractor
from app.models import ImageHit, ImageMeta
from app.repo import Repository
from app.searx import SearxClient, DEFAULT_UA

Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard


def _is_remote_url(url: str) -> bool:
    """True for a real internet URL (http/https). Local/loopback copies —
    ``file://`` upload paths and ``http://127.0.0.1:8010/uploads/...`` — are
    NOT remote; the canonical link must point at the site hosting the image,
    never at our own storage."""
    if not url:
        return False
    low = url.lower()
    if not (low.startswith("http://") or low.startswith("https://")):
        return False
    host = (url.split("://", 1)[1].split("/", 1)[0] or "").lower()
    if host in {"127.0.0.1", "localhost"} or host.startswith("127.") or host.startswith("10.") or host.startswith("192.168."):
        return False
    return True



@dataclass
class PipelineStatus:
    running: bool = False
    started: float = 0.0
    discovery_last_run: float = 0.0
    last_progress: float = 0.0
    last_error: str = ""
    fetched: int = 0
    indexed: int = 0
    failed: int = 0
    deduped: int = 0
    pending: int = 0
    workers_alive: int = 0
    searx_requests: int = 0
    searx_circuit_open: bool = False


class Indexer:
    def __init__(
        self,
        repo: Repository,
        settings: Settings,
        searx: SearxClient | None = None,
        google: GoogleImagesReverse | None = None,
        bing: BingImagesReverse | None = None,
        yandex: YandexImagesReverse | None = None,
        ai: AiClient | None = None,
        ocr: OcrExtractor | None = None,
        face: FaceId | None = None,
        clip: Clipper | None = None,
    ):
        self.repo = repo
        self.settings = settings
        self.ocr = ocr or OcrExtractor(enabled=settings.ocr_enabled, lang=settings.ocr_lang)
        self.face = face or FaceId.from_settings(settings)
        self.clip = clip or Clipper.from_settings(settings)
        self.searx = searx or SearxClient(
            settings.searxng_url,
            timeout=settings.searxng_timeout_seconds,
            min_interval=settings.searxng_query_interval,
            max_pages=settings.searxng_max_pages,
        )
        # Google reverse provider for the tracker loop (SearXNG fallback).
        self.google = google or GoogleImagesReverse(
            settings.proxify_fetch_url,
            fetch_proxy_url=settings.fetch_proxy_url,
            use_browser=settings.proxify_use_browser,
            direct_upload=settings.google_upload_direct,
        )
        # Bing reverse provider (byte-upload for private-box images).
        self.bing = bing or BingImagesReverse(
            proxify_fetch_url=settings.proxify_fetch_url,
            fetch_proxy_url=settings.fetch_proxy_url,
        )
        # Yandex reverse provider (least-censored, finds faces Google hides).
        self.yandex = yandex or YandexImagesReverse(
            proxify_fetch_url=settings.proxify_fetch_url,
            fetch_proxy_url=settings.fetch_proxy_url,
        )
        self.ai = ai
        # background work may wait longer for an engine slot than a live search
        self.guard = EngineGuard(repo.db, max_wait=60.0)
        self.log = child_logger("imify.indexer", worker="indexer")
        self.status = PipelineStatus()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self.status.running:
            return
        self.status.running = True
        self.status.started = time.time()
        self.status.last_progress = time.time()
        threads = [
            threading.Thread(target=self._discovery_loop, name="discovery", daemon=True),
            threading.Thread(target=self._recrawl_loop, name="recrawl", daemon=True),
            threading.Thread(target=self._tracker_loop, name="tracker", daemon=True),
            threading.Thread(target=self._ai_backfill_loop, name="ai-backfill", daemon=True),
        ]
        for i in range(self.settings.indexer_fetch_workers):
            threads.append(threading.Thread(target=self._worker_loop, name=f"fetch-{i}", daemon=True))
        self._threads = threads
        for t in threads:
            t.start()
        # first tracker round promptly (new trackers have last_run=0 -> due now)
        if self.settings.auto_track_searches:
            self.request_tracker_round()
        self.log.info("indexer started", extra={"event": "start", "workers": len(threads)})

    def stop(self) -> None:
        self.status.running = False
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)

    def snapshot(self) -> PipelineStatus:
        with self._lock:
            st = self.status
            st.pending = self.repo.pending_count()
            st.workers_alive = sum(1 for t in self._threads if t.is_alive())
            st.searx_requests = self.searx.total_requests
            st.searx_circuit_open = not self.searx.available
            return st

    # ------------------------------------------------------- discovery loop
    def _discovery_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_discovery_round()
            except Exception as exc:  # noqa: BLE001 — worker thread must survive
                self._set_error(f"discovery: {exc}")
                self.log.warning("discovery round failed", extra={"event": "discovery_error", "error": str(exc)})
            self._stop.wait(self.settings.crawler_interval_seconds)

    def _run_discovery_round(self) -> None:
        queries = [q for q in self.repo.list_queries() if q.get("enabled")]
        if not queries:
            self.status.discovery_last_run = time.time()
            return
        for q in queries:
            if self._stop.is_set():
                return
            query_text = q["query"]
            now = time.time()
            interval = q.get("interval_seconds") or self.settings.crawler_interval_seconds
            if now - q.get("last_run", 0) < interval:
                continue
            try:
                hits = self.searx.discover(query_text)
                enqueued = 0
                for hit in hits:
                    if self.repo.enqueue_pending(
                        hit.img_src,
                        {
                            "page_url": hit.url,
                            "title": hit.title,
                            "engine": hit.engine,
                            "query": query_text,
                            "width": hit.width,
                            "height": hit.height,
                        },
                    ):
                        enqueued += 1
                self.repo.touch_query(q["id"], len(hits))
                self.status.discovery_last_run = now
                self._bump_progress()
                self.log.info(
                    "discovered",
                    extra={"event": "discover", "query": query_text, "hits": len(hits), "enqueued": enqueued},
                )
            except Exception as exc:  # noqa: BLE001
                self._set_error(f"discover '{query_text}': {exc}")

    # ---------------------------------------------------------- recrawl loop
    def _recrawl_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.settings.crawler_interval_seconds)
            try:
                self._recrawl_stale()
                self._cleanup_stale_leases()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("recrawl failed", extra={"event": "recrawl_error", "error": str(exc)})

    def _cleanup_stale_leases(self) -> None:
        """A worker that crashed mid-task leaves a lease behind; reset it after
        10 minutes so the URL can be claimed again."""
        now = time.time()
        for key in self.repo.db.keys("pending:"):
            data = self.repo.db.get_json(key)
            if data and now - data.get("added", 0) > 600:
                self.repo.db.delete("claim:" + key[len("pending:"):])
                self.log.info("lease reset", extra={"event": "lease_reset", "url": data.get("url", "")})

    def _recrawl_stale(self) -> None:
        cutoff = time.time() - self.settings.crawler_recrawl_after_seconds
        stale = [m for m in self.repo.all_images() if m.last_seen < cutoff][:200]
        if not stale:
            return
        for meta in stale:
            self.repo.enqueue_pending(
                meta.url,
                {
                    "page_url": meta.page_url,
                    "title": meta.title,
                    "engine": meta.engine,
                    "query": meta.query,
                    "width": meta.width,
                    "height": meta.height,
                    "existing_id": meta.id,
                },
            )
        self.log.info("enqueued recrawl", extra={"event": "recrawl", "count": len(stale)})

    # ----------------------------------------------------------- worker loop
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            batch = []
            try:
                batch = self.repo.claim_pending(8)
            except Exception as exc:  # noqa: BLE001
                self._set_error(f"claim_pending: {exc}")
                self._stop.wait(self.settings.indexer_poll_interval_seconds)
                continue
            for item in batch:
                if self._stop.is_set():
                    return
                task = item["data"]
                try:
                    ok = self._process_task(task)
                except Exception as exc:  # noqa: BLE001
                    self._set_error(f"process: {exc}")
                    ok = False
                if ok:
                    self.repo.finish_pending(item, done=True)
                else:
                    attempts = int(task.get("attempts", 0)) + 1
                    if attempts >= self.settings.indexer_max_attempts:
                        self._incr_failed()
                        self.repo.finish_pending(item, done=True)  # give up
                    else:
                        self.repo.finish_pending(item, done=False)  # retry later
            self._stop.wait(self.settings.indexer_poll_interval_seconds)

    def _process_task(self, task: dict) -> bool:
        """Fetch, validate, deduplicate and index one image. Returns success."""
        url: str = task["url"]
        self.log.debug("fetching", extra={"url": url})
        raw = self._fetch_image(url)
        if raw is None:
            return False
        self.status.fetched += 1
        self.repo.incr_stat("fetched")
        img = decode_image(raw)
        if img is None or min(img.size) < self.settings.image_min_dimension:
            return False

        features = extract_features(img, raw)
        existing = self._find_duplicate(features["sha256"], features["phash"], url)
        if existing is not None:
            self._merge_duplicate(existing, url, task)
            self.status.deduped += 1
            self.repo.incr_stat("deduped")
            self._bump_progress()
            return True

        image_id = self.repo.next_image_id()
        now = time.time()
        meta = ImageMeta(
            id=image_id,
            url=url,
            page_url=task.get("page_url", ""),
            title=task.get("title", ""),
            engine=task.get("engine", ""),
            query=task.get("query", ""),
            phash=int(features["phash"]),
            dhash=int(features["dhash"]),
            ahash=int(features["ahash"]),
            color=features["color"],
            sha256=features["sha256"],
            width=features["width"],
            height=features["height"],
            size_bytes=features["size_bytes"],
            first_seen=now,
            last_seen=now,
            source_urls=[url],
        )
        # Write order: vector + metadata atomically-ish; maps last.
        try:
            self.repo.db.vadd(image_id, _hash_to_vector(meta.phash))
            self.repo.put_image(meta)
            self.repo.register_sha(meta.sha256, image_id)
            self.repo.register_url(url, image_id)
            self.repo.register_phash(meta.phash, image_id)
            self._save_thumbnail(image_id, img)
            # Optional AI: caption the newly-indexed image (best-effort, non-
            # blocking). The caption becomes searchable alt-text and powers
            # natural-language search. Failures never fail the index write.
            if (
                self.settings.ai_caption_on_index
                and self.ai is not None
                and self.ai.available
                and raw
            ):
                try:
                    cap = self.ai.caption(
                        raw,
                        mime="image/jpeg",
                        max_len=240,
                    )
                    if cap:
                        meta.caption = cap
                        # persist the caption embedding at INDEX time too, so
                        # by-description search is a cosine over stored vectors
                        # (never a per-query re-embed of every caption).
                        try:
                            if self.ai and self.ai.available:
                                cvec = self.ai.embed(text=cap)
                                if cvec:
                                    meta.caption_embedding = cvec
                        except Exception:  # noqa: BLE001
                            pass
                        self.repo.put_image(meta)
                except Exception:  # noqa: BLE001
                    pass
            # Optional OCR: extract visible text from the newly-indexed image.
            if self.ocr.available:
                try:
                    ocr_text = self.ocr.extract(img)
                    if ocr_text:
                        meta.ocr_text = ocr_text
                        self.repo.put_image(meta)
                except Exception:  # noqa: BLE001
                    pass
            # Optional Face-ID: detect faces + extract identity embeddings.
            if self.face.available:
                try:
                    faces = self.face.embed_image(img)
                    meta.face_embeddings = [f["embedding"] for f in faces]
                    meta.face_count = len(faces)
                    meta.face_model = FaceId.MODEL_TAG
                    self.repo.put_image(meta)
                except Exception:  # noqa: BLE001
                    pass
            # Optional CLIP: whole-image semantic embedding ("what is this").
            if self.clip.available:
                try:
                    clip_emb = self.clip.embed_image(img)
                    if clip_emb:
                        meta.clip_embedding = clip_emb
                        self.repo.put_image(meta)
                except Exception:  # noqa: BLE001
                    pass
            self.repo.incr_stat("indexed")
            self.status.indexed += 1
            self._bump_progress()
            self.repo.db.set("stat:pipeline_last_progress", str(time.time()))
            # storage guard: LRU-evict oldest when over the cap so the index
            # can never fill the disk (on-the-fly matching keeps working)
            try:
                self.repo.evict_if_full(self.settings.max_index_images)
            except Exception:  # noqa: BLE001 — eviction must never break indexing
                pass
            self.log.info(
                "indexed",
                extra={"event": "indexed", "image_id": image_id, "url": url, "w": meta.width, "h": meta.height},
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "index failed",
                extra={"event": "index_failed", "url": url, "error": str(exc)},
                exc_info=True,
            )
            return False
        return True

    def _find_duplicate(self, sha256: str, phash: int, url: str) -> ImageMeta | None:
        existing = self.repo.sha_to_id(sha256) or self.repo.url_to_id(url) or self.repo.phash_to_id(phash)
        if existing:
            return self.repo.get_image(existing)
        return None

    def _merge_duplicate(self, meta: ImageMeta, url: str, task: dict) -> None:
        meta.fetch_count += 1
        meta.last_seen = time.time()
        # A REAL remote source URL (the article/page carrying this image) is
        # always preferred as the canonical link over a local/loopback copy
        # (file:// upload path, http://127.0.0.1:8010/uploads/...). Without
        # this, a user upload indexed via loopback becomes the canonical entry
        # and the original remote URL only lands in source_urls — so the next
        # search returns the dead local link instead of the site that hosts it.
        if _is_remote_url(url) and not _is_remote_url(meta.url):
            meta.url = url
        if url not in meta.source_urls:
            meta.source_urls.append(url)
            meta.source_count = len(meta.source_urls)
            if task.get("page_url") and task["page_url"] != meta.page_url:
                meta.page_url = task["page_url"]
            if task.get("title") and not meta.title:
                meta.title = task["title"]
        self.repo.update_image(meta)
        self.repo.register_url(url, meta.id)
        self.log.debug("deduped", extra={"event": "dedup", "image_id": meta.id, "url": url})

    # ------------------------------------------------------- engine guard
    def _guarded_call(self, key: str, fn):
        """Background tracker calls go through the same EngineGuard as live
        searches: skipped while the engine cools down after a block, paced
        with jitter, and the outcome is fed back. A skip returns no hits."""
        allowed, why = self.guard.allow(key)
        if not allowed or not self.guard.pace(key):
            self.log.info("tracker skipped engine", extra={"engine": key, "reason": why or "paced"})
            return []
        try:
            out = fn()
        except Exception as exc:
            self.guard.report(key, ok=False, text=str(exc))
            raise
        self.guard.report(key, ok=True)
        return out

    # ------------------------------------------------------- AI backfill
    def _ai_backfill_loop(self) -> None:
        """Keep every index entry's face/CLIP embeddings current.

        Entries indexed before a layer was enabled, or embedded by an older
        (buggy) face pipeline, are re-embedded in small batches in the
        background — searches never pay for it and never see stale vectors.
        """
        self._stop.wait(5)
        while not self._stop.is_set():
            try:
                done = self.backfill_ai(limit=100)
            except Exception as exc:  # noqa: BLE001 — worker thread must survive
                self.log.warning("ai backfill failed", extra={"event": "ai_backfill_error", "error": str(exc)})
                done = {}
            busy = done.get("faces", 0) + done.get("clip", 0)
            self._stop.wait(2 if busy else 600)

    def backfill_ai(self, limit: int = 50, force_faces: bool = False) -> dict:
        """Re-embed up to ``limit`` entries whose face embeddings are missing
        or from an older pipeline, or whose CLIP embedding is missing."""
        face_on = self.face.available
        clip_on = self.clip.available
        if not (face_on or clip_on):
            return {"scanned": 0, "faces": 0, "clip": 0}
        scanned = faces_done = clip_done = 0
        for meta in self.repo.all_images():
            if scanned >= limit or self._stop.is_set():
                break
            need_face = face_on and (force_faces or meta.face_model != FaceId.MODEL_TAG)
            need_clip = clip_on and not meta.clip_embedding
            if not (need_face or need_clip):
                continue
            scanned += 1
            img = local_pixels(self.settings, meta)
            if img is None:
                # nothing local to embed from: drop stale vectors so they can
                # never produce a wrong identity match, and stop retrying
                if need_face:
                    meta.face_embeddings = []
                    meta.face_count = 0
                    meta.face_model = FaceId.MODEL_TAG
                    self.repo.put_image(meta)
                continue
            if need_face:
                faces = self.face.embed_image(img)
                meta.face_embeddings = [f["embedding"] for f in faces]
                meta.face_count = len(faces)
                meta.face_model = FaceId.MODEL_TAG
                faces_done += 1
            if need_clip:
                emb = self.clip.embed_image(img)
                if emb:
                    meta.clip_embedding = emb
                    clip_done += 1
            self.repo.put_image(meta)
        if faces_done or clip_done:
            self.log.info("ai backfill", extra={"event": "ai_backfill", "faces": faces_done, "clip": clip_done})
        return {"scanned": scanned, "faces": faces_done, "clip": clip_done}

    # --------------------------------------------------------------- helpers
    def _fetch_image(self, url: str) -> bytes | None:
        if url.startswith("file://"):
            try:
                with open(url[len("file://"):], "rb") as fh:
                    return fh.read()
            except OSError:
                return None
        for attempt in range(self.settings.indexer_max_attempts):
            try:
                resp = httpx.get(
                    url,
                    timeout=self.settings.image_fetch_timeout_seconds,
                    headers={"User-Agent": DEFAULT_UA},
                    follow_redirects=True,
                    proxy=self.settings.fetch_proxy_url or None,
                )
                if resp.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    return None
                if len(resp.content) > self.settings.image_max_bytes:
                    return None
                ctype = resp.headers.get("content-type", "")
                if ctype and not ctype.startswith("image/"):
                    return None
                return resp.content
            except httpx.HTTPError:
                time.sleep(0.5 * (attempt + 1))
        return None

    def _save_thumbnail(self, image_id: int, img: Image.Image) -> None:
        if not self.settings.store_thumbnails:
            return
        thumb = img.copy()
        thumb.thumbnail((self.settings.thumb_size, self.settings.thumb_size), Image.Resampling.LANCZOS)
        path = self.settings.thumbs_dir / f"{image_id}.jpg"
        thumb.convert("RGB").save(path, "JPEG", quality=82, optimize=True)

    # ----------------------------------------------------------- tracker loop
    # The "auto-update" feature: every tracked query image is reverse-searched
    # across the whole internet (SearXNG -> Google via proxify) on a schedule;
    # every NEW match is enqueued into the own index, so the same image finds
    # MORE matches over time — TinEye-style continuous learning.
    def _tracker_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.settings.tracker_interval_seconds)
            try:
                self.run_tracker_round()
            except Exception as exc:  # noqa: BLE001 — worker thread must survive
                self._set_error(f"tracker: {exc}")
                self.log.warning("tracker round failed", extra={"event": "tracker_error", "error": str(exc)})

    def run_tracker_round(self) -> int:
        """Reverse-search every enabled, due tracker once. Returns tracks processed."""
        tracks = [t for t in self.repo.list_tracks() if t.get("enabled")]
        now = time.time()
        interval = self.settings.tracker_interval_seconds
        processed = 0
        for track in tracks:
            if self._stop.is_set():
                break
            if now - track.get("last_run", 0) < interval:
                continue
            try:
                self._track_one(track)
                processed += 1
            except Exception as exc:  # noqa: BLE001 — one bad track never kills the round
                self.log.warning(
                    "track failed",
                    extra={"event": "track_error", "track_id": track["id"], "error": str(exc)},
                )
                try:
                    self.repo.touch_track(track["id"], 0, note=f"error: {exc}")
                except Exception:  # noqa: BLE001
                    pass
        return processed

    def _track_one(self, track: dict) -> None:
        public_url = track.get("public_url") or ""
        hits: list[ImageHit] = []
        degraded: list[str] = []
        if public_url:
            provider_names = [p.strip() for p in self.settings.reverse_providers.split(",") if p.strip()]
            hits, degraded = run_reverse_chain(
                public_url,
                provider_names,
                self.searx.reverse_search,
                self.google.search,
                self.bing.search,
                self.yandex.search,
            )
            for p in provider_names:
                self.repo.record_provider(p, ok=(f"provider:{p}" not in degraded))
        elif track.get("local_path") and (self.settings.bing_upload_enabled or self.settings.google_upload_enabled):
            # no public URL -> byte-upload the saved file (Bing Visual Search
            # POST first, then Google Lens via the proxify tunnel) so NEW
            # images can still be found once they appear online.
            from pathlib import Path as _Path

            path = _Path(track["local_path"])
            if path.exists():
                raw = path.read_bytes()
                if self.settings.bing_upload_enabled:
                    try:
                        hits = self._guarded_call("bing-upload", lambda: self.bing.search_upload(raw, filename=path.name))
                        self.repo.record_provider("bing-upload", ok=True)
                    except Exception as exc:  # noqa: BLE001 — degrade, never crash the loop
                        self.repo.record_provider("bing-upload", ok=False)
                        degraded.append("provider:bing-upload")
                        self.log.warning("track bing-upload failed", extra={"event": "track_upload_fail", "error": str(exc)})
                if not hits and self.settings.yandex_upload_enabled:
                    try:
                        hits = self._guarded_call("yandex-upload", lambda: self.yandex.search_upload(raw, filename=path.name))
                        self.repo.record_provider("yandex-upload", ok=True)
                    except Exception as exc:  # noqa: BLE001 — degrade, never crash the loop
                        self.repo.record_provider("yandex-upload", ok=False)
                        degraded.append("provider:yandex-upload")
                        self.log.warning("track yandex-upload failed", extra={"event": "track_upload_fail", "error": str(exc)})
                if not hits and self.settings.google_upload_enabled:
                    try:
                        hits = self._guarded_call("google-upload", lambda: self.google.search_upload(raw, filename=path.name))
                        self.repo.record_provider("google-upload", ok=True)
                    except Exception as exc:  # noqa: BLE001 — degrade, never crash the loop
                        self.repo.record_provider("google-upload", ok=False)
                        degraded.append("provider:google-upload")
                        self.log.warning("track google-upload failed", extra={"event": "track_upload_fail", "error": str(exc)})
            else:
                degraded.append("no_local_upload")
        else:
            degraded.append("no_public_url")
        new = 0
        # TinEye verdict: the engine only found candidates; OUR fused hashes
        # judge each one before it can enter the index (so unrelated images
        # never pollute it, and "hits" only counts confirmed matches).
        candidates = hits
        if self.settings.tracker_verify_candidates:
            # legacy tracks (pre-multi-hash) carry dhash/ahash == 0; only
            # include them when they are real so verification falls back to
            # pHash-only instead of comparing against garbage zero hashes.
            qf = {"phash": int(track.get("phash") or 0)}
            if track.get("dhash") and track.get("ahash"):
                qf["dhash"] = int(track["dhash"])
                qf["ahash"] = int(track["ahash"])
            candidates = [h for h in hits if self._hash_accepts(qf, h.img_src)]
        for hit in candidates:
            if self.repo.enqueue_pending(
                hit.img_src,
                {"page_url": hit.url, "title": hit.title, "engine": hit.engine, "query": "auto-track"},
            ):
                new += 1
        note = f"{len(candidates)}/{len(hits)} verified, {new} new"
        if degraded:
            note += f" (degraded: {','.join(degraded)})"
        self.repo.touch_track(track["id"], new, note=note)
        self._bump_progress()
        self.log.info(
            "tracked",
            extra={"event": "track", "track_id": track["id"], "hits": len(hits), "new": new, "degraded": degraded},
        )

    def _hash_accepts(self, query_features: dict, url: str) -> bool:
        """Fetch a candidate and accept it only if its fused hash distance to
        the tracked image is within the near threshold. Unfetchable candidates
        are rejected (the tracker never ingests unverified images)."""
        from app.matcher import Matcher

        try:
            raw = self._fetch_image(url)
            if raw is None:
                return False
            img = decode_image(raw)
            if img is None or min(img.size) < self.settings.image_min_dimension:
                return False
            feats = extract_features(img, raw)
            fused, dp, _, _ = Matcher._fused_from_hashes(
                query_features,
                int(feats["phash"]),
                int(feats["dhash"]),
                int(feats["ahash"]),
            )
            return fused <= self.settings.fused_near_threshold
        except Exception:  # noqa: BLE001 — a bad candidate must never crash the round
            return False

    def request_tracker_round(self) -> None:
        """Kick a tracker round in the background (never block an API request)."""
        threading.Thread(target=self.run_tracker_round, name="tracker-once", daemon=True).start()

    def request_discovery(self) -> None:
        """Kick a discovery round in the background (never block an API request)."""
        threading.Thread(target=self._run_discovery_round, name="discovery-once", daemon=True).start()

    def _bump_progress(self) -> None:
        self.status.last_progress = time.time()

    def _incr_failed(self) -> None:
        self.status.failed += 1
        try:
            self.repo.incr_stat("failed")
        except Exception:  # noqa: BLE001 — a counter failure must never kill a worker
            pass

    def _set_error(self, message: str) -> None:
        self.status.last_error = message


def local_pixels(settings: Settings, meta: ImageMeta) -> Image.Image | None:
    """Best LOCAL pixels for an index entry, without touching the network:
    the original upload when we have it (full resolution — matters for small
    faces), else the stored thumbnail."""
    from pathlib import Path

    candidates: list[Path] = []
    url = meta.url or ""
    if url.startswith("file://"):
        candidates.append(Path(url[len("file://"):]))
    if "/uploads/" in url and not _is_remote_url(url):
        candidates.append(Path(settings.uploads_dir) / url.rsplit("/uploads/", 1)[1])
    if meta.sha256:
        candidates.extend(sorted(Path(settings.uploads_dir).glob(f"{meta.sha256}.*")))
    candidates.append(settings.thumbs_dir / f"{meta.id}.jpg")
    for path in candidates:
        try:
            if path.is_file():
                img = decode_image(path.read_bytes())
                if img is not None:
                    return img
        except OSError:
            continue
    return None


def _hash_to_vector(phash: int) -> list[float]:
    from app.features import encode_hash_vector

    return encode_hash_vector(phash)
