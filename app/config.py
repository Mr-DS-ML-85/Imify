"""Environment-driven configuration for Imify.

Everything the app does is configured here so a production deployment is
just a `.env` file + `uv run uvicorn`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=APP_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "Imify"
    env: str = "production"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    # Public base URL of THIS instance. When set, uploaded images are served
    # from a public endpoint so SearXNG engines (e.g. Bing) can reverse-search
    # them. Leave empty on a private/LAN box — the live web layer degrades
    # gracefully and the own index still answers.
    public_base_url: str = ""

    # --- Optional AI layer (OpenAI-compatible: llama.cpp / NVIDIA NIM / Ollama) ---
    # Leave AI_BASE_URL empty to disable AI entirely (pure perceptual pipeline,
    # zero external calls). When set, Imify auto-captions indexed images and can
    # match natural-language queries by embedding similarity.
    #
    #   llama.cpp embed server:  http://127.0.0.1:8374/v1   (set via AI_OLLAMA_URL
    #                             or AI_BASE_URL — both end in /v1)
    #   NVIDIA NIM:              https://integrate.api.nvidia.com/v1 (needs AI_API_KEY)
    #   Ollama:                  http://127.0.0.1:11434/v1
    ai_ollama_url: str = ""
    # Normalized base URL (same as AI_OLLAMA_URL; /v1 appended if missing).
    ai_base_url: str = ""
    # Hosted vision endpoint (NVIDIA NIM) — separate from the local embed
    # server. Leave empty to use AI_BASE_URL for captions too.
    ai_caption_url: str = ""
    ai_api_key: str = ""
    ai_caption_model: str = "meta/llama-3.2-90b-vision-instruct"
    ai_embed_model: str = "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
    # auto-caption indexed images in the pipeline (adds human-readable alt-text
    # + enables text search over captions). Rate-limited per-image in the worker.
    ai_caption_on_index: bool = False
    # Max seconds a single AI call (caption or embed) may take — never stalls
    # the pipeline.
    ai_caption_timeout_seconds: float = 45.0
    # By-description (/v1/search/text) similarity floor: caption embeddings
    # below this cosine to the query are dropped, so a query never returns
    # every indexed caption that merely has a nonzero overlap. Tune to your
    # embed model (nomic-embed text captions sit ~0.35-0.60 for the whole
    # library, so 0.50 keeps only the genuinely close descriptions).
    ai_text_min_sim: float = 0.50

    # --- Optional OCR layer (Tesseract, self-hosted) ---
    # When the `pytesseract` package + a local `tesseract` binary exist, Imify
    # extracts visible text from indexed images (logos, memes, screenshots)
    # and can match a text query against that OCR text. Disabled by default —
    # pure perceptual pipeline with zero native deps.
    ocr_enabled: bool = False
    ocr_lang: str = "eng"

    # --- Optional Face-ID layer (ONNX Runtime CPU, self-hosted) ---
    # Face detection (YuNet) + face embedding (ArcFace MobileFaceNet) run
    # locally on CPU (plain `onnxruntime`). Imify deliberately never starts a
    # CUDA session — a GPU driver change taking CUDA down crashed the whole
    # search previously. Extracts an
    # identity embedding from detected faces so /v1/search/face matches the
    # SAME PERSON across photos ("found this face on N sites"), facecheck.io
    # style, with face-specific confidence tiers.
    face_id_enabled: bool = False
    face_id_device: str = "cpu"  # reserved; Imify always runs Face-ID on CPU
    # Directory where the ONNX models are downloaded on first use.
    face_id_models_dir: str = str(APP_ROOT / "data" / "models" / "face")
    # Min face side in pixels (in the original image) to embed a face —
    # ArcFace on tinier crops is unreliable.
    face_id_min_face_size: int = 36
    # YuNet detection confidence floor (lower = more false "faces" on
    # textures/logos).
    face_id_score_threshold: float = 0.75
    # Max faces extracted per image (a group photo can have many).
    face_id_max_faces: int = 5
    # ArcFace cosine similarity to call two faces the SAME person. Measured on
    # this box with correct alignment: same person >= 0.88, different people
    # <= 0.30 (median ~0.01). 0.42 leaves margin for age/pose/lighting
    # changes (literature EER for MBF sits around 0.3-0.4).
    face_id_same_threshold: float = 0.42
    # FaceCheck-style recall boost: when the query shows a person, ALSO
    # upload a tight crop of the main face to Yandex (people search there is
    # face-driven; a crop of a group/holiday photo finds the person, the full
    # frame finds the scene). Candidates are face-verified either way.
    face_crop_upload: bool = True

    # --- Optional CLIP content layer (ONNX CPU, self-hosted) ---
    # CLIP ViT-B/32 embeddings capture WHAT an image shows (semantic content),
    # not just its pixels. Perceptual hashes separate copies of the SAME image
    # but cannot tell a busy homelab scene from an unrelated black-dog photo;
    # CLIP embeddings can. When enabled, every indexed image gets a 512-dim
    # CLIP embedding and /v1/search/clip ranks the library by semantic
    # similarity (the "what else is like this" signal). Runs on CPU via
    # onnxruntime (small ViT-B/32 int8 model, ~90MB, ~40ms/image).
    clip_enabled: bool = False
    clip_models_dir: str = str(APP_ROOT / "data" / "models" / "clip")
    # Threshold on CLIP embedding cosine similarity (0-1) to call two images
    # semantically the SAME content ("this is the same thing"). ~0.80 is a
    # sensible ViT-B/32 default for duplicate/similar scenes.
    clip_same_threshold: float = 0.90
    # CLIP floor for web-candidate VERIFICATION: a candidate that hash-matches
    # (fused <= fused_near) but whose CLIP embedding is below this similarity
    # to the query is semantically DIFFERENT content (two university logos, a
    # homelab vs a dog/pool photo) — downgrade it to unverified instead of
    # surfacing it as a "near" match. 0 disables the gate. A real copy of the
    # query is >= 0.90; two different logos of the same style ~0.7-0.8.
    clip_verify_min_sim: float = 0.80
    # Google-Lens-style "visually similar": a candidate that is NOT a copy
    # and NOT the same face, but whose CLIP embedding is at least this close,
    # is shown in a separate "similar" section (never counted as found).
    clip_similar_min_sim: float = 0.78

    # --- Geometric verification (TinEye-style crops / edits) ---
    # Hashes break on crops, borders, overlays and heavy edits. Candidates
    # the hashes reject are checked with SIFT keypoints + RANSAC homography:
    # enough geometrically consistent matches = the same picture, modified.
    # Measured on this box: real copies give 58-330 RANSAC inliers, unrelated
    # pairs up to ~30 (always with CLIP < 0.62). So >= strong inliers is
    # accepted alone; the medium band also needs CLIP >= clip_verify_min_sim.
    geometric_verify_enabled: bool = True
    geometric_strong_inliers: int = 40
    geometric_min_inliers: int = 18
    geometric_min_inlier_ratio: float = 0.40
    # Only spend SIFT time on candidates whose CLIP similarity is at least
    # this (when CLIP is live) — an unrelated picture cannot be a crop.
    geometric_clip_prefilter: float = 0.60
    # CLIP floor for consolidation: two result copies whose pHashes differ by
    # MORE than search_dedup_threshold Hamming bits (re-encoded/resized
    # re-uploads) but whose CLIP embeddings are at least this similar are the
    # SAME semantic content — merge them into one card and take the best
    # confidence. 0 disables (hash-only merge). Used only when CLIP is live.
    search_dedup_clip_threshold: float = 0.90

    # --- Blur restore (enhance) ---
    # Real-ESRGAN x4 + BiRefNet upscale/refine for blurry uploads. CPU fp16 is
    # slow (~48s for a 560x710 -> 2240x2832), so it's off unless enabled.
    enhancer_enabled: bool = False
    enhance_models_dir: str = str(APP_ROOT / "data" / "models" / "enhance")
    # Laplacian-variance blur ceiling: a query below this is "blurry" and gets
    # restored before hashing/web-upload. Higher = restore more aggressively.
    enhance_blur_limit: float = 90.0
    # Real-ESRGAN restores only LOW-resolution queries (width*height at or
    # below this); higher-resolution images are left as they are.
    enhance_max_pixels: int = 250_000  # ~500x500; x4plus costs ~110 s per input MP on CPU
    # Apply the enhanced pixels to the web engines' uploaded copy too (sharper
    # query for the reverse engines). Costs the enhance time on the hot path.
    enhance_before_web_upload: bool = False
    # Where ESRGAN and BiRefNet run: "auto" places each on the GPU when its
    # VRAM need fits the free VRAM (bigger model first), else in system RAM;
    # "cpu" / "gpu" force both. Needs a CUDA build of onnxruntime for the GPU;
    # a failing GPU run falls back to RAM. VRAM needs are estimates in MB.
    enhance_device: str = "auto"
    enhance_esrgan_vram_mb: int = 1500
    enhance_birefnet_vram_mb: int = 4500

    # --- Storage ---
    # "strikedb" (default) or "sqlite" (dev/test fallback).
    storage_backend: str = "strikedb"
    strikedb_host: str = "127.0.0.1"
    # Port 6008 — deliberately off the common 6379/6380 range so Imify never
    # collides with another Redis-family process on the same box.
    strikedb_port: int = 6008
    strikedb_pass: str = ""
    sqlite_path: str = str(APP_ROOT / "data" / "imify.db")

    # --- SearXNG ---
    searxng_url: str = "http://127.0.0.1:8080"
    searxng_timeout_seconds: float = 12.0
    # Minimum seconds between two SearXNG queries (politeness/rate-limit guard).
    searxng_query_interval: float = 5.0
    searxng_max_pages: int = 3
    # SearchOS-style fallback ladder (TinEye model):
    #   local index answers INSTANTLY, but the web layer ALWAYS runs too
    #   (bounded + cached) so known images still surface remote links and
    #   similar candidates — never just the local copy. 0 = never skip web
    #   because a local hit exists. >0 = skip web when local has >= this many.
    web_fallback_local_min_hits: int = 0
    web_reverse_enabled: bool = True
    # Hard cap on the live-web phase (providers + candidate verification) so a
    # slow engine can never stall the whole search — the own index answers in
    # sub-ms, web results fill in up to this budget, then the response returns.
    web_phase_budget_seconds: float = 18.0
    # Run all configured reverse providers in PARALLEL instead of one after
    # another (real-time requirement). Providers still report degraded
    # individually; hits are merged and deduped before verification.
    web_providers_parallel: bool = True
    # Verified web matches are cached per query-image hash for this long, so a
    # repeat search of the same image is served from memory (real-time) without
    # re-hitting SearXNG/Bing/Google/Yandex. TinEye-like: "I already know where
    # this image lives" instead of a fresh crawl every time.
    web_results_cache_ttl_seconds: int = 86400  # 24h

    # --- Outbound proxy (optional: your proxify anti-bot gateway) ---    # Point at proxify's HTTP CONNECT tunnel (default http://127.0.0.1:8888)
    # when SearXNG engines or image hosts block the datacenter IP. Applied to
    # image downloads and image_url fetches. httpx also honors HTTP(S)_PROXY.
    fetch_proxy_url: str = ""
    # proxify's REST /fetch endpoint (http://<host>:8080/fetch). Used by the
    # Google Images reverse provider to fetch with a real browser session.
    # NOTE: proxify's default API port is 8080 — if SearXNG already owns 8080
    # on this box, start proxify with API_PORT=8081 and set it here.
    proxify_fetch_url: str = ""
    # Netscape cookie file with the real user session (proxify exports it via
    # scripts/brave_cookies.py). Candidate verification attaches cookies for
    # the target host so hotlink-protected CDNs (lookaside.fbsbx.com, fbcdn)
    # resolve instead of returning login stubs. Empty = no cookies attached.
    cookie_file: str = str(APP_ROOT / "gui-cookies.txt")
    # Force proxify's real-browser path (gui_chrome) for Google — the strategy
    # that actually beats Google's sorry/captcha walls. Slower but reliable.
    proxify_use_browser: bool = True
    # Byte-upload reverse search: for images with NO public URL, the raw bytes
    # are POSTed to engines that accept them — "circle-to-search" style.
    # Bing Visual Search POST (reliable, works for private-box images) is tried
    # first, then Google Lens via proxify (captcha-prone).
    google_upload_enabled: bool = True
    google_upload_direct: bool = True
    bing_upload_enabled: bool = True
    # Yandex Images: the least-censored engine — returns face/people matches
    # that Google suppresses. Byte-upload ``rpt=imageview``.
    yandex_upload_enabled: bool = True
    # Live reverse providers, in order: "searxng" and/or "bing" and/or
    # "yandex" and/or "google".
    reverse_providers: str = "searxng,bing,yandex,google"
    # --- Text-guided discovery (Google-Lens "best guess") ---
    # After the reverse engines answer, their result titles yield a "best
    # guess" query ("python logo"); it is searched on SearXNG's image/icon
    # engines (blocked ones retried in proxify's real browser) and every
    # candidate goes through the same verifier. Finds copies no reverse
    # index has seen.
    text_discovery_enabled: bool = True
    # Comma-separated SearXNG engine names; empty = all web + general image
    # engines from app/engines/text_images.py (icon engines are added
    # automatically for logo/icon-like queries — see text_discovery_icons).
    text_discovery_engines: str = ""
    # "auto" (logo/icon-like queries only) | "always" | "never"
    text_discovery_icons: str = "auto"
    text_discovery_budget_seconds: float = 6.0
    # Max blocked engines retried through proxify's browser per search.
    text_discovery_browser_max: int = 2

    # --- On-the-fly matching / storage bounds ---
    # By default searches return hash-verified web matches WITHOUT persisting
    # them (learn=false). The index grows only via explicit ingestion, seed
    # discovery, or learn=true — so storage never fills from casual searches.
    # Hard cap on indexed images: when the index is full, the least-recently-
    # seen image is evicted to make room (LRU), so the DB can never run out.
    max_index_images: int = 50_000

    # --- Indexing pipeline ---
    indexer_enabled: bool = True
    # Run the pipeline threads inside the API process (single-node default).
    # Set false when running `python -m app.worker` as a separate process.
    indexer_run_inline: bool = True
    indexer_fetch_workers: int = 4
    indexer_poll_interval_seconds: float = 2.0
    indexer_max_attempts: int = 3
    # Crawler / freshness
    crawler_interval_seconds: int = 3600
    crawler_recrawl_after_seconds: int = 604800  # 7 days
    # Image fetch guards
    image_fetch_timeout_seconds: float = 15.0
    image_max_bytes: int = 20 * 1024 * 1024
    image_min_dimension: int = 16
    thumb_size: int = 256
    store_thumbnails: bool = True
    data_dir: str = str(APP_ROOT / "data")

    # --- Matching ---
    # pHash hamming thresholds (0-64). <= exact is a "match", <= near is a
    # "near match" (TinEye's "most changed" band).
    hamming_exact_threshold: int = 4
    hamming_near_threshold: int = 16
    # Multi-hash fused distance (0-1): 0.5*pHash + 0.3*dHash + 0.2*aHash.
    # Fusion is far more precise than any single hash — crops, watermarks and
    # re-encodes of the same image stay close while unrelated images separate
    # (Google limits face results; our own matcher does not).
    fused_exact_threshold: float = 0.14
    fused_near_threshold: float = 0.35
    # Rotated / mirrored copies (phone photos turned 90°, flipped re-uploads)
    # are matched by hashing the query in all 8 orientations. A real rotated
    # copy scores ~0; a non-identity orientation must be at least this tight
    # (trying 8 orientations multiplies chance collisions on flat images).
    rotation_variant_max_fused: float = 0.20
    exact_scan_max_images: int = 200_000
    vsearch_k: int = 128
    default_num_results: int = 40
    max_num_results: int = 200

    # Hamming threshold (bits) for consolidating results whose ACTUAL verified
    # content is the same photo (different re-encodes / sites) into one card
    # with a growing "found on N sites" source list. 0 => disable.
    search_dedup_threshold: int = 8

    # --- Web candidate verification (the "TinEye judge") ---
    # Engines (SearXNG / Google) only FIND candidate URLs; every candidate is
    # downloaded and fused-hash-verified against the query image before being
    # reported as a match. Google's face limits don't matter: our own hash
    # matcher is the verdict, never the engine's. Candidates that can't be
    # fetched or don't match are still listed but flagged "unverified" so the
    # UI never presents a false positive as a confirmed site.
    web_verify_candidates: bool = True
    web_verify_concurrency: int = 4
    web_verify_timeout_seconds: float = 15.0
    web_verify_max_candidates: int = 40
    # How "strong" an unverified candidate must be to be shown at all. Engines
    # (esp. Bing visual-search) return visually-similar PREVIEWS that are NOT
    # the same image (Δ can be animals/girls for a company logo). fused
    # distance 0-1; only candidates with fused <= this threshold are worth
    # surfacing as "similar" — the rest is noise the reference RIS sites
    # (reverseimagesearch / Reversely) never show. 0 disables web unverified
    # rows entirely (near-duplicate-only results).
    web_unverified_max_fused: float = 0.36
    # Per-engine quality bias: when the SAME candidate could satisfy multiple
    # engines, engines with a history of junk are de-ranked so Bing previews
    # stop drowning logo matches. Float weight multiplied into fused distance
    # (higher = penalized). 1.0 = neutral.
    web_engine_bing_penalty: float = 1.35
    # Max proxify browser renders per search. Facebook comet photo pages are
    # JS-only (the image URL exists only in the DOM) and each render costs
    # ~10s through the single shared Chrome — capping keeps a FB-heavy result
    # set from stalling the whole search.
    web_verify_render_max: int = 4
    # Candidate downloads refused directly (403 hotlink protection, 429,
    # anti-bot fronts) are retried through proxify's /fetch raw-bytes mode.
    # Max such retries per search (each costs a proxify round-trip).
    web_verify_proxify_max: int = 12
    # --- Bandwidth guards (verification downloads) ---
    # Triage every candidate on the engine's THUMBNAIL (~10 KB) first and
    # download the full image only when the thumbnail is promising (hash
    # close, CLIP close, or a face when the query has one). Most engine junk
    # is rejected without ever fetching the full-size file.
    web_verify_thumbnail_triage: bool = True
    # Downloaded candidate bytes are cached per URL (on disk, under
    # uploads/web-preview) so the same candidate is never fetched twice
    # within this window — repeat searches and the tracker reuse it.
    web_candidate_cache_ttl_seconds: int = 7 * 86400
    # Cap on cached candidate previews (oldest pruned first).
    web_preview_max_files: int = 20000
    # Facebook photo pages need a cookie + browser-render chain that breaks
    # whenever FB changes its markup or the session cookies expire, and FB's
    # ToS forbids scraping. Off = FB candidates are only used when a plain
    # fetch already returns the image.
    facebook_resolution_enabled: bool = True

    # --- Persistent tracking / auto-update ("search the whole internet in bg") --
    # When a search uploads a new image it is indexed AND a background tracker
    # keeps reverse-searching it (SearXNG -> Google via proxify) and ingests
    # every new match into the index, so the same image finds MORE matches over
    # time — TinEye-style continuous learning.
    auto_track_searches: bool = True
    tracker_interval_seconds: int = 3600
    uploads_dir: str = str(APP_ROOT / "data" / "uploads")
    # Tracker candidates are hash-verified against the tracked image before
    # being ingested, so unrelated images never pollute the index.
    tracker_verify_candidates: bool = True

    # --- Auth & rate limits ---
    # "open" — public endpoints need no key, admin endpoints need an admin key.
    # "key"  — every /v1/* endpoint needs a valid API key.
    auth_mode: str = "open"
    bootstrap_admin_key: str = ""
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 600
    rate_limit_per_day: int = 5000

    # --- Observability ---
    prometheus_enabled: bool = True

    @property
    def thumbs_dir(self) -> Path:
        return Path(self.data_dir) / "thumbs"

    @property
    def is_strikedb(self) -> bool:
        return self.storage_backend.lower() == "strikedb"

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
