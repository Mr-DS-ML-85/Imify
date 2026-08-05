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

    # --- Outbound proxy (optional: your proxify anti-bot gateway) ---
    # Point at proxify's HTTP CONNECT tunnel (default http://127.0.0.1:8888)
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
    fused_exact_threshold: float = 0.08
    fused_near_threshold: float = 0.22
    # Use VSEARCH ANN (fast, scalable) above this index size, exact scan below.
    exact_scan_max_images: int = 200_000
    vsearch_k: int = 128
    default_num_results: int = 40
    max_num_results: int = 200

    # --- Web candidate verification (the "TinEye judge") ---
    # Engines (SearXNG / Google) only FIND candidate URLs; every candidate is
    # downloaded and fused-hash-verified against the query image before being
    # reported as a match. Google's face limits don't matter: our own hash
    # matcher is the verdict, never the engine's. Candidates that can't be
    # fetched or don't match are still listed but flagged "unverified" so the
    # UI never presents a false positive as a confirmed site.
    web_verify_candidates: bool = True
    web_verify_concurrency: int = 6
    web_verify_timeout_seconds: float = 15.0
    web_verify_max_candidates: int = 24
    # Max proxify browser renders per search. Facebook comet photo pages are
    # JS-only (the image URL exists only in the DOM) and each render costs
    # ~10s through the single shared Chrome — capping keeps a FB-heavy result
    # set from stalling the whole search.
    web_verify_render_max: int = 4

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
