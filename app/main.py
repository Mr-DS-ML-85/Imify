"""Imify — FastAPI application factory.

Lifecycle: build settings -> open storage -> wire repository/matcher/indexer ->
start the continuous indexing pipeline -> serve the API + static UI.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import build_api
from app.auth import AuthService, RateLimiter
from app.config import get_settings
from app.engines.bing_images import BingImagesReverse
from app.engines.google_images import GoogleImagesReverse
from app.engines.yandex_images import YandexImagesReverse
from app.indexer import Indexer
from app.logging import child_logger, new_request_id, request_id_var, setup_logging
from app.matcher import Matcher
from app.repo import Repository
from app.searx import SearxClient
from app.storage import StorageError, get_storage

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Module-level singletons (imported by uvicorn).
settings = get_settings()
setup_logging(settings.log_level)
log = child_logger("imify")
storage = get_storage(settings)
repo = Repository(storage, thumbs_dir=settings.thumbs_dir)
searx = SearxClient(
    settings.searxng_url,
    timeout=settings.searxng_timeout_seconds,
    min_interval=settings.searxng_query_interval,
    max_pages=settings.searxng_max_pages,
)
matcher = Matcher(repo, settings)
google = GoogleImagesReverse(
    settings.proxify_fetch_url,
    fetch_proxy_url=settings.fetch_proxy_url,
    use_browser=settings.proxify_use_browser,
    direct_upload=settings.google_upload_direct,
)
bing = BingImagesReverse(
    proxify_fetch_url=settings.proxify_fetch_url,
    fetch_proxy_url=settings.fetch_proxy_url,
)
yandex = YandexImagesReverse(
    proxify_fetch_url=settings.proxify_fetch_url,
    fetch_proxy_url=settings.fetch_proxy_url,
)
indexer = Indexer(repo, settings, searx, google, bing, yandex)
auth = AuthService(repo, settings.bootstrap_admin_key)
limiter = RateLimiter(
    per_minute=settings.rate_limit_per_minute,
    per_hour=settings.rate_limit_per_hour,
    per_day=settings.rate_limit_per_day,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    if settings.is_strikedb:
        # StrikeDB may still be replaying its WAL on boot — poll briefly.
        storage_ready = False
        for _ in range(60):
            if storage.ping():
                storage_ready = True
                break
            await asyncio.sleep(1)
        if not storage_ready:
            raise RuntimeError(
                "StrikeDB is unreachable at "
                f"{settings.strikedb_host}:{settings.strikedb_port}. "
                "Start it (`DBSTRIKE_WAL=./dbstrike.wal dbstrike 127.0.0.1:6008`) or set STORAGE_BACKEND=sqlite."
            )
    if settings.indexer_enabled and settings.indexer_run_inline:
        indexer.start()
    log.info("imify ready", extra={"event": "startup", "backend": settings.storage_backend})
    yield
    indexer.stop()
    storage.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id_var.set(new_request_id())
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        log.exception("unhandled error", extra={"path": request.url.path})
        response = JSONResponse({"detail": "internal error"}, status_code=500)
    response.headers["X-Request-Id"] = request_id_var.get()
    duration_ms = int((time.monotonic() - started) * 1000)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    log.info(
        "http",
        extra={"event": "http", "method": request.method, "path": request.url.path, "status": response.status_code, "ms": duration_ms},
    )
    return response


@app.exception_handler(StorageError)
async def storage_error_handler(request: Request, exc: StorageError):
    return JSONResponse({"detail": f"storage error: {exc}"}, status_code=503)


app.include_router(
    build_api(settings, repo, matcher, indexer, auth, searx, limiter, google, bing, yandex),
    prefix="",
)

# Uploaded query images (auto-tracked): served so reverse engines can reach them.
Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.uploads_dir), name="uploads")


# ------------------------------------------------------------- thumbnails
@app.get("/img/{name}")
def thumbnail(name: str):
    safe = name.replace("..", "").replace("/", "")
    if safe.startswith("query-"):
        path = settings.thumbs_dir / f"q-{safe[len('query-'):]}.jpg"
        if path.exists():
            return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})
        raise JSONResponse({"detail": "not found"}, status_code=404)
    parts = safe.rsplit(".", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        raise JSONResponse({"detail": "not found"}, status_code=404)
    path = settings.thumbs_dir / f"{parts[0]}.jpg"
    if path.exists():
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
    raise JSONResponse({"detail": "not found"}, status_code=404)


@app.get("/")
def index():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/admin")
def admin():
    return FileResponse(
        STATIC_DIR / "admin.html",
        headers={"Cache-Control": "no-cache"},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
