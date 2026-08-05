"""Standalone indexer worker — the SearchOS-style gateway/worker split.

Run the API and the pipeline in separate processes for production:

    uvicorn app.main:app            # API only (INDEXER_RUN_INLINE=false)
    python -m app.worker            # pipeline only (discovery + fetch + index)

Both share the same StrikeDB, so admin stats stay consistent across processes
(KV-backed counters).
"""

from __future__ import annotations

import signal
import threading
import time

from app.config import get_settings
from app.indexer import Indexer
from app.logging import child_logger, setup_logging
from app.repo import Repository
from app.searx import SearxClient
from app.storage import get_storage

log = child_logger("imify.worker", worker="indexer")


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_dirs()

    storage = get_storage(settings)
    if settings.is_strikedb:
        # StrikeDB may still be replaying its WAL on boot — poll briefly.
        storage_ready = False
        for _ in range(60):
            if storage.ping():
                storage_ready = True
                break
            time.sleep(1)
        if not storage_ready:
            raise RuntimeError(
                f"StrikeDB unreachable at {settings.strikedb_host}:{settings.strikedb_port}. "
                "Start it (`DBSTRIKE_WAL=./dbstrike.wal dbstrike 127.0.0.1:6008`) or set STORAGE_BACKEND=sqlite."
            )

    repo = Repository(storage)
    searx = SearxClient(
        settings.searxng_url,
        timeout=settings.searxng_timeout_seconds,
        min_interval=settings.searxng_query_interval,
        max_pages=settings.searxng_max_pages,
    )
    indexer = Indexer(repo, settings, searx)
    indexer.start()

    stop_event = threading.Event()

    def shutdown(*_):
        log.info("worker shutting down", extra={"event": "shutdown"})
        stop_event.set()
        indexer.stop()
        storage.close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log.info("worker ready", extra={"event": "start", "backend": settings.storage_backend})
    while not stop_event.wait(60):
        pass


if __name__ == "__main__":
    main()
