"""Storage layer for Imify.

Two interchangeable backends behind one protocol:

- ``strikedb`` (default, production) — the user's own StrikeDB engine:
  durable KV metadata + native HNSW vector ANN over RESP.
- ``sqlite`` (dev/test fallback) — same protocol, stdlib sqlite3. Lets the
  app run and be tested with zero external services, and acts as the escape
  hatch if StrikeDB ever misbehaves in production.

Because metadata (hashes, histograms, URLs) is the durable source of truth in
KV either way, the vector ANN is always rebuildable — a restart cannot lose
the index.
"""

from __future__ import annotations

from app.config import Settings

from .base import Storage, StorageError
from .sqlite import SqliteStorage
from .strikedb import StrikeDbStorage


def get_storage(settings: Settings) -> Storage:
    if settings.is_strikedb:
        return StrikeDbStorage(
            host=settings.strikedb_host,
            port=settings.strikedb_port,
            password=settings.strikedb_pass,
        )
    return SqliteStorage(settings.sqlite_path)


__all__ = ["Storage", "StorageError", "SqliteStorage", "StrikeDbStorage", "get_storage"]
