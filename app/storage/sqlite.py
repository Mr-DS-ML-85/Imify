"""SQLite backend — dev/test fallback implementing the Storage protocol.

Uses only the stdlib. The vector "ANN" is an exact brute-force cosine scan
(with a small numpy batch kernel), which is plenty for tests and for small
dev indexes. Metadata is stored in the same JSON shape as StrikeDB so the
product can switch backends without losing anything.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from .base import Storage, StorageError


class SqliteStorage(Storage):
    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # One shared connection guarded by a lock: sqlite3 connections are not
        # safe for concurrent use across threads (the pipeline runs workers in
        # threads), so every op serializes through `self._lock`.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
            self._conn.execute("CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, vec BLOB)")
            self._conn.commit()

    # -- KV -----------------------------------------------------------
    def ping(self) -> bool:
        try:
            with self._lock:
                self._conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (key, value),
            )
            self._conn.commit()

    def mget(self, keys: Iterable[str]) -> list[str | None]:
        key_list = list(keys)
        if not key_list:
            return []
        placeholders = ",".join("?" * len(key_list))
        with self._lock:
            rows = self._conn.execute(f"SELECT k, v FROM meta WHERE k IN ({placeholders})", key_list).fetchall()
        lookup = dict(rows)
        return [lookup.get(k) for k in key_list]

    def mset(self, kvs: dict[str, str]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                list(kvs.items()),
            )
            self._conn.commit()

    def delete(self, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM meta WHERE k=?", (key,))
            self._conn.commit()
            return cur.rowcount > 0

    def incr(self, key: str, by: int = 1) -> int:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(k, v) VALUES(?, 0) ON CONFLICT(k) DO NOTHING",
                (key,),
            )
            self._conn.execute(
                "UPDATE meta SET v = CAST(v AS INTEGER) + ? WHERE k=?",
                (by, key),
            )
            row = self._conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
            self._conn.commit()
            return int(row[0]) if row else 0

    def keys(self, prefix: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT k FROM meta WHERE k LIKE ?", (prefix + "%",)).fetchall()
            return [r[0] for r in rows]

    # -- Vectors -------------------------------------------------------
    def vadd(self, id: int, vector: list[float]) -> None:
        blob = struct.pack(f"<{len(vector)}f", *vector)
        with self._lock:
            self._conn.execute(
                "INSERT INTO vectors(id, vec) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET vec=excluded.vec",
                (id, blob),
            )
            self._conn.commit()

    def vdel(self, ids: list[int]) -> int:
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            cur = self._conn.execute(f"DELETE FROM vectors WHERE id IN ({placeholders})", ids)
            self._conn.commit()
            return cur.rowcount

    def vsearch(self, k: int, vector: list[float]) -> list[tuple[int, float]]:
        query = np.asarray(vector, dtype=np.float32)
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0:
            return []
        with self._lock:
            rows = self._conn.execute("SELECT id, vec FROM vectors").fetchall()
        if not rows:
            return []
        ids = np.asarray([r[0] for r in rows], dtype=np.int64)
        mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        norms = np.linalg.norm(mat, axis=1)
        sims = (mat @ query) / (norms * qnorm + 1e-9)
        dists = np.clip(1.0 - sims, 0.0, 2.0)
        top = np.argsort(dists)[:k]
        return [(int(ids[i]), float(dists[i])) for i in top]

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except sqlite3.Error:
            pass

    def get_json(self, key: str):
        raw = self.get(key)
        return json.loads(raw) if raw else None
