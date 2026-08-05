"""API-key auth and rate limiting.

Keys are stored as SHA-256 hashes (never plaintext) in the database. The raw
key is shown exactly once at creation. Rate limiting uses a fixed-window
in-memory limiter — fine for a single-node self-hosted deployment, and it
needs no TTL support from the storage engine.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time

from app.repo import Repository

KEY_PREFIX = "key:"


class AuthService:
    def __init__(self, repo: Repository, bootstrap_admin_key: str = ""):
        self.repo = repo
        if bootstrap_admin_key:
            self.ensure_key(bootstrap_admin_key, name="bootstrap-admin", role="admin")

    @staticmethod
    def _hash(raw: str) -> str:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def ensure_key(self, raw: str, name: str, role: str = "user") -> None:
        key = self._hash(raw)
        if self.repo.db.get(f"{KEY_PREFIX}{key}") is None:
            self.repo.db.set_json(
                f"{KEY_PREFIX}{key}",
                {"name": name, "role": role, "created": time.time(), "last_used": 0.0},
            )

    def create_key(self, name: str, role: str = "user") -> str:
        raw = f"sk_{secrets.token_urlsafe(32)}"
        self.repo.db.set_json(
            f"{KEY_PREFIX}{self._hash(raw)}",
            {"name": name, "role": role, "created": time.time(), "last_used": 0.0},
        )
        return raw

    def verify(self, raw: str) -> dict | None:
        record = self.repo.db.get_json(f"{KEY_PREFIX}{self._hash(raw)}")
        if record:
            record["last_used"] = time.time()
            self.repo.db.set_json(f"{KEY_PREFIX}{self._hash(raw)}", record)
        return record

    def list_keys(self) -> list[dict]:
        out = []
        for key in self.repo.db.keys(KEY_PREFIX):
            record = self.repo.db.get_json(key)
            if record:
                out.append({**record, "key_id": key[len(KEY_PREFIX):][:8]})
        return sorted(out, key=lambda k: k["created"])

    def delete_key(self, name: str) -> bool:
        for key in self.repo.db.keys(KEY_PREFIX):
            record = self.repo.db.get_json(key)
            if record and record.get("name") == name:
                return self.repo.db.delete(key)
        return False


class RateLimiter:
    def __init__(self, per_minute: int = 60, per_hour: int = 600, per_day: int = 5000):
        self.per_minute = per_minute
        self.per_hour = per_hour
        self.per_day = per_day
        self._lock = threading.Lock()
        self._windows: dict[tuple[str, int], tuple[int, int]] = {}

    def check(self, scope: str) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        now = int(time.time())
        with self._lock:
            for bucket, limit in ((60, self.per_minute), (3600, self.per_hour), (86400, self.per_day)):
                window_start = now - (now % bucket)
                key = (scope, bucket)
                cur_start, count = self._windows.get(key, (window_start, 0))
                if cur_start != window_start:
                    cur_start, count = window_start, 0
                if count >= limit:
                    return False, max(1, cur_start + bucket - now)
                self._windows[key] = (cur_start, count + 1)
            # prune old windows
            if len(self._windows) > 10_000:
                self._windows = {k: v for k, v in self._windows.items() if v[0] >= now - 86400}
        return True, 0
