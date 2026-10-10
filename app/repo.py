"""Domain repository — the index's business logic over the Storage protocol.

This is where image metadata, dedup maps, seed queries, pending work, search
history and counters live. Both backends (StrikeDB / SQLite) satisfy the
protocol, so nothing above this layer knows which database is underneath.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable

from app.models import ImageMeta
from app.storage import Storage

META = "img:meta:"
URL = "img:url:"
SHA = "img:sha:"
PHASH = "img:phash:"
REMOVED = "img:removed:"
QUERY = "query:meta:"
PENDING = "pending:"
CLAIM = "claim:"  # lease keys so concurrent workers never double-fetch
TRACK = "track:meta:"
TRACK_BY_PHASH = "track:by_phash:"
KEY = "key:"
HIST = "hist:"


class Repository:
    def __init__(self, storage: Storage, thumbs_dir: Any | None = None):
        self.db = storage
        self.thumbs_dir = thumbs_dir  # optional: where remove_image cleans up thumbnails

    # --- ids -------------------------------------------------------------
    def next_image_id(self) -> int:
        return self.db.incr("img:seq")

    def next_query_id(self) -> int:
        return self.db.incr("query:seq")

    # --- image metadata ---------------------------------------------------
    def put_image(self, meta: ImageMeta) -> None:
        self.db.set_json(f"{META}{meta.id}", meta.to_dict())
        self.db.incr("stat:images")

    def _removed(self, image_id: int) -> bool:
        return self.db.get(f"{REMOVED}{image_id}") is not None

    def get_image(self, image_id: int) -> ImageMeta | None:
        if self._removed(image_id):
            return None
        data = self.db.get_json(f"{META}{image_id}")
        return ImageMeta.from_dict(data) if data else None

    def get_images(self, ids: Iterable[int]) -> list[ImageMeta | None]:
        keys = [f"{META}{i}" for i in ids]
        rows = self.db.mget(keys)
        out: list[ImageMeta | None] = []
        for key, row in zip(keys, rows):
            image_id = int(key[len(META):])
            if self._removed(image_id) or row is None:
                out.append(None)
            else:
                try:
                    out.append(ImageMeta.from_dict(json.loads(row)))
                except json.JSONDecodeError:
                    out.append(None)
        return out

    def remove_image(self, image_id: int) -> bool:
        """Remove an image from the index (metadata + maps + vector + tombstone).

        With StrikeDB VDEL the ±1 vector is tombstoned in the HNSW graph too,
        so removed images never occupy graph slots or appear in searches (the
        old behaviour left dead vectors behind)."""
        meta = self.get_image(image_id)
        if meta is None:
            return False
        self.db.delete(f"{URL}{hash_key(meta.url)}")
        if meta.sha256:
            self.db.delete(f"{SHA}{meta.sha256}")
        self.db.delete(f"{PHASH}{meta.phash:016x}")
        self.db.delete(f"{META}{image_id}")
        self.db.set(f"{REMOVED}{image_id}", "1")
        self.db.incr("stat:removed")
        # VDEL the vector so the ANN graph frees the node (no dead slots).
        try:
            self.db.vdel([image_id])
        except Exception:  # noqa: BLE001 — a failed VDEL must never block removal
            pass
        if self.thumbs_dir is not None:
            try:
                (self.thumbs_dir / f"{image_id}.jpg").unlink(missing_ok=True)
            except OSError:  # noqa: BLE001 — a stale thumb must never block removal
                pass
        # keep the images counter accurate (guarded against drift below zero)
        if int(self.db.get("stat:images") or 0) > 0:
            self.db.incr("stat:images", -1)
        return True

    def purge_images(self, url_contains: str = "", query: str = "") -> int:
        """Bulk-remove images whose URL contains a substring or whose source
        query matches. Returns how many were removed."""
        removed = 0
        for meta in self.all_images():
            if url_contains and url_contains.lower() not in meta.url.lower():
                continue
            if query and query.lower() not in (meta.query or "").lower():
                continue
            if self.remove_image(meta.id):
                removed += 1
        return removed

    def update_image(self, meta: ImageMeta) -> None:
        self.db.set_json(f"{META}{meta.id}", meta.to_dict())

    def all_image_ids(self) -> list[int]:
        keys = self.db.keys(META)
        return sorted(int(k[len(META):]) for k in keys)

    def all_images(self) -> list[ImageMeta]:
        ids = self.all_image_ids()
        metas = self.get_images(ids)
        return [m for m in metas if m is not None]

    def image_count(self) -> int:
        return len(self.db.keys(META))

    def evict_if_full(self, max_images: int) -> int:
        """LRU eviction: if the index is over `max_images`, remove the oldest
        (least-recently-seen) images until under the cap. Returns how many were
        evicted. This is the storage guard that lets on-the-fly mode run forever
        without the DB filling up."""
        if max_images <= 0 or self.image_count() <= max_images:
            return 0
        metas = self.all_images()
        metas.sort(key=lambda m: m.last_seen)
        evicted = 0
        for meta in metas:
            if self.image_count() <= max_images:
                break
            if self.remove_image(meta.id):
                evicted += 1
        return evicted

    # --- dedup maps --------------------------------------------------------
    def url_to_id(self, url: str) -> int | None:
        raw = self.db.get(f"{URL}{hash_key(url)}")
        return int(raw) if raw else None

    def register_url(self, url: str, image_id: int) -> None:
        self.db.set(f"{URL}{hash_key(url)}", str(image_id))

    def add_source_link(self, image_id: int, url: str, page_url: str = "", title: str = "") -> bool:
        """Record a URL where the canonical image was found, WITHOUT fetching
        or storing the raw bytes again (the link is the index entry — the
        TinEye model: one canonical image + the list of sites carrying it).

        Returns True when a new source URL was added. The URL map is updated so
        repeat reverse hits of the same URL are recognized as known sources.
        """
        if not url:
            return False
        meta = self.get_image(image_id)
        if meta is None:
            return False
        if url in meta.source_urls:
            return False
        meta.source_urls.append(url)
        meta.source_count = len(meta.source_urls)
        meta.last_seen = time.time()

        def _is_remote(u: str) -> bool:
            low = (u or "").lower()
            if not (low.startswith("http://") or low.startswith("https://")):
                return False
            host = (u.split("://", 1)[1].split("/", 1)[0] or "").lower()
            return host not in {"127.0.0.1", "localhost"} and not host.startswith(("127.", "10.", "192.168."))

        # Prefer a real remote source as the canonical link/page over a stale
        # local (file:// / 127.0.0.1 loopback) entry, so the index surfaces the
        # site hosting the image — not our own disk copy.
        if _is_remote(url) and not _is_remote(meta.url):
            meta.url = url
        if page_url and page_url != meta.page_url:
            if _is_remote(page_url) or not meta.page_url:
                meta.page_url = page_url
        if title and not meta.title:
            meta.title = title
        self.update_image(meta)
        self.register_url(url, image_id)
        return True

    def sha_to_id(self, sha256: str) -> int | None:
        raw = self.db.get(f"{SHA}{sha256}")
        return int(raw) if raw else None

    def register_sha(self, sha256: str, image_id: int) -> None:
        self.db.set(f"{SHA}{sha256}", str(image_id))

    def phash_to_id(self, phash: int) -> int | None:
        raw = self.db.get(f"{PHASH}{phash:016x}")
        return int(raw) if raw else None

    def register_phash(self, phash: int, image_id: int) -> None:
        self.db.set(f"{PHASH}{phash:016x}", str(image_id))

    # --- seed queries ------------------------------------------------------
    def put_query(self, query: str, enabled: bool = True, interval_seconds: int = 0) -> int:
        qid = self.next_query_id()
        now = time.time()
        self.db.set_json(
            f"{QUERY}{qid}",
            {
                "id": qid,
                "query": query,
                "enabled": enabled,
                "interval_seconds": interval_seconds,
                "created": now,
                "last_run": 0.0,
                "last_count": 0,
                "total": 0,
            },
        )
        return qid

    def list_queries(self) -> list[dict[str, Any]]:
        keys = self.db.keys(QUERY)
        rows = self.db.mget(keys)
        out = []
        for key, row in zip(keys, rows):
            if row:
                data = json.loads(row)
                data["id"] = int(key[len(QUERY):])
                out.append(data)
        return sorted(out, key=lambda q: q["id"])

    def get_query(self, qid: int) -> dict[str, Any] | None:
        return self.db.get_json(f"{QUERY}{qid}")

    def set_query_enabled(self, qid: int, enabled: bool) -> None:
        data = self.db.get_json(f"{QUERY}{qid}")
        if data:
            data["enabled"] = enabled
            self.db.set_json(f"{QUERY}{qid}", data)

    def delete_query(self, qid: int) -> None:
        self.db.delete(f"{QUERY}{qid}")

    def touch_query(self, qid: int, count: int) -> None:
        data = self.db.get_json(f"{QUERY}{qid}")
        if data:
            data["last_run"] = time.time()
            data["last_count"] = count
            data["total"] = int(data.get("total", 0)) + count
            self.db.set_json(f"{QUERY}{qid}", data)

    # --- pending fetch queue (resumable) -----------------------------------
    def enqueue_pending(self, url: str, meta: dict[str, Any]) -> bool:
        """Returns False when this URL is already pending or indexed."""
        if self.url_to_id(url) is not None:
            return False
        key = f"{PENDING}{hash_key(url)}"
        if self.db.get(key) is not None:
            return False
        self.db.set_json(key, {"url": url, "added": time.time(), "attempts": 0, **meta})
        return True

    def claim_pending(self, limit: int = 16) -> list[dict[str, Any]]:
        """Atomically claim up to `limit` pending tasks.

        Each task gets a lease key (INCR: the first worker to reach 1 owns the
        task). The task key itself stays put until finish_pending — a crash
        cannot lose the URL, and two workers can never process the same image
        concurrently (this was a real duplicate-index bug).
        """
        keys = self.db.keys(PENDING)[: limit * 4]
        claimed: list[dict[str, Any]] = []
        for key in keys:
            if len(claimed) >= limit:
                break
            urlsha = key[len(PENDING):]
            claim_key = f"{CLAIM}{urlsha}"
            if self.db.incr(claim_key) != 1:  # lease already held
                continue
            row = self.db.get(key)
            if row is None:
                self.db.delete(claim_key)
                continue
            claimed.append({"key": key, "claim_key": claim_key, "data": json.loads(row)})
        return claimed

    def finish_pending(self, claimed: dict[str, Any], done: bool) -> None:
        """Release a claim. done=True drops the task; False keeps it for retry."""
        if done:
            self.db.delete(claimed["key"])
        else:
            claimed["data"]["attempts"] = int(claimed["data"].get("attempts", 0)) + 1
            self.db.set_json(claimed["key"], claimed["data"])
        self.db.delete(claimed["claim_key"])

    def pending_count(self) -> int:
        return len(self.db.keys(PENDING))

    # --- counters / stats ---------------------------------------------------
    def incr_stat(self, name: str, by: int = 1) -> int:
        return self.db.incr(f"stat:{name}", by)

    def get_stat(self, name: str) -> int:
        raw = self.db.get(f"stat:{name}")
        return int(raw) if raw else 0

    def record_search(self, found: bool) -> None:
        """Success-rate counter: how many searches found at least one match."""
        self.db.incr("stat:search_found" if found else "stat:search_notfound")

    def record_provider(self, provider: str, ok: bool) -> None:
        """Per-provider success/failure counter (searxng, google, google-upload)."""
        self.db.incr(f"stat:prov:{provider}:{'ok' if ok else 'fail'}")

    def search_rate(self) -> dict:
        found = self.get_stat("search_found")
        notfound = self.get_stat("search_notfound")
        total = found + notfound
        return {"total": total, "found": found, "not_found": notfound, "rate": round(found / total, 6) if total else None}

    def provider_stats(self) -> dict:
        out: dict[str, dict] = {}
        for name in ("searxng", "bing", "bing-upload", "yandex", "yandex-upload", "google", "google-upload"):
            ok = self.get_stat(f"prov:{name}:ok")
            fail = self.get_stat(f"prov:{name}:fail")
            total = ok + fail
            out[name] = {"ok": ok, "fail": fail, "total": total, "rate": round(ok / total, 6) if total else None}
        return out

    def bump_fetch(self, image_id: int) -> None:
        meta = self.get_image(image_id)
        if meta:
            meta.fetch_count += 1
            meta.last_seen = time.time()
            self.update_image(meta)

    # --- tracking (background reverse-search of a query image) -------------------
    def upsert_track(
        self,
        sha256: str,
        phash: int,
        public_url: str = "",
        title: str = "",
        local_path: str = "",
        dhash: int = 0,
        ahash: int = 0,
    ) -> int:
        """Create a tracker for a query image, or refresh an existing one.
        Returns the track id.

        ``public_url`` is the publicly reachable URL the background tracker
        will hand to reverse-search engines (SearXNG / Google via proxify).
        ``local_path`` is the on-disk upload (used for byte-upload reverse
        when no public URL exists). ``dhash``/``ahash`` complete the fused
        fingerprint so candidate matches are hash-verified against the tracked
        image before being ingested (never trust the engine's verdict)."""
        existing = self.db.get(f"{TRACK_BY_PHASH}{phash:016x}")
        if existing:
            track_id = int(existing)
            data = self.db.get_json(f"{TRACK}{track_id}")
            if data:
                if public_url:
                    data["public_url"] = public_url
                if title:
                    data["title"] = title
                if local_path:
                    data["local_path"] = local_path
                if dhash:
                    data["dhash"] = dhash
                if ahash:
                    data["ahash"] = ahash
                self.db.set_json(f"{TRACK}{track_id}", data)
            return track_id
        track_id = self.db.incr("track:seq")
        now = time.time()
        self.db.set_json(
            f"{TRACK}{track_id}",
            {
                "id": track_id,
                "sha256": sha256,
                "phash": phash,
                "dhash": dhash,
                "ahash": ahash,
                "public_url": public_url,
                "local_path": local_path,
                "title": title,
                "created": now,
                "last_run": 0.0,
                "hits": 0,
                "enabled": True,
                "note": "",
            },
        )
        self.db.set(f"{TRACK_BY_PHASH}{phash:016x}", str(track_id))
        return track_id

    def list_tracks(self) -> list[dict[str, Any]]:
        keys = self.db.keys(TRACK)
        rows = self.db.mget(keys)
        out = []
        for key, row in zip(keys, rows):
            if row:
                data = json.loads(row)
                data["id"] = int(key[len(TRACK):])
                data["image_id"] = self.phash_to_id(int(data.get("phash", 0)))
                out.append(data)
        return sorted(out, key=lambda t: t["id"])

    def get_track(self, track_id: int) -> dict[str, Any] | None:
        data = self.db.get_json(f"{TRACK}{track_id}")
        if data:
            data["id"] = track_id
            data["image_id"] = self.phash_to_id(int(data.get("phash", 0)))
        return data

    def set_track_enabled(self, track_id: int, enabled: bool) -> None:
        data = self.db.get_json(f"{TRACK}{track_id}")
        if data:
            data["enabled"] = enabled
            self.db.set_json(f"{TRACK}{track_id}", data)

    def delete_track(self, track_id: int) -> bool:
        data = self.db.get_json(f"{TRACK}{track_id}")
        if not data:
            return False
        self.db.delete(f"{TRACK}{track_id}")
        self.db.delete(f"{TRACK_BY_PHASH}{int(data.get('phash', 0)):016x}")
        return True

    def touch_track(self, track_id: int, hits: int, note: str = "") -> None:
        data = self.db.get_json(f"{TRACK}{track_id}")
        if data:
            data["last_run"] = time.time()
            data["hits"] = int(data.get("hits", 0)) + hits
            if note:
                data["note"] = note
            self.db.set_json(f"{TRACK}{track_id}", data)

    # --- verified web results cache (real-time repeat searches) ---------------
    # Keyed by query-image sha256 so a repeat search of the SAME image is served
    # from memory instead of re-hitting SearXNG/Bing/Google/Yandex. Verified
    # matches only — unverified candidates are never cached (they are not "where
    # this image lives"). TTL keeps the cache from going stale forever.
    WEB_CACHE = "wb2:"

    def get_web_results(self, sha256: str, ttl_seconds: int) -> list[dict] | None:
        if not sha256:
            return None
        raw = self.db.get_json(f"{self.WEB_CACHE}{sha256}")
        if not raw:
            return None
        # The stored TTL (written at cache time) wins — unverified outcomes are
        # cached for a shorter window than verified matches. ttl <= 0 means
        # instantly stale (never serve it).
        stored_ttl = float(raw.get("ttl", ttl_seconds))
        if stored_ttl <= 0 or time.time() - float(raw.get("ts", 0)) > stored_ttl:
            self.db.delete(f"{self.WEB_CACHE}{sha256}")
            return None
        results = raw.get("results")
        return results if isinstance(results, list) else None

    def set_web_results(self, sha256: str, results: list[dict], ttl_seconds: int) -> None:
        if not sha256:
            return
        self.db.set_json(
            f"{self.WEB_CACHE}{sha256}",
            {"ts": time.time(), "ttl": float(ttl_seconds), "results": results},
        )

    # --- search history -------------------------------------------------------
    def push_history(self, entry: dict[str, Any]) -> None:
        self.db.set_json(f"{HIST}{int(time.time() * 1000)}", entry)
        # keep history bounded
        keys = sorted(self.db.keys(HIST))
        for old in keys[:-100]:
            self.db.delete(old)

    def list_history(self, limit: int = 20) -> list[dict[str, Any]]:
        keys = sorted(self.db.keys(HIST))[-limit:]
        rows = self.db.mget(keys)
        out = []
        for row in rows:
            if row:
                try:
                    out.append(json.loads(row))
                except json.JSONDecodeError:
                    pass
        return out[::-1]


def hash_key(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
