"""StrikeDB backend — the production storage for Imify.

Key namespaces (all under one engine):

- ``img:meta:<u64id>``   → JSON ImageMeta        (source of truth)
- ``img:url:<sha256>``   → id                     (URL dedup)
- ``img:sha:<sha256>``   → id                     (byte-identical dedup)
- ``img:phash:<hex64>``  → id                     (exact perceptual dup)
- ``query:meta:<n>``     → JSON seed-query state
- ``pending:<sha256>``   → JSON fetch task        (resumable queue)
- ``key:<sha256>``       → JSON API key record
- ``hist:<ts>``          → JSON search history entry
- ``stat:*`` / ``img:seq`` — counters

Vectors: pHash lives in the flat 64-dim graph (each image's pHash is stored
as 64 floats of ±1 via ``VADD`` so ``VSEARCH`` returns Hamming-ordered
near-duplicates, cosine_dist = hamming/32). Face (512-dim) and CLIP (512-dim)
embeddings live in their own StrikeDB namespaces via ``VADDNS``/``VSEARCHNS``
so they never collide with the 64-dim graph. Color histograms ride along in
the metadata JSON and are matched with an exact in-app scan.
"""

from __future__ import annotations

from typing import Iterable

from .base import Storage, StorageError
from .resp import RespClient


class StrikeDbStorage(Storage):
    def __init__(self, host: str = "127.0.0.1", port: int = 6008, password: str = ""):
        self._client = RespClient(host, port, password)

    def ping(self) -> bool:
        return self._client.ping()

    def get(self, key: str) -> str | None:
        raw = self._client.get(key)
        return raw.decode("utf-8", "replace") if raw is not None else None

    def set(self, key: str, value: str) -> None:
        self._client.set(key, value)

    def mget(self, keys: Iterable[str]) -> list[str | None]:
        out: list[str | None] = []
        for chunk in _chunks(list(keys), 200):
            replies = self._client.mget(chunk)
            out.extend(None if r is None else r.decode("utf-8", "replace") for r in replies)
        return out

    def mset(self, kvs: dict[str, str]) -> None:
        for chunk in _chunks(list(kvs.items()), 200):
            self._client.mset(dict(chunk))

    def delete(self, key: str) -> bool:
        return self._client.delete(key)

    def incr(self, key: str, by: int = 1) -> int:
        return self._client.incr(key, by)

    def keys(self, prefix: str) -> list[str]:
        return self._client.keys(prefix)

    def vadd(self, id: int, vector: list[float]) -> None:
        self._client.vadd(id, vector)

    def vdel(self, ids: list[int]) -> int:
        return self._client.vdel(ids)

    def vsearch(self, k: int, vector: list[float]) -> list[tuple[int, float]]:
        return self._client.vsearch(k, vector)

    def vadd_ns(self, namespace: str, id: int, vector: list[float]) -> None:
        self._client.vadd_ns(namespace, id, vector)

    def vsearch_ns(self, namespace: str, k: int, vector: list[float]) -> list[tuple[int, float]]:
        return self._client.vsearch_ns(namespace, k, vector)

    def close(self) -> None:
        self._client.disconnect()


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
