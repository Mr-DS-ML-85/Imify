"""The Storage protocol: everything Imify needs from its database.

Designed to be small and implementable by both StrikeDB (RESP) and SQLite so
the product never depends on experimental persistence semantics.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Iterable


class StorageError(RuntimeError):
    pass


class Storage(ABC):
    # --- KV -----------------------------------------------------------
    @abstractmethod
    def ping(self) -> bool: ...

    @abstractmethod
    def get(self, key: str) -> str | None: ...

    @abstractmethod
    def set(self, key: str, value: str) -> None: ...

    @abstractmethod
    def mget(self, keys: Iterable[str]) -> list[str | None]: ...

    @abstractmethod
    def mset(self, kvs: dict[str, str]) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def incr(self, key: str, by: int = 1) -> int: ...

    @abstractmethod
    def keys(self, prefix: str) -> list[str]: ...

    # --- Vector ANN ----------------------------------------------------
    @abstractmethod
    def vadd(self, id: int, vector: list[float]) -> None: ...

    @abstractmethod
    def vdel(self, ids: list[int]) -> int:
        """Tombstone/remove vectors by id (StrikeDB VDEL). Returns how many
        ids were actually present and deleted. SQLite fallback: hard-delete."""
        ...

    @abstractmethod
    def vsearch(self, k: int, vector: list[float]) -> list[tuple[int, float]]: ...

    # Namespaced ANN (StrikeDB VADDNS/VSEARCHNS). Each namespace owns its own
    # dimension, so 512-dim face and CLIP embeddings can live alongside the
    # 64-dim pHash graph without dimension conflicts. The base default routes
    # to the flat API so backends (test doubles) that lack namespaces still
    # compile — real namespaced search only happens on StrikeDB.
    def vadd_ns(self, namespace: str, id: int, vector: list[float]) -> None:
        self.vadd(id, vector)

    def vsearch_ns(self, namespace: str, k: int, vector: list[float]) -> list[tuple[int, float]]:
        return self.vsearch(k, vector)

    # --- JSON helpers ---------------------------------------------------
    def get_json(self, key: str) -> dict[str, Any] | None:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: dict[str, Any]) -> None:
        self.set(key, json.dumps(value, default=str))

    @abstractmethod
    def close(self) -> None: ...
