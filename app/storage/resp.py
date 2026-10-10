"""Minimal pure-Python RESP client — no third-party Redis library.

Talks to any RESP server (StrikeDB speaks RESP). Supports the commands Imify
uses: PING/AUTH/SET/GET/MSET/MGET/DEL/INCR/KEYS/VADD/VSEARCH. Implements
pipelined MGET/MSET (one round trip) and safe reconnect + timeout handling.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

from .base import StorageError


class RespError(StorageError):
    pass


class RespClient:
    def __init__(self, host: str, port: int, password: str = "", timeout: float = 10.0):
        self.host = host
        self.port = port
        self.password = password
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = b""
        # One socket, many callers (API threads + indexer worker threads): every
        # command round-trip is serialized so wire frames can never interleave.
        # StrikeDB replies in microseconds, so a single lock is not a bottleneck
        # at Imify scale and is far simpler than a connection pool.
        self._lock = threading.RLock()

    # -- connection -------------------------------------------------------
    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._sock = sock
        self._buf = b""
        if self.password:
            self._request_raw(["AUTH", self.password])

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._buf = b""

    def _read_until(self, marker: bytes) -> bytes:
        while marker not in self._buf:
            assert self._sock is not None
            chunk = self._sock.recv(65536)
            if not chunk:
                raise RespError("connection closed by server")
            self._buf += chunk
        line, self._buf = self._buf.split(marker, 1)
        return line

    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            assert self._sock is not None
            chunk = self._sock.recv(65536)
            if not chunk:
                raise RespError("connection closed by server")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _read_reply(self) -> Any:
        line = self._read_until(b"\r\n")
        if not line:
            raise RespError("empty reply line")
        t, payload = line[:1], line[1:]
        if t == b"+":
            return payload.decode("utf-8", "replace")
        if t == b"-":
            raise RespError(payload.decode("utf-8", "replace"))
        if t == b":":
            return int(payload)
        if t == b"$":
            length = int(payload)
            if length == -1:
                return None
            data = self._read_exact(length)
            self._read_exact(2)  # trailing CRLF
            return data
        if t == b"*":
            count = int(payload)
            if count == -1:
                return None
            return [self._read_reply() for _ in range(count)]
        raise RespError(f"unknown RESP type byte {t!r}")

    def _request_raw(self, args: list[str | bytes]) -> Any:
        encoded: list[bytes] = [a.encode() if isinstance(a, str) else a for a in args]
        parts = [f"*{len(encoded)}\r\n".encode()]
        for arg in encoded:
            parts.append(f"${len(arg)}\r\n".encode() + arg + b"\r\n")
        assert self._sock is not None
        self._sock.sendall(b"".join(parts))
        return self._read_reply()

    def request(self, *args: str | bytes) -> Any:
        with self._lock:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    self.connect()
                    return self._request_raw(list(args))
                except (OSError, RespError) as exc:
                    last_error = exc
                    self.disconnect()
                    if isinstance(exc, RespError):
                        msg = str(exc).lower()
                        # Protocol/server errors: retrying won't fix them.
                        if "bad" in msg or "syntax" in msg or "unknown" in msg or "not a" in msg or "requires" in msg:
                            raise
                    time.sleep(0.05 * (attempt + 1))
            raise RespError(f"RESP request failed after retries: {last_error}")

    # -- commands ---------------------------------------------------------
    def ping(self) -> bool:
        try:
            return self.request("PING") == "PONG"
        except RespError:
            return False

    def get(self, key: str) -> bytes | None:
        reply = self.request("GET", key)
        return reply if isinstance(reply, bytes) else None

    def set(self, key: str, value: str | bytes) -> None:
        self.request("SET", key, value)

    def mget(self, keys: list[str]) -> list[bytes | None]:
        if not keys:
            return []
        reply = self.request("MGET", *keys)
        return list(reply)

    def mset(self, kvs: dict[str, str | bytes]) -> None:
        if not kvs:
            return
        args: list[str | bytes] = []
        for k, v in kvs.items():
            args.extend([k, v])
        self.request("MSET", *args)

    def delete(self, key: str) -> bool:
        return bool(self.request("DEL", key))

    def incr(self, key: str, by: int = 1) -> int:
        if by == 1:
            return int(self.request("INCR", key))
        return int(self.request("INCRBY", key, str(by)))

    def keys(self, prefix: str) -> list[str]:
        reply = self.request("KEYS", prefix + "*")
        return [k.decode("utf-8", "replace") for k in reply]

    def vadd(self, id: int, vector: list[float]) -> None:
        self.request("VADD", str(id), *[f"{x:.6f}" for x in vector])

    def vdel(self, ids: list[int]) -> int:
        if not ids:
            return 0
        return int(self.request("VDEL", *[str(i) for i in ids]))

    def vsearch(self, k: int, vector: list[float]) -> list[tuple[int, float]]:
        reply = self.request("VSEARCH", str(k), *[f"{x:.6f}" for x in vector])
        out: list[tuple[int, float]] = []
        for i in range(0, len(reply), 2):
            vid = int(reply[i])
            dist = float(reply[i + 1])
            out.append((vid, dist))
        return out

    def vadd_ns(self, namespace: str, id: int, vector: list[float]) -> None:
        self.request("VADDNS", namespace, str(id), *[f"{x:.6f}" for x in vector])

    def vsearch_ns(self, namespace: str, k: int, vector: list[float]) -> list[tuple[int, float]]:
        reply = self.request("VSEARCHNS", namespace, str(k), *[f"{x:.6f}" for x in vector])
        out: list[tuple[int, float]] = []
        for i in range(0, len(reply), 2):
            vid = int(reply[i])
            dist = float(reply[i + 1])
            out.append((vid, dist))
        return out
