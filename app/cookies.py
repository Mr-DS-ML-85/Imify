"""Netscape-format cookie-file reader for candidate verification.

proxify's real user session lives in a Netscape cookie file (``gui-cookies.txt``
in the project root by default). Hotlink-protected CDNs like Facebook's
``lookaside.fbsbx.com`` / ``scontent-*.fbcdn.net`` return login stubs to
anonymous fetches — attaching the real session cookies makes those candidates
fetchable so the fused-hash matcher can actually judge them.

Only the Cookie header is needed: the CONNECT tunnel (proxify) already carries
the request to the right IP; cookies complete the authenticated persona.

The module is deliberately tiny and dependency-free — a cookie file is just
tab-separated rows. Expired cookies are skipped, and domain matching follows
the ``includeSubdomains`` flag (and a host-suffix fallback so ``.facebook.com``
covers ``lookaside.fbsbx.com``-adjacent hosts when the flag is set).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path


class CookieFile:
    """Lazily-loaded Netscape cookie file with per-host header export.

    Thread-safe: the file is re-read at most once per mtime change, and the
    read is guarded so concurrent verify workers (``web_verify_concurrency``
    threads share one instance) never observe a half-loaded table.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mtime: float = -1.0
        self._by_domain: dict[str, dict[str, str]] = {}
        self._domain_flags: dict[str, bool] = {}  # domain -> include_subdomains
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def _load_if_stale(self) -> None:
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return  # file missing — keep whatever we had
        if mtime == self._mtime:
            return
        by_domain: dict[str, dict[str, str]] = {}
        domain_flags: dict[str, bool] = {}
        try:
            with open(self.path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 7:
                        continue
                    domain, include_sub, _, _secure, expires, name, value = parts[:7]
                    try:
                        if int(expires) < time.time():
                            continue  # expired — never send
                    except ValueError:
                        pass
                    domain = domain.lstrip(".")
                    if not domain:
                        continue
                    domain_flags[domain] = include_sub.strip().upper() == "TRUE"
                    by_domain.setdefault(domain, {})[name] = value
        except OSError:
            return
        # publish atomically under the lock so readers never see a half-built table
        self._by_domain = by_domain
        self._domain_flags = domain_flags
        self._mtime = mtime

    # ----------------------------------------------------------------- query
    def header_for(self, host: str) -> str:
        """Build a ``Cookie`` header value for *host* (or "" if none apply)."""
        self._load_if_stale()
        if not self._by_domain:
            return ""
        host = host.lower().rstrip(".")
        parts = host.split(".")
        merged: dict[str, str] = {}

        def collect(domain: str) -> None:
            for cand, include_sub in self._domain_flags.items():
                if not include_sub and cand != domain:
                    continue
                if include_sub and not (domain == cand or domain.endswith("." + cand)):
                    continue
                if domain == cand or domain.endswith("." + cand):
                    merged.update(self._by_domain.get(cand, {}))

        # exact + progressively shorter suffixes
        for i in range(len(parts)):
            collect(".".join(parts[i:]))
        if not merged:
            return ""
        return "; ".join(f"{k}={v}" for k, v in merged.items())

    @property
    def count(self) -> int:
        self._load_if_stale()
        return sum(len(v) for v in self._by_domain.values())

    @property
    def domains(self) -> list[str]:
        self._load_if_stale()
        return sorted(self._by_domain)


def build_cookie_header(cookie_path: str | Path | None, host: str) -> str:
    """One-shot convenience: read the file fresh and export for *host*."""
    if not cookie_path:
        return ""
    return CookieFile(cookie_path).header_for(host)
