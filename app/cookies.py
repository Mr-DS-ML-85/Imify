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


# Seconds between 1601-01-01 (Chromium's epoch) and 1970-01-01.
_WINDOWS_EPOCH_OFFSET = 11_644_473_600


def normalize_expiry(exp: int) -> int:
    """Unix expiry seconds. Some exporters write Chromium's 1601-based
    seconds as-is, pushing every expiry ~369 years into the future so dead
    sessions never look expired; shift those back."""
    if exp > time.time() + 100 * 365 * 86400:
        return exp - _WINDOWS_EPOCH_OFFSET
    return exp


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
                        if normalize_expiry(int(expires)) < time.time():
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


def cookie_health(path: str | Path | None, domains: tuple[str, ...] = ("facebook.com", "google.com", "yandex.com", "bing.com")) -> dict:
    """Freshness report for the session-cookie file (never exposes values).

    Session cookies expire (Facebook's ``xs``/``c_user`` typically within
    days), after which the FB/Google resolution chain silently degrades to
    login stubs. Surfacing per-domain counts + the soonest expiry lets the
    admin re-export cookies BEFORE searches start failing.
    """
    if not path:
        return {"configured": False}
    p = Path(path)
    if not p.is_file():
        return {"configured": True, "exists": False}
    now = time.time()
    report: dict[str, dict] = {d: {"live": 0, "expired": 0, "soonest_expiry_h": None} for d in domains}
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except OSError:
        return {"configured": True, "exists": True, "readable": False}
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        host = parts[0].replace("#HttpOnly_", "").lstrip(".").lower()
        for d in domains:
            if host == d or host.endswith("." + d):
                try:
                    exp = normalize_expiry(int(parts[4] or 0))
                except ValueError:
                    exp = 0
                entry = report[d]
                if exp and exp < now:
                    entry["expired"] += 1
                else:
                    entry["live"] += 1
                    if exp:
                        hours = round((exp - now) / 3600, 1)
                        if entry["soonest_expiry_h"] is None or hours < entry["soonest_expiry_h"]:
                            entry["soonest_expiry_h"] = hours
    return {
        "configured": True,
        "exists": True,
        "file_age_h": round((now - p.stat().st_mtime) / 3600, 1),
        "domains": report,
    }
