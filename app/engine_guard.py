"""EngineGuard — stay under the radar instead of fighting CAPTCHAs.

BrowserPilot's lesson applied to Imify's engine traffic: anti-bot systems
escalate on *behavior*. A soft challenge (CAPTCHA) becomes a hard IP ban when
the client keeps hammering through it. So instead of solving challenges,
every engine call goes through this guard:

- **pacing** — a per-engine minimum interval with human-like jitter, so a
  burst of searches never turns into a burst of identical requests;
- **block detection** — CAPTCHA / "unusual traffic" / 403 / 429 answers are
  recognized from the engine's error text or page;
- **cooldown with exponential backoff** — a blocked engine rests (10 min,
  then 20, 40 ... up to 6 h) and is skipped meanwhile, other engines carry
  the search; one clean answer resets the strike count;
- **daily budget** — a hard cap of calls per engine per day.

State lives in storage, so a restart never resets a cooldown.
"""

from __future__ import annotations

import random
import re
import threading
import time

from app.logging import child_logger

log = child_logger("imify.engine_guard")

# Text that means "you were challenged/blocked", not "no results".
_BLOCK = re.compile(
    r"captcha|/sorry|unusual traffic|automated queries|are you a robot|not a robot|"
    r"verify you are (a )?human|access denied|blocked|status=40[13]|status=429|"
    r"too many requests|rate.?limit|anomaly|cf-challenge|just a moment|attention required|"
    r"showcaptcha|smartcaptcha|подтвердите",
    re.I,
)

# Minimum seconds between two calls to the same engine (jittered x0.7-1.6).
DEFAULT_INTERVALS = {
    # Google is the trigger-happy one; the others tolerate human cadence
    # (a person really does search twice within a few seconds).
    "google": 12.0,
    "google-upload": 12.0,
    "yandex": 4.0,
    "yandex-upload": 4.0,
    "yandex-face": 4.0,
    "bing": 3.0,
    "bing-upload": 3.0,
}
DEFAULT_DAILY_BUDGET = {"google": 150, "google-upload": 150}


def looks_blocked(text: str) -> bool:
    return bool(text) and bool(_BLOCK.search(text))


class EngineGuard:
    def __init__(
        self,
        db,
        base_cooldown: float = 600.0,
        max_cooldown: float = 6 * 3600.0,
        intervals: dict[str, float] | None = None,
        daily_budget: dict[str, int] | None = None,
        max_wait: float = 8.0,
    ):
        self.db = db
        self.base_cooldown = base_cooldown
        self.max_cooldown = max_cooldown
        self.intervals = {**DEFAULT_INTERVALS, **(intervals or {})}
        self.daily_budget = {**DEFAULT_DAILY_BUDGET, **(daily_budget or {})}
        self.max_wait = max_wait
        self._next_slot: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ keys
    @staticmethod
    def _key(engine: str) -> str:
        return f"eg:{engine}"

    @staticmethod
    def _day_key(engine: str) -> str:
        return f"eg:n:{engine}:{time.strftime('%Y%m%d')}"

    def _state(self, engine: str) -> dict:
        try:
            return self.db.get_json(self._key(engine)) or {}
        except Exception:  # noqa: BLE001 — guard must never break a search
            return {}

    # ------------------------------------------------------------ gate
    def allow(self, engine: str) -> tuple[bool, str]:
        """(allowed, reason). Not allowed while cooling down or over budget."""
        st = self._state(engine)
        until = float(st.get("until", 0))
        if until > time.time():
            return False, f"cooldown {int(until - time.time())}s after block"
        budget = self.daily_budget.get(engine)
        if budget:
            try:
                used = int(self.db.get(self._day_key(engine)) or 0)
            except Exception:  # noqa: BLE001
                used = 0
            if used >= budget:
                return False, f"daily budget {budget} used"
        return True, ""

    def pace(self, engine: str) -> bool:
        """Reserve this engine's next call slot, sleeping up to ``max_wait``
        for it. Returns False when the slot is further away than that (the
        caller skips the engine this time rather than queueing a burst)."""
        interval = self.intervals.get(engine, 0.0)
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(engine, 0.0))
            wait = slot - now
            if wait > self.max_wait:
                return False
            self._next_slot[engine] = slot + interval * random.uniform(0.7, 1.6)
        if wait > 0:
            time.sleep(wait)
        try:
            self.db.incr(self._day_key(engine))
        except Exception:  # noqa: BLE001
            pass
        return True

    # ------------------------------------------------------------ feedback
    def report(self, engine: str, ok: bool, text: str = "") -> None:
        """Feed back the outcome. A block signal starts/extends the
        cooldown; a clean answer clears the strikes."""
        st = self._state(engine)
        if ok and not looks_blocked(text):
            if st.get("strikes"):
                self._save(engine, {"strikes": 0, "until": 0, "last_ok": time.time()})
            return
        if not looks_blocked(text):
            return  # a plain failure (timeout, empty) is not a block signal
        strikes = int(st.get("strikes", 0)) + 1
        rest = min(self.max_cooldown, self.base_cooldown * (2 ** (strikes - 1)))
        rest *= random.uniform(0.9, 1.3)  # no clockwork retry pattern
        self._save(engine, {
            "strikes": strikes,
            "until": time.time() + rest,
            "reason": (text or "")[:160],
            "blocked_at": time.time(),
        })
        log.warning("engine blocked — cooling down", extra={"engine": engine, "strikes": strikes, "rest_s": int(rest)})

    def _save(self, engine: str, state: dict) -> None:
        try:
            self.db.set_json(self._key(engine), state)
        except Exception:  # noqa: BLE001
            pass

    def status(self, engines: list[str]) -> dict:
        now = time.time()
        out = {}
        for e in engines:
            st = self._state(e)
            until = float(st.get("until", 0))
            out[e] = {
                "cooling_down_s": max(0, int(until - now)),
                "strikes": int(st.get("strikes", 0)),
                "reason": st.get("reason", "") if until > now else "",
            }
        return out
