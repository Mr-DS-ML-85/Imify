#!/bin/bash
# Start proxify's orchestrator (API :8081, CONNECT tunnel :8888) from the
# already-built image, on the compose network shared with proxy-redis and
# proxy-flaresolverr. Bound to 127.0.0.1 so the tunnel is never an open proxy.
# The GUI Chrome profile lives in the `proxify-gui-profile` volume so its
# cookies/cache/trust survive container re-creation.
# usage: scripts/start_proxify.sh [--recreate]
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=proxy-orchestrator
# proxify-chrome = base image + official Google Chrome (docker/Dockerfile.proxify-chrome)
IMAGE=proxy-orchestrator-orchestrator:latest
docker image inspect proxify-chrome:latest >/dev/null 2>&1 && IMAGE=proxify-chrome:latest
# Only search-engine / Facebook sessions go into the bot browser — never the
# whole exported cookie jar (password managers, mail, shops...).
COOKIES="$PWD/data/proxify-cookies.txt"
if [ -f gui-cookies.txt ]; then python3 scripts/filter_cookies.py gui-cookies.txt "$COOKIES" >/dev/null; chmod 600 "$COOKIES"; fi
# Honest, consistent persona (BrowserPilot fingerprint-profile rule): the GUI
# browser IS Chromium on Linux, so every signal — UA, client hints, platform —
# says exactly that. Claiming Windows Chrome 146 from a Linux Chromium 153 was a
# contradiction Google's risk engine scores as automation (-> CAPTCHA).
CHROME_MAJOR="${CHROME_MAJOR:-$(docker run --rm --entrypoint sh "$IMAGE" -c 'f=/usr/bin/google-chrome-stable; [ -x "$f" ] || f=$(find /ms-playwright -name chrome -path "*chrome-linux64*" -type f | head -1); "$f" --version 2>/dev/null' | grep -oE '[0-9]+' | head -1)}"
CHROME_MAJOR="${CHROME_MAJOR:-153}"
# the browser's clock must match where the IP is (a UTC clock on a Dhaka IP is
# an automation tell) — use the host's own timezone, natively
HOST_TZ="$(timedatectl show -p Timezone --value 2>/dev/null || readlink /etc/localtime | sed 's#.*/zoneinfo/##')"
HOST_TZ="${HOST_TZ:-UTC}"
PERSONA_UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${CHROME_MAJOR}.0.0.0 Safari/537.36"
PERSONA_SEC_CH_UA="\"Chromium\";v=\"${CHROME_MAJOR}\", \"Google Chrome\";v=\"${CHROME_MAJOR}\", \"Not.A/Brand\";v=\"99\""
[ "${1:-}" = "--recreate" ] && docker rm -f "$NAME" >/dev/null 2>&1 || true
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "$NAME already running"; exit 0; fi
docker rm "$NAME" >/dev/null 2>&1 || true
# A profile carried over from a previous container still holds Chromium's
# SingletonLock (tied to the old hostname) -> Chrome refuses to start. The
# container is gone, so the lock is stale: clear it.
docker run --rm --entrypoint sh -v proxify-gui-profile:/p "$IMAGE" -c 'rm -f /p/SingletonLock /p/SingletonCookie /p/SingletonSocket' >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network proxy-orchestrator_default \
  -p 127.0.0.1:8081:8081 -p 127.0.0.1:8888:8888 \
  -e API_PORT=8081 -e PROXY_PORT=8888 \
  -e REDIS_URL=redis://redis:6379/0 -e REDIS_ENABLED=true \
  -e FLARESOLVERR_URL=http://flaresolverr:8191/v1 -e FLARESOLVERR_ENABLED=true \
  -e LOG_LEVEL=INFO \
  -e PERSONA_UA="$PERSONA_UA" -e PERSONA_PLATFORM='"Linux"' -e PERSONA_PLATFORM_VERSION=6.8.0 \
  -e PERSONA_SEC_CH_UA="$PERSONA_SEC_CH_UA" \
  -e GUI_TLS_MITM="${GUI_TLS_MITM:-false}" -e GUI_HONEST_MODE="${GUI_HONEST_MODE:-true}" \
  -v "$PWD/reference/proxify/proxy-orchestrator/strategies/gui_chrome.py:/app/strategies/gui_chrome.py:ro" \
  -v "$PWD/reference/proxify/proxy-orchestrator/scripts/gui_browser.sh:/app/scripts/gui_browser.sh:ro" \
  -v "$PWD/reference/proxify/proxy-orchestrator/engine/decision_engine.py:/app/engine/decision_engine.py:ro" -e PLAYWRIGHT_BROWSERS_PATH=/ms-playwright -e TZ="$HOST_TZ" \
  -v proxify-gui-profile:/app/gui-profile \
  $( [ -f "$COOKIES" ] && echo "-v $COOKIES:/app/gui-cookies.txt:ro -e COOKIE_FILE=/app/gui-cookies.txt" ) \
  --shm-size 1g --init --restart unless-stopped "$IMAGE" >/dev/null
for _ in $(seq 1 90); do
  if curl -sf -m 2 http://127.0.0.1:8081/health >/dev/null; then
    curl -s -m 30 -X POST http://127.0.0.1:8081/cookies/refresh >/dev/null || true
    echo "proxify up (:8081 API, :8888 tunnel)"; exit 0
  fi
  sleep 2
done
echo "proxify did not become healthy — docker logs $NAME" >&2; exit 1
