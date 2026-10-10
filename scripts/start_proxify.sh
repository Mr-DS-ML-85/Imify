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
# Only search-engine / Facebook sessions go into the bot browser — never the
# whole exported cookie jar (password managers, mail, shops...).
COOKIES="$PWD/data/proxify-cookies.txt"
if [ -f gui-cookies.txt ]; then python3 scripts/filter_cookies.py gui-cookies.txt "$COOKIES" >/dev/null; chmod 600 "$COOKIES"; fi
[ "${1:-}" = "--recreate" ] && docker rm -f "$NAME" >/dev/null 2>&1 || true
IMAGE=proxy-orchestrator-orchestrator:latest
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "$NAME already running"; exit 0; fi
docker rm "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network proxy-orchestrator_default \
  -p 127.0.0.1:8081:8081 -p 127.0.0.1:8888:8888 \
  -e API_PORT=8081 -e PROXY_PORT=8888 \
  -e REDIS_URL=redis://redis:6379/0 -e REDIS_ENABLED=true \
  -e FLARESOLVERR_URL=http://flaresolverr:8191/v1 -e FLARESOLVERR_ENABLED=true \
  -e LOG_LEVEL=INFO -e PLAYWRIGHT_BROWSERS_PATH=/ms-playwright -e TZ=UTC \
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
