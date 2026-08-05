#!/usr/bin/env bash
# Start the Imify stack robustly (detached, survives shell exit).
#   StrikeDB on :6008 (WAL = data/dbstrike.wal)
#   API+UI   on :8010
set -u
cd "$(dirname "$0")/.."
mkdir -p data

echo "== starting StrikeDB :6008 (WAL=data/dbstrike.wal) =="
pkill -f 'dbstrike 127.0.0.1:6008' 2>/dev/null
sleep 1
DBSTRIKE_WAL=./data/dbstrike.wal setsid nohup reference/StrikeDB/target/release/dbstrike 127.0.0.1:6008 \
  >> data/dbstrike.log 2>&1 < /dev/null &
sleep 2
redis-cli -p 6008 PING && echo "StrikeDB up" || echo "StrikeDB FAILED"

echo "== starting app :8010 =="
pkill -f 'uvicorn app.main:app.*8010' 2>/dev/null
sleep 1
setsid nohup uv run uvicorn app.main:app --host 127.0.0.1 --port 8010 \
  >> data/app-8010.log 2>&1 < /dev/null &
sleep 6
curl -s --max-time 5 http://127.0.0.1:8010/health/live && echo " <- app health" || echo "app not responding yet"

echo "== done =="
