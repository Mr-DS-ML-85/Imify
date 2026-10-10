#!/bin/bash
# Start StrikeDB (Imify's own DB) on 127.0.0.1:6008, CPU-only.
# dbstrike dynamically LINKS libnvrtc.so.12 even in CPU mode; data/lib holds a
# symlink to it so nothing else from a CUDA toolkit dir shadows system libs.
# DBSTRIKE_GPU=cpu guarantees the device is never initialized (CPU-only policy).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=reference/StrikeDB/target/release/dbstrike
if redis-cli -p 6008 PING >/dev/null 2>&1; then echo "StrikeDB already up on :6008"; exit 0; fi
export LD_LIBRARY_PATH="$PWD/data/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DBSTRIKE_GPU=cpu DBSTRIKE_WAL=data/dbstrike.wal
setsid nohup "$BIN" 127.0.0.1:6008 >> data/dbstrike.log 2>&1 < /dev/null &
for _ in $(seq 1 60); do redis-cli -p 6008 PING >/dev/null 2>&1 && { echo "StrikeDB up on :6008"; exit 0; }; sleep 1; done
echo "StrikeDB failed to start — see data/dbstrike.log" >&2; exit 1
