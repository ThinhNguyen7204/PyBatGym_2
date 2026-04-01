#!/bin/sh
# Wrapper script: wait for Python ZeroMQ server to bind before BatSim connects.
# The oarteam/batsim:3.1.0 Nix image does not have 'sleep' in PATH,
# so we locate it inside the Nix store.

BATSIM_BIN=/nix/store/1q244bhfrlsgjgamyhrzdyyrk8bmsdwb-batsim-3.1.0/bin/batsim

# Find 'sleep' inside the Nix store
SLEEP_BIN=$(find /nix/store -name sleep -type f 2>/dev/null | head -1)

if [ -n "$SLEEP_BIN" ]; then
    echo "[batsim_start] Waiting 5s for Python ZMQ server to bind..."
    "$SLEEP_BIN" 5
else
    echo "[batsim_start] 'sleep' not found in /nix/store, proceeding immediately."
fi

# Ensure output directory exists
mkdir -p /workspace/data/batsim_out

echo "[batsim_start] Starting BatSim..."
exec "$BATSIM_BIN" \
    -p /workspace/batsim_data/platforms/small_platform.xml \
    -w /workspace/data/workloads/tiny_workload.json \
    -e /workspace/data/batsim_out \
    -s tcp://shell:28000
