#!/bin/sh
# Wrapper: wait for Python ZeroMQ server to bind, then launch BatSim.
# Uses ONLY shell builtins + shell glob — no grep/head/cut/find needed.

BATSIM_BIN=/nix/store/1q244bhfrlsgjgamyhrzdyyrk8bmsdwb-batsim-3.1.0/bin/batsim

# Use shell glob to locate 'sleep' anywhere inside /nix/store (pure sh, no head)
SLEEP_BIN=""
for f in /nix/store/*/bin/sleep; do
    [ -f "$f" ] && SLEEP_BIN="$f" && break
done

if [ -n "$SLEEP_BIN" ]; then
    echo "[batsim_start] Found sleep at $SLEEP_BIN — waiting 6s..."
    "$SLEEP_BIN" 6
else
    echo "[batsim_start] sleep not found — continuing immediately."
fi

# Ensure output directory exists
mkdir -p /workspace/data/batsim_out

echo "[batsim_start] Launching BatSim..."
exec "$BATSIM_BIN" \
    -p /workspace/batsim_data/platforms/small_platform.xml \
    -w /workspace/data/workloads/tiny_workload.json \
    -e /workspace/data/batsim_out \
    -s tcp://shell:28000
