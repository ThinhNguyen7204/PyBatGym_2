#!/bin/sh
# Wrapper: wait for Python ZeroMQ server to bind, then launch BatSim.
# Uses ONLY shell builtins + shell glob — no grep/head/cut/find needed.

# ── Locate BatSim binary dynamically (avoid hardcoded Nix hash) ──────────────
BATSIM_BIN=""
for f in /nix/store/*-batsim-*/bin/batsim; do
    [ -x "$f" ] && BATSIM_BIN="$f" && break
done

if [ -z "$BATSIM_BIN" ]; then
    echo "[batsim_start] ERROR: batsim binary not found in /nix/store!"
    exit 1
fi
echo "[batsim_start] Found batsim at $BATSIM_BIN"

# ── Locate 'sleep' via Nix store glob (not in PATH by default) ───────────────
SLEEP_BIN=""
for f in /nix/store/*/bin/sleep; do
    [ -f "$f" ] && SLEEP_BIN="$f" && break
done

if [ -n "$SLEEP_BIN" ]; then
    echo "[batsim_start] Found sleep at $SLEEP_BIN — waiting 6s for Python ZMQ bind..."
    "$SLEEP_BIN" 6
else
    echo "[batsim_start] sleep not found — continuing immediately."
fi

# ── Ensure output directory exists (mkdir via Nix store glob) ────────────────
MKDIR_BIN=""
for f in /nix/store/*/bin/mkdir; do
    [ -f "$f" ] && MKDIR_BIN="$f" && break
done

if [ -n "$MKDIR_BIN" ]; then
    "$MKDIR_BIN" -p /workspace/data/batsim_out
else
    echo "[batsim_start] mkdir not found — skipping output dir creation."
fi

echo "[batsim_start] Launching BatSim..."
exec "$BATSIM_BIN" \
    -p /workspace/batsim_data/platforms/small_platform.xml \
    -w /workspace/data/workloads/tiny_workload.json \
    -e /workspace/data/batsim_out \
    -s tcp://shell:28000
