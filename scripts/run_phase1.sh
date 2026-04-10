#!/bin/bash
# Phase 1 runner — execute inside Docker container
set -e

# Activate venv
source /workspace/.venv_ubuntu/bin/activate

# Install pybatgym in dev mode
pip install -e /workspace --quiet 2>/dev/null || true

# Generate workload if missing
if [ ! -f /workspace/data/workloads/medium_workload.json ]; then
    echo "[Phase 1] Generating medium workload..."
    python3 /workspace/scripts/generate_workload.py \
        --preset medium \
        --output /workspace/data/workloads/medium_workload.json
fi

echo ""
echo "[Phase 1] Starting PPO training..."
echo ""

# Run training
python3 /workspace/examples/train_ppo_phase1.py
