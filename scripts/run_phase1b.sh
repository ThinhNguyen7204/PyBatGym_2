#!/bin/bash
# Phase 1b: PPO with convergence-based stopping
set -e

source /workspace/.venv_ubuntu/bin/activate
pip install -e /workspace --quiet 2>/dev/null || true

if [ ! -f /workspace/data/workloads/medium_workload.json ]; then
    echo "[Phase 1b] Generating medium workload..."
    python3 /workspace/scripts/generate_workload.py \
        --preset medium \
        --output /workspace/data/workloads/medium_workload.json
fi

echo ""
echo "[Phase 1b] Training PPO (convergence-based, max 2M steps)..."
echo ""

python3 /workspace/examples/train_ppo_phase1b.py
