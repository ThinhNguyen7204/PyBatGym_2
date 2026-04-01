#!/bin/bash
# ============================================================
# PyBatGym Entrypoint — Minimal
# Image đã có sẵn mọi thứ tại /opt/venv
# Chỉ activate venv rồi mở bash tại /workspace
# ============================================================

VENV="/opt/venv"
BATSIM_BIN="/workspace/batsim_data/result/bin"

# Activate venv đã baked trong image
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# Cài pybatgym từ workspace (code mount từ host)
pip install -e /workspace --quiet 2>/dev/null || true

# Thêm BatSim vào PATH nếu có build sẵn
if [ -f "$BATSIM_BIN/batsim" ]; then
    export PATH="$BATSIM_BIN:$PATH"
fi

# In hướng dẫn
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   PyBatGym — Ubuntu $(python3 --version 2>&1 | cut -d' ' -f2)                    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  python3 examples/test_trace.py       # test (5 giây)"
echo "  python3 examples/train_ppo_trace.py  # PPO training"
echo "  pytest tests/ -v                     # unit tests"
echo "  python3 examples/quickstart.py       # mock demo"
echo ""

cd /workspace
exec bash
