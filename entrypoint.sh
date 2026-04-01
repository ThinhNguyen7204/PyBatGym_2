#!/bin/bash
# ============================================================
# PyBatGym Entrypoint
# Tự động setup venv + môi trường, rồi mở shell tại /workspace
# ============================================================

set -e

VENV_DIR="/workspace/.venv_ubuntu"
BATSIM_BIN="/workspace/batsim_data/result/bin"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║          PyBatGym — Ubuntu Container         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Tạo venv nếu chưa có ───────────────────────────────
if [ ! -f "$VENV_DIR/bin/python3" ]; then
    echo "⚙️  Tạo virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✅ venv tạo xong tại $VENV_DIR"
else
    echo "✅ venv đã có sẵn"
fi

# ── 2. Activate venv ──────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "✅ venv activated"

# ── 3. Cài packages nếu chưa có ──────────────────────────
if ! python3 -c "import pybatgym" 2>/dev/null; then
    echo ""
    echo "📦 Cài packages Python (lần đầu ~2 phút)..."
    pip install --upgrade pip --quiet
    pip install -e /workspace --quiet
    pip install stable-baselines3 tensorboard pytest pybatsim --quiet
    echo "✅ Packages đã cài xong"
else
    echo "✅ Packages đã có sẵn"
fi

# ── 4. Thêm BatSim vào PATH nếu có ──────────────────────
if [ -f "$BATSIM_BIN/batsim" ]; then
    export PATH="$BATSIM_BIN:$PATH"
    echo "✅ BatSim: $(batsim --version)"
else
    echo "⚠️  BatSim không có trong image (xem docs/UBUNTU_DOCKER_GUIDE.md để build)"
fi

# ── 5. Auto-activate venv trong mọi shell mới ────────────
echo "source $VENV_DIR/bin/activate" >> ~/.bashrc
echo "export PATH=$BATSIM_BIN:\$PATH" >> ~/.bashrc

# ── 6. In hướng dẫn nhanh ────────────────────────────────
echo ""
echo "━━━━━━━━━━━━ Lệnh có thể chạy ngay ━━━━━━━━━━━━━━━━"
echo "  python3 examples/test_trace.py      # test nhanh"
echo "  python3 examples/train_ppo_trace.py # PPO training"
echo "  python3 examples/quickstart.py      # mock demo"
echo "  pytest tests/ -v                    # unit tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 7. Mở bash tại /workspace ────────────────────────────
cd /workspace
exec bash
