# Hướng Dẫn Chạy PyBatGym Trên Ubuntu (Docker)

> **Mục tiêu:** Chạy PyBatGym với BatSim thật (C++ simulator) qua ZeroMQ,
> sử dụng Docker container Ubuntu 22.04 trên Windows (WSL2).

---

## Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Yêu Cầu Hệ Thống](#2-yêu-cầu-hệ-thống)
3. [Bước 1: Khởi Tạo Docker Container](#bước-1-khởi-tạo-docker-container)
4. [Bước 2: Cài Đặt Python Environment](#bước-2-cài-đặt-python-environment)
5. [Bước 3: Cài Đặt Nix Package Manager](#bước-3-cài-đặt-nix-package-manager)
6. [Bước 4: Build BatSim Từ Source](#bước-4-build-batsim-từ-source)
7. [Bước 5: Cài PyBatsim Bridge](#bước-5-cài-pybatsim-bridge)
8. [Bước 6: Chạy Test](#bước-6-chạy-test)
9. [Lưu Ý Quan Trọng](#lưu-ý-quan-trọng)
10. [Troubleshooting](#troubleshooting)
11. [Tham Khảo Nhanh (Cheat Sheet)](#tham-khảo-nhanh-cheat-sheet)

---

## 1. Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Container (Ubuntu 22.04)              │
│                                                                 │
│  ┌──────────────┐   ZeroMQ    ┌─────────────────────────────┐  │
│  │   BatSim     │◄──────────►│    PyBatGym (Python)         │  │
│  │   (C++ bin)  │  tcp:28000 │    ├── RealBatsimAdapter     │  │
│  │              │            │    ├── PyBatsim bridge        │  │
│  │  Nix-built   │            │    └── RL Agent (SB3/PPO)    │  │
│  └──────────────┘            └─────────────────────────────────┘│
│                                                                 │
│  /workspace (mount) ←──→ D:\PyBat\PyBatGym_2 (Windows host)   │
└─────────────────────────────────────────────────────────────────┘
```

**Luồng hoạt động:**
1. `RealBatsimAdapter` spawn BatSim C++ process
2. BatSim mô phỏng HPC cluster, gửi events qua ZeroMQ
3. `pybatsim` (Python bridge) nhận events, chuyển thành PyBatGym `Event` objects
4. RL Agent (PPO/SJF/EASY) nhận observation, ra quyết định scheduling
5. Quyết định gửi ngược lại BatSim qua ZeroMQ → vòng lặp tiếp tục

---

## 2. Yêu Cầu Hệ Thống

| Yêu cầu | Chi tiết |
|----------|----------|
| **OS Host** | Windows 10/11 với WSL2 |
| **Docker** | Docker Desktop for Windows (WSL2 backend) |
| **RAM** | ≥ 8GB (Nix build cần ~4GB) |
| **Disk** | ≥ 10GB trống (Nix store + BatSim build) |
| **Project** | `D:\PyBat\PyBatGym_2` (hoặc path tùy chọn) |

---

## Bước 1: Khởi Tạo Docker Container

### 1.1. Mở Docker Desktop

Đảm bảo Docker Desktop đang chạy (icon Docker ở system tray).

### 1.2. Tạo container với volume mount

```powershell
# Chạy từ PowerShell trên Windows
docker run -it --rm -v D:\PyBat\PyBatGym_2:/workspace ubuntu:22.04 bash
```

> **Lưu ý:** Flag `--rm` sẽ xóa container khi thoát.
> Bỏ `--rm` nếu muốn giữ container để dùng lại:
> ```powershell
> docker run -it --name pybatgym -v D:\PyBat\PyBatGym_2:/workspace ubuntu:22.04 bash
> # Lần sau: docker start -i pybatgym
> ```

### 1.3. Cài công cụ cơ bản

```bash
apt-get update -qq && apt-get install -y \
  python3 python3-pip python3-venv \
  git curl xz-utils sudo \
  cmake g++ pkg-config \
  libboost-all-dev libzmq3-dev rapidjson-dev libczmq-dev \
  meson ninja-build unzip
```

---

## Bước 2: Cài Đặt Python Environment

### 2.1. Tạo virtual environment

```bash
cd /workspace
python3 -m venv .venv_ubuntu
source .venv_ubuntu/bin/activate
```

### 2.2. Cài dependencies Python

```bash
pip install --upgrade pip setuptools wheel
pip install -e .                    # PyBatGym package
pip install stable-baselines3       # RL algorithms
pip install tensorboard torch       # Training monitoring
pip install pytest                  # Testing
```

### 2.3. Kiểm tra cài đặt

```bash
python -c "from pybatgym.env import PyBatGymEnv; print('✅ PyBatGym OK')"
python -c "import stable_baselines3; print('✅ SB3 OK')"
```

---

## Bước 3: Cài Đặt Nix Package Manager

BatSim 5.0 dùng **Nix flake** để quản lý tất cả C++ dependencies (SimGrid, intervalset, batprotocol...).

### 3.1. Cài Nix

```bash
# Cài Nix (single-user mode, không cần systemd)
sh <(curl -L https://nixos.org/nix/install) --no-daemon
```

### 3.2. Load Nix vào shell

```bash
source ~/.nix-profile/etc/profile.d/nix.sh
# Kiểm tra
nix --version
```

### 3.3. Bật Flakes (experimental features)

```bash
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

### 3.4. Tạo nixbld group (nếu thiếu)

```bash
groupadd -f nixbld
for n in $(seq 1 10); do
  useradd -g nixbld -M -N "nixbld$n" 2>/dev/null || true
done
```

---

## Bước 4: Build BatSim Từ Source

### 4.1. BatSim source

Source code BatSim nằm tại `/workspace/batsim_data/` (đã có sẵn trong repo).

> **Lưu ý:** `batsim_data/` đã được thêm vào `.gitignore`.
> Nếu cần clone lại:
> ```bash
> git clone https://framagit.org/batsim/batsim.git /workspace/batsim_data
> ```

### 4.2. Build bằng Nix (KHUYẾN NGHỊ)

```bash
export PATH=$HOME/.nix-profile/bin:$PATH
cd /workspace/batsim_data
nix build .#batsim
```

⏳ **Thời gian:** 10-30 phút lần đầu. Nix tự tải và build:
- SimGrid (physics simulation)
- intervalset (từ framagit.org)
- batprotocol-cpp (ZMQ protocol)
- CLI11, và các deps khác

### 4.3. Kiểm tra build

```bash
./result/bin/batsim --version
# Expected: 5.0.0-rc1
```

### 4.4. Thêm vào PATH

```bash
export PATH=/workspace/batsim_data/result/bin:$PATH
echo 'export PATH=/workspace/batsim_data/result/bin:$PATH' >> ~/.bashrc

# Verify
which batsim
batsim --version
```

---

## Bước 5: Cài PyBatsim Bridge

PyBatsim là Python bridge giao tiếp với BatSim qua ZeroMQ.

```bash
source /workspace/.venv_ubuntu/bin/activate
pip install pybatsim
```

> **Phiên bản:** pybatsim 3.2.x tương thích với BatSim 4.x protocol.
> BatSim 5.0 dùng batprotocol mới — nếu gặp lỗi communication,
> xem phần [Troubleshooting](#troubleshooting).

---

## Bước 6: Chạy Test

### 6.1. Test Mock Mode (không cần BatSim)

```bash
cd /workspace
source .venv_ubuntu/bin/activate

# Unit tests
pytest tests/ -v

# Quick mock simulation
python examples/quickstart.py

# Trace-based mock
python examples/test_trace.py
```

### 6.2. Test Real Mode (Cần chạy 2 Terminal song song)

Chế độ `real` yêu cầu giao tiếp ZeroMQ giữa Python và BatSim. Do chạy qua Docker Compose, bạn cần bật Python lên trước để mở port, sau đó bật BatSim.

**Terminal 1 (Ubuntu shell):** Khởi chạy tác nhân Python
```bash
# Đảm bảo đã vào trong Ubuntu container bằng: docker exec -it pybatgym_2-shell-1 bash
source /workspace/.venv_ubuntu/bin/activate
export PATH=/workspace/batsim_data/result/bin:$PATH

python examples/test_real.py
# Script sẽ nạp và dừng ở dòng: 
# "Assuming BatSim is running externally (e.g., Plan B via Docker)."
```

**Terminal 2 (PowerShell):** Khởi chạy BatSim (C++)
```powershell
cd D:\PyBat\PyBatGym_2
# BatSim tự động chờ 6 giây (nhờ batsim_start.sh) rồi mới kết nối vào Python
docker-compose up batsim
```

**Kết quả thành công:**
Bên Terminal 1 (Python) sẽ in ra chuỗi sự kiện và tóm tắt `--- RESULTS ---`:
```
[RealBatsimAdapter] Local 'batsim' binary not found. Assuming BatSim is running externally...
[Run] Running Shortest-Job-First Baseline...
--- RESULTS ---
{'avg_reward': -0.52, 'avg_utilization': 0.0, 'avg_waiting_time': 0.0, 'avg_slowdown': 0.0}
Test finished.
```

### 6.3. Training PPO

```bash
python examples/train_ppo_trace.py
```

### 6.4. TensorBoard

```bash
# Trong container
tensorboard --logdir logs/tensorboard_ppo --bind_all --port 6006

# Truy cập từ Windows browser: http://localhost:6006
```

---

## Lưu Ý Quan Trọng

### Mỗi lần mở container mới, cần chạy:

```bash
# 1. Activate Python env
source /workspace/.venv_ubuntu/bin/activate

# 2. Load Nix
source ~/.nix-profile/etc/profile.d/nix.sh

# 3. Thêm BatSim vào PATH
export PATH=/workspace/batsim_data/result/bin:$PATH

# 4. Verify
batsim --version && python -c "import pybatgym; print('Ready!')"
```

### File paths

- **Tất cả `.py` files dùng relative path** từ `Path(__file__)` — không hardcode absolute path.
- **Windows:** `D:\PyBat\PyBatGym_2\...`
- **Container:** `/workspace/...`
- Code tự detect đúng path nhờ `pathlib`.

### Git workflow

```bash
# .agent/ và batsim_data/ đã có trong .gitignore
# Nếu đã lỡ push lên git, chạy:
git rm -r --cached .agent/
git rm -r --cached batsim_data/
git rm -r --cached __pycache__/
git rm -r --cached .pytest_cache/
git rm -r --cached .venv_ubuntu/
git rm -r --cached pybatgym.egg-info/
git commit -m "chore: untrack ignored files"
```

---

## Troubleshooting

### ❌ `docker: error during connect`
**Lý do:** Docker Desktop chưa chạy.
**Fix:** Mở Docker Desktop, đợi icon chuyển xanh, thử lại.

### ❌ `batsim: command not found`
**Fix:**
```bash
export PATH=/workspace/batsim_data/result/bin:$PATH
```

### ❌ `TypeError: Batsim.__init__() got an unexpected keyword argument`
**Lý do:** pybatsim 3.x dùng positional args.
**Fix đã áp dụng:** `Batsim(self, self.socket_endpoint)` thay vì keyword args.

### ❌ `Can't instantiate abstract class RealBatsimAdapter`
**Lý do:** Thiếu implement `start()`, `step()`, `get_current_time()`.
**Fix đã áp dụng:** `real_adapter.py` đã được rewrite đầy đủ.

### ❌ `Timeout waiting for BatSim`
**Kiểm tra:**
1. BatSim có trong PATH không? → `which batsim`
2. Port 28000 có bị chiếm? → `ss -tlnp | grep 28000`
3. Workload file có tồn tại? → `ls /workspace/data/workloads/`
4. Platform file có tồn tại? → `ls /workspace/batsim_data/platforms/`

### ❌ BatSim 5.0 + pybatsim 3.x protocol mismatch
BatSim 5.0 dùng `batprotocol` (mới), pybatsim 3.x dùng protocol cũ.
**Workaround:** Dùng BatSim 4.x hoặc chờ pybatsim 4.x hỗ trợ batprotocol.

### ❌ Nix build lỗi `error: unable to fork`
**Fix:** Container cần thêm RAM. Thêm flag: `docker run --memory=4g ...`

---

## Tham Khảo Nhanh (Cheat Sheet)

```bash
# ══════════════════════════════════════════════════════════════════
# KHỞI TẠO (Lần đầu — ~30 phút)
# ══════════════════════════════════════════════════════════════════

# Windows PowerShell:
docker run -it --name pybatgym -v D:\PyBat\PyBatGym_2:/workspace ubuntu:22.04 bash

# Trong container:
apt-get update -qq && apt-get install -y python3 python3-pip python3-venv git curl xz-utils sudo cmake g++ pkg-config libboost-all-dev libzmq3-dev rapidjson-dev libczmq-dev meson ninja-build unzip

cd /workspace
python3 -m venv .venv_ubuntu && source .venv_ubuntu/bin/activate
pip install --upgrade pip && pip install -e . stable-baselines3 tensorboard torch pytest pybatsim

sh <(curl -L https://nixos.org/nix/install) --no-daemon
source ~/.nix-profile/etc/profile.d/nix.sh
mkdir -p ~/.config/nix && echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

cd /workspace/batsim_data && nix build .#batsim
export PATH=/workspace/batsim_data/result/bin:$PATH
echo 'export PATH=/workspace/batsim_data/result/bin:$PATH' >> ~/.bashrc

batsim --version  # → 5.0.0-rc1 ✅

# ══════════════════════════════════════════════════════════════════
# SỬ DỤNG HÀNG NGÀY (~5 giây)
# ══════════════════════════════════════════════════════════════════

# Windows PowerShell:
docker start -i pybatgym

# Trong container:
source /workspace/.venv_ubuntu/bin/activate
source ~/.nix-profile/etc/profile.d/nix.sh
export PATH=/workspace/batsim_data/result/bin:$PATH

cd /workspace
python examples/test_real.py       # Real BatSim test
python examples/train_ppo_trace.py # PPO training
pytest tests/ -v                   # Unit tests
```

---

## Cấu Trúc Project

```
PyBatGym_2/
├── .gitignore              # Ignore __pycache__, .agent/, batsim_data/, .venv_ubuntu/
├── pyproject.toml           # Package config
├── README.md
│
├── pybatgym/                # Core library
│   ├── env.py               # Gymnasium environment
│   ├── batsim_adapter.py    # Abstract adapter + MockAdapter
│   ├── real_adapter.py      # RealBatsimAdapter (ZMQ ↔ BatSim)
│   ├── models.py            # Job, Resource, Event dataclasses
│   ├── observation.py       # Observation builder
│   ├── action.py            # Action mapper
│   ├── reward.py            # Reward calculator
│   ├── workload_parser.py   # JSON workload parser
│   └── plugins/             # Benchmark, logging plugins
│
├── data/                    # Workloads & platforms cho MockAdapter
│   ├── workloads/
│   │   └── tiny_workload.json
│   └── platforms/
│
├── batsim_data/             # ⚠️ GITIGNORED — BatSim C++ source
│   ├── flake.nix            # Nix build config
│   ├── result/bin/batsim    # Built binary (after nix build)
│   ├── platforms/           # SimGrid platform XMLs
│   └── workloads/           # BatSim-format workloads
│
├── configs/
│   └── default.yaml         # Default PyBatGym config
│
├── examples/
│   ├── quickstart.py        # Mock mode demo
│   ├── test_trace.py        # Trace-based mock test
│   ├── test_real.py         # Real BatSim test (Docker only!)
│   └── train_ppo_trace.py   # PPO training script
│
├── tests/
│   ├── test_workload_parser.py
│   └── test_reward.py
│
├── docs/
│   └── UBUNTU_DOCKER_GUIDE.md  # ← Bạn đang đọc file này
│
└── logs/                    # ⚠️ GITIGNORED
    └── tensorboard_ppo/
```

---

*Tạo ngày: 2026-03-31 | PyBatGym v2 + BatSim 5.0.0-rc1*
