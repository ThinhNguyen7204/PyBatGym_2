# Hướng Dẫn Cho Người Dùng Mới — Chỉ Cần Docker

> **Không cần cài Python, Nix, BatSim, hay bất kỳ thư viện nào.**  
> Chỉ cần **Docker Desktop** và **3 lệnh**.

---

## Yêu cầu duy nhất

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)
- Git (để clone code)

---

## Bước 1 — Cài Docker Desktop

Tải và cài từ: **https://www.docker.com/products/docker-desktop/**

Sau cài đặt: khởi động Docker Desktop, đợi icon Docker ở taskbar chuyển **xanh lá** ✅

---

## Bước 2 — Clone repo và chạy

Mở **PowerShell** (Windows) hoặc **Terminal** (Mac/Linux):

```bash
# Clone project
git clone https://github.com/khiemvuong/PyBatGym_2
cd PyBatGym_2

# Pull Docker image (chứa BatSim + Python + tất cả thư viện)
# Lần đầu ~5-10 phút (download ~7GB), lần sau rất nhanh
docker-compose pull
```

---

## Bước 3 — Chọn lệnh muốn chạy

### 🧪 Chạy test (kết quả ngay lập tức, ~5 giây)
```bash
docker-compose run test
```
**Output mong đợi:**
```
--- Testing PyBatGym with custom JSON trace ---
Running SJF Baseline on trace...
Metrics (SJF): {'avg_reward': 0.36, 'avg_utilization': 0.0, ...}
Running EASY Backfilling Baseline...
Metrics (EASY): {'avg_reward': 0.36, ...}
```

---

### 🤖 Chạy PPO Training (~vài phút)
```bash
docker-compose run train
```
**Output mong đợi:**
```
Using cpu device
Wrapping the env in a DummyVecEnv.
---------------------------------
| rollout/         |            |
|    ep_len_mean   | 6          |
|    ep_rew_mean   | 0.35       |
| time/            |            |
|    total_timesteps | 1000     |
---------------------------------
✅ Training complete!
```

---

### 📊 Xem TensorBoard (training metrics)
```bash
# Terminal 1: chạy training
docker-compose run train

# Terminal 2: mở TensorBoard
docker-compose up tensorboard
```
Mở trình duyệt: **http://localhost:6006**

---

### 🔬 Chạy unit tests
```bash
docker-compose run pytest
```

---

### 💻 Vào shell để tự khám phá
```bash
docker-compose run shell

# Bên trong container:
python3 examples/quickstart.py
python3 examples/test_trace.py
python3 examples/train_ppo_trace.py
```

---

## Lệnh hữu ích

```bash
# Xem container đang chạy
docker ps

# Dừng tất cả
docker-compose down

# Cập nhật image mới nhất
docker-compose pull

# Xóa image để giải phóng dung lượng
docker rmi ghcr.io/khiemvuong/pybatgym:latest
```

---

## Cấu trúc thư mục quan trọng

```
PyBatGym_2/
├── examples/
│   ├── test_trace.py        # ← docker-compose run test
│   └── train_ppo_trace.py   # ← docker-compose run train
├── data/workloads/
│   └── tiny_workload.json   # Workload 6 jobs cho testing
├── docker-compose.yml       # ← File điều khiển Docker
└── logs/                    # Output logs & TensorBoard data
```

---

## Troubleshooting

### ❌ `docker-compose: command not found`
→ Cài Docker Desktop (đã bao gồm docker-compose).

### ❌ `permission denied` khi pull image
→ Image đang để Private. Liên hệ `khiemvuong` để được add vào.

### ❌ Port 6006 đang bị dùng (TensorBoard)
```bash
# Đổi port trong docker-compose.yml:
ports:
  - "6007:6006"   # Dùng 6007 thay vì 6006
```

### ❌ Chạy chậm / hết RAM
→ Mở Docker Desktop → Settings → Resources → tăng RAM lên ≥ 4GB.

---

*PyBatGym v2 | Image: `ghcr.io/khiemvuong/pybatgym:latest`*
