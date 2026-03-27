# Tài Liệu Hướng Dẫn Kiểm Định & Benchmarking (PyBatGym)

Tài liệu này hướng dẫn chi tiết về **Quy trình Validation và Performance Benchmarking** dành cho môi trường **PyBatGym**, phục vụ việc đánh giá năng lực của các mô hình RL so với các thuật toán lập lịch truyền thống của HPC Lab.

---

## 1. Sử Dụng Dữ Liệu Thực Tế (Traces)

Môi trường mặc định tự sinh ra (synthetic) các job ngẫu nhiên. Để test thực tế, bạn cần sử dụng các file `.json` từ [Parallel Workloads Archive](https://www.cs.huji.ac.il/labs/parallel/workload/) hoặc dữ liệu mẫu từ repo BatSim.

### Cách cấu hình để chạy với file trace:
```python
from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv

config = PyBatGymConfig()
config.workload.source = "trace"
config.workload.trace_path = "data/workloads/tiny_workload.json"
env = PyBatGymEnv(config=config)
```

**Dữ liệu có sẵn trong repo:**
- `data/workloads/tiny_workload.json` — 6 jobs, lấy từ BatSim test suite
- `data/platforms/small_platform.xml` — SimGrid platform 5 nodes

---

## 2. Kiểm Tra Tính Hợp Lệ Của Môi Trường (Sanity Check)

### 2.1 TesterPlugin (Tự động)
```python
from pybatgym.plugins.tester import TesterPlugin

env.register_plugin(TesterPlugin(run_check_env=True, run_sanity_check=True))
env.reset()  # TesterPlugin sẽ in báo cáo tại đây
```

### 2.2 Gọi thủ công
```python
from gymnasium.utils.env_checker import check_env
check_env(env)
```

---

## 3. Trực Quan Hoá Quá Trình Training (TensorBoard)

### Sử dụng TensorBoardLoggerPlugin
```python
from pybatgym.plugins.tensorboard_logger import TensorBoardLoggerPlugin

env.register_plugin(TensorBoardLoggerPlugin(log_dir="logs/tensorboard"))
```

**Xem kết quả:**
```bash
pip install tensorboard torch  # (nếu chưa cài)
tensorboard --logdir logs/tensorboard
```

Plugin sẽ ghi các metrics `Episode/Total_Reward`, `Metrics/Utilization`, `Metrics/Average_Waiting_Time` vào TensorBoard mỗi episode.

---

## 4. Benchmarking So Sánh RL vs Heuristics

PyBatGym có sẵn **3 chiến lược heuristic** làm baseline:

| Heuristic | Mô tả | Ưu điểm |
|-----------|--------|---------|
| **FCFS** | Đến trước phục vụ trước | Đơn giản, công bằng |
| **SJF** | Ưu tiên job ngắn nhất | Giảm avg waiting time |
| **EASY Backfilling** | Đặt chỗ cho job đầu, nhét job nhỏ vào khoảng trống | Chuẩn công nghiệp HPC |

### Cách chạy so sánh
```python
from pybatgym.plugins.benchmark import run_baseline, sjf_policy, easy_backfilling_policy, fcfs_policy

metrics_fcfs = run_baseline(env, fcfs_policy, num_episodes=5)
metrics_sjf = run_baseline(env, sjf_policy, num_episodes=5)
metrics_easy = run_baseline(env, easy_backfilling_policy, num_episodes=5)
```

### So sánh với PPO

Xem script mẫu đầy đủ tại `examples/train_ppo_trace.py`:

```bash
python examples/train_ppo_trace.py
```

Script sẽ tự in bảng so sánh giữa SJF, EASY Backfilling, và PPO:
```text
Metric                    | SJF Baseline    | EASY Backfill   | Trained PPO
---------------------------------------------------------------------------
Avg Waiting Time (s)      | 0.00            | 0.00            | 0.00
Avg Slowdown              | 1.00            | 1.00            | 0.00
Avg Utilization (%)       | 0.0%            | 0.0%            | 0.0%
```

---

## 5. Danh Sách Các Bài Test Tự Động (Pytest)

### Chạy toàn bộ test:
```bash
python -m pytest tests/ -v
```

### Danh sách test files:

| File | Mục đích |
|------|---------|
| `test_observation.py` | Shape, normalization, action mask |
| `test_action.py` | WAIT action, invalid action, resource check |
| `test_reward.py` | Step/episodic reward, penalty |
| `test_adapter.py` | MockAdapter lifecycle, reset, close |
| `test_env.py` | Gym API compliance, render, determinism |
| `test_integration.py` | Full episode + PPO training (conditional) |
| `test_workload_parser.py` | JSON trace parsing accuracy |

---

## 6. Kiểm Thử Với Real BatSim (WSL/Linux)

Đây là quy trình kiểm thử cao cấp nhất, kết nối Python với lõi C++ của BatSim thực sự.

### Yêu cầu:
1. WSL Ubuntu hoặc Linux
2. `batsim` đã cài qua Nix: `nix-env -iA nixpkgs.batsim`
3. `pybatsim` đã cài: `pip install pybatsim`

### Chạy test:
```bash
# Trong WSL terminal
cd /mnt/d/ThinhProject
python examples/test_real.py
```

### Kết quả mong đợi:
- BatSim C++ subprocess được tự động gọi
- ZeroMQ kết nối thành công
- Baseline SJF chạy trên mô phỏng thực
- Metrics in ra console

---

## 7. Tóm Tắt Lệnh Test Nhanh

| Mục đích | Lệnh |
|----------|-------|
| Kiểm tra toàn bộ source code | `pytest tests/ -v` |
| Chạy baseline trên trace | `python examples/test_trace.py` |
| Train PPO + so sánh | `python examples/train_ppo_trace.py` |
| Test BatSim thật (WSL) | `python examples/test_real.py` |
| Xem TensorBoard | `tensorboard --logdir logs/tensorboard` |
