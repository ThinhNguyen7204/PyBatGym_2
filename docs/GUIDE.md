# PyBatGym — Hướng Dẫn Chi Tiết

## 1. Giới Thiệu

**PyBatGym** là một môi trường Gymnasium-compatible dùng để huấn luyện các tác tử RL (Reinforcement Learning) giải quyết bài toán lập lịch công việc trên hệ thống HPC (High Performance Computing). Dự án sử dụng BatSim làm backend mô phỏng (hoặc MockAdapter cho phát triển nhanh).

### Kiến Trúc 6 Lớp

| Lớp | Module | Chức năng |
|-----|--------|-----------|
| 1 | `env.py` | Giao diện Gymnasium (reset/step/render) |
| 2 | `observation.py`, `action.py`, `reward.py` | Logic RL: xây dựng observation, ánh xạ action, tính reward |
| 3 | `env.py` (internal) | Đồng bộ sự kiện, cache trạng thái |
| 4 | `batsim_adapter.py`, `real_adapter.py` | Giao tiếp với BatSim (Mock hoặc Real qua ZeroMQ) |
| 5 | `config/` | Hệ thống cấu hình Pydantic + YAML |
| 6 | `plugins/` | Logging, benchmarking, monitoring, validation |

---

## 2. Cài Đặt

```bash
# Clone dự án
cd D:\ThinhProject

# Cài đặt ở chế độ phát triển
pip install -e .

# Cài thêm dependencies cho RL
pip install -e ".[rl]"

# Cài thêm cho test
pip install -e ".[dev]"

# Hoặc tất cả
pip install -e ".[all]"
```

### Dependencies

| Package | Version | Mô tả |
|---------|---------|--------|
| gymnasium | ≥0.29.0 | Gymnasium API |
| numpy | ≥1.24.0 | Tính toán số |
| pydantic | ≥2.0.0 | Validation config |
| pyyaml | ≥6.0 | Đọc file YAML |
| stable-baselines3 | ≥2.0.0 | (Optional) Thuật toán RL |
| pybatsim | latest | (Optional) Kết nối BatSim thật qua ZeroMQ |
| tensorboard | latest | (Optional) TensorBoard visualization |
| pytest | ≥7.0 | (Dev) Testing |

---

## 3. Sử Dụng Cơ Bản

### 3.1 Chạy với Gymnasium

```python
import gymnasium as gym
import pybatgym  # Đăng ký env

env = gym.make("PyBatGym-v0")
obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### 3.2 Cấu Hình Tùy Chỉnh

```python
from pybatgym.config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv

config = PyBatGymConfig()
config.platform.total_nodes = 32
config.platform.cores_per_node = 8
config.workload.num_jobs = 200
config.workload.seed = 123
config.reward_weights.utilization = 0.4
config.reward_weights.waiting_time = 0.3

env = PyBatGymEnv(config=config)
```

Hoặc dùng YAML:

```python
env = PyBatGymEnv(config_path="configs/default.yaml")
```

### 3.3 Chế Độ Mock vs Real

```python
# Mock (mặc định) - chạy nhanh, không cần BatSim
config.mode = "mock"

# Real - kết nối BatSim thật qua ZeroMQ (yêu cầu WSL/Linux)
config.mode = "real"
```

---

## 4. Observation Space

Vector đặc trưng có kích thước cố định `11 + 4K` (K = `top_k_jobs`, mặc định 10):

| Segment | Dim | Mô tả |
|---------|-----|--------|
| Global | 5 | `[time_progress, queue_length, utilization, avg_wait, avg_bsd]` |
| Job Queue | 4×K | Cho mỗi job: `[waiting_time, walltime, cores, urgency]` |
| Resource | 6 | `[free_cores, free_nodes, util, max_util, std_util, fragmentation]` |

Tất cả giá trị được chuẩn hóa về `[0, 1]`.

**Action Mask**: Vector `(K+1,)` — `1.0` = action hợp lệ, `0.0` = không thể thực thi.

---

## 5. Action Space

`Discrete(K+1)`:

| Action | Ý nghĩa |
|--------|---------|
| `0` đến `K-1` | Chọn job thứ i từ top-K hàng đợi |
| `K` | WAIT — không lập lịch, chờ sự kiện tiếp theo |

Nếu job được chọn không đủ tài nguyên → tự động chuyển thành WAIT.

---

## 6. Reward System

### Ba chế độ reward:

| Chế độ | Mô tả |
|--------|--------|
| `step` | Reward tại mỗi bước: ΔUtilization + schedule_bonus − idle_penalty |
| `episodic` | Reward cuối episode: tổng hợp avg_wait, avg_slowdown, utilization |
| `hybrid` | Kết hợp step + bonus cuối episode (mặc định) |

### Trọng số có thể chỉnh:

```yaml
reward_weights:
  utilization: 0.3    # Tỷ lệ sử dụng cluster
  waiting_time: 0.3   # Thời gian chờ đợi (penalty)
  slowdown: 0.3       # Bounded slowdown (penalty)
  throughput: 0.1     # Thông lượng job
```

---

## 7. Plugin System

PyBatGym có 5 plugin tích hợp sẵn:

### 7.1 CSV Logger
```python
from pybatgym.plugins.logger import CSVLoggerPlugin

env = PyBatGymEnv(config=config)
env.register_plugin(CSVLoggerPlugin(output_dir="logs"))
```
Mỗi episode tạo một file `pybatgym_ep0001.csv` chứa metrics từng bước.

### 7.2 Baseline Benchmarks (FCFS / SJF / EASY Backfilling)
```python
from pybatgym.plugins.benchmark import fcfs_policy, sjf_policy, easy_backfilling_policy, run_baseline

results_fcfs = run_baseline(env, fcfs_policy, num_episodes=5)
results_sjf = run_baseline(env, sjf_policy, num_episodes=5)
results_easy = run_baseline(env, easy_backfilling_policy, num_episodes=5)
```

### 7.3 TensorBoard Logger
```python
from pybatgym.plugins.tensorboard_logger import TensorBoardLoggerPlugin

env.register_plugin(TensorBoardLoggerPlugin(log_dir="logs/tensorboard"))
# Xem kết quả: tensorboard --logdir logs/tensorboard
```

### 7.4 Tester (Gymnasium check_env)
```python
from pybatgym.plugins.tester import TesterPlugin

env.register_plugin(TesterPlugin(run_check_env=True, run_sanity_check=True))
env.reset()  # TesterPlugin tự chạy check_env ở lần reset đầu
```

### 7.5 Benchmark Plugin
```python
from pybatgym.plugins.benchmark import BenchmarkPlugin

env.register_plugin(BenchmarkPlugin(run_on_close=True))
```

### 7.6 Tạo Plugin Riêng
```python
from pybatgym.plugins.registry import Plugin

class MyPlugin(Plugin):
    @property
    def name(self) -> str:
        return "my_plugin"

    def on_step(self, action, reward, state, done):
        pass
```

---

## 8. Nạp Dữ Liệu Thực (Workload Traces)

Thay vì tự sinh job ngẫu nhiên, bạn có thể nạp file JSON từ BatSim:

```python
config.workload.source = "trace"
config.workload.trace_path = "data/workloads/tiny_workload.json"
```

Module `pybatgym.workload_parser` sẽ tự đọc và chuyển đổi sang `List[Job]`.

**Nguồn dữ liệu:**
- Có sẵn: `data/workloads/tiny_workload.json` (6 jobs, lấy từ BatSim)
- Thêm: [Parallel Workloads Archive](https://www.cs.huji.ac.il/labs/parallel/workload/)

---

## 9. Train PPO (Stable-Baselines3)

```python
from stable_baselines3 import PPO
from pybatgym.env import PyBatGymEnv
from pybatgym.config import PyBatGymConfig

config = PyBatGymConfig()
config.workload.num_jobs = 100
env = PyBatGymEnv(config=config)

model = PPO("MultiInputPolicy", env, verbose=1, n_steps=256, batch_size=64)
model.learn(total_timesteps=50_000)

# Đánh giá
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    done = terminated or truncated

print(f"Completed: {info['completed']} jobs, Util: {info['utilization']:.1%}")
```

Xem script mẫu đầy đủ: `examples/train_ppo_trace.py`

---

## 10. Chạy Tests

```bash
# Tất cả tests
pytest tests/ -v

# Chỉ test observation
pytest tests/test_observation.py -v

# Với coverage
pytest tests/ --cov=pybatgym --cov-report=html
```

---

## 11. Cấu Trúc File Dự Án

```
D:\ThinhProject\
├── pyproject.toml               # Build config & dependencies
├── README.md                    # Overview (English)
├── configs/
│   └── default.yaml             # Default YAML config
├── data/
│   ├── platforms/
│   │   └── small_platform.xml   # SimGrid platform mẫu
│   └── workloads/
│       └── tiny_workload.json   # Workload JSON mẫu (6 jobs)
├── pybatgym/
│   ├── __init__.py              # Package init + env registration
│   ├── env.py                   # PyBatGymEnv (Layer 1+3)
│   ├── observation.py           # ObservationBuilder (Layer 2)
│   ├── action.py                # ActionMapper (Layer 2)
│   ├── reward.py                # RewardCalculator (Layer 2)
│   ├── batsim_adapter.py        # MockAdapter (Layer 4)
│   ├── real_adapter.py          # RealBatsimAdapter (Layer 4) [WSL/Linux]
│   ├── workload_parser.py       # JSON/SWF trace loader
│   ├── models.py                # Data models (Job, Resource, Event)
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base_config.py       # Pydantic config (Layer 5)
│   │   └── loader.py            # YAML loader
│   └── plugins/
│       ├── __init__.py
│       ├── registry.py          # Plugin ABC + Registry (Layer 6)
│       ├── logger.py            # CSV Logger plugin
│       ├── benchmark.py         # FCFS/SJF/EASY + BenchmarkPlugin
│       ├── tensorboard_logger.py # TensorBoard metric logging
│       └── tester.py            # Gymnasium check_env plugin
├── tests/
│   ├── test_observation.py
│   ├── test_action.py
│   ├── test_reward.py
│   ├── test_adapter.py
│   ├── test_env.py
│   ├── test_integration.py
│   └── test_workload_parser.py
├── examples/
│   ├── quickstart.py            # Demo cơ bản
│   ├── test_trace.py            # Chạy baseline trên workload trace
│   ├── train_ppo_trace.py       # Train PPO + so sánh heuristics
│   └── test_real.py             # Test với BatSim thực (WSL)
└── docs/
    ├── GUIDE.md                 # Tài liệu này
    ├── TESTING.md               # Quy trình kiểm định & benchmark
    └── PROJECT_REPORT.md        # Báo cáo tổng kết
```

---

## 12. Kết Nối BatSim Thật (Real Adapter)

Để chạy mô phỏng chuẩn xác 100% trên C++ simulator thay vì Mock:

### 12.1 Yêu Cầu
- **WSL Ubuntu** hoặc **Linux** (BatSim không chạy trực tiếp trên Windows)
- Cài đặt `batsim` qua Nix: `nix-env -iA nixpkgs.batsim`
- Cài đặt `pybatsim`: `pip install pybatsim`

### 12.2 Sử Dụng
```python
config = PyBatGymConfig()
config.mode = "real"  # Chuyển sang Real Adapter
config.workload.source = "trace"
config.workload.trace_path = "/mnt/d/ThinhProject/data/workloads/tiny_workload.json"

env = PyBatGymEnv(config=config)
obs, info = env.reset()  # Khởi động BatSim + ZeroMQ kết nối
```

Xem script mẫu: `examples/test_real.py`

---

## 13. Mở Rộng (Roadmap)

| Feature | Trạng thái | Mô tả |
|---------|-----------|--------|
| MockAdapter | ✅ Done | Giả lập nội bộ tốc độ cao |
| RealBatsimAdapter | ✅ Done | Kết nối BatSim thật qua ZeroMQ/pybatsim |
| TensorBoard logging | ✅ Done | `TensorBoardLoggerPlugin` |
| Workload traces | ✅ Done | `workload_parser.py` hỗ trợ JSON |
| Testing/Benchmark | ✅ Done | FCFS, SJF, EASY Backfilling, TesterPlugin |
| PPO Training | ✅ Done | `train_ppo_trace.py` + bảng so sánh |
| Multi-resource | 🔜 Todo | GPU, memory ngoài cores |
| SWF Parser | 🔜 Todo | Đọc .swf từ Parallel Workloads Archive |
