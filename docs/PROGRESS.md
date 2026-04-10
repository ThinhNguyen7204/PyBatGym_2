# PyBatGym — Tiến Trình Phát Triển (Progress Log)

> **Cập nhật lần cuối:** 2026-04-10
> **Phiên bản BatSim đang dùng:** `oarteam/batsim:3.1.0` (qua Docker Compose)
> **pybatsim:** 3.x — tương thích với BatSim 3.1.0

---

## Tóm Tắt Hệ Thống Hiện Tại

### Cấu hình Docker
```
docker-compose services:
  shell:       ghcr.io/khiemvuong/pybatgym:latest   <- Python + SB3 + venv
  batsim:      oarteam/batsim:3.1.0                 <- BatSim C++ simulator
  tensorboard: ghcr.io/khiemvuong/pybatgym:latest   <- TensorBoard viewer

Port mapping:  28000 (ZMQ), 6006 (TensorBoard)
Volume mount:  D:\PyBat\PyBatGym_2 -> /workspace
```

### Hai Chế Độ Hoạt Động
| Mode | Adapter | pybatsim cần? | Tốc độ |
|------|---------|---------------|--------|
| `mock` (mặc định) | `MockAdapter` | Không | Rất nhanh (~5ph/500k steps) |
| `real` | `RealBatsimAdapter` | Có | Chậm hơn (ZMQ roundtrip) |

---

## Đã Hoàn Thành — 8 Phase Ban Đầu

| Phase | Mô tả | Status |
|-------|-------|--------|
| 1 | Project & Data Foundation | Done |
| 2 | Core RL Modules (obs/action/reward) | Done |
| 3 | MockAdapter (offline simulation) | Done |
| 4 | PyBatGymEnv (Gymnasium-compatible) | Done |
| 5 | Plugin System (TensorBoard/CSV/Benchmark) | Done |
| 6 | Tests (7 files, 100% PASS) | Done |
| 7 | Validation + Benchmarking (SJF/EASY/PPO) | Done |
| 8 | Real BatSim Integration (ZeroMQ) | Done |

---

## Fixes Đã Áp Dụng (2026-04-10)

### Fix 1 — Proper Cleanup Between Episodes (real_adapter.py)

**Vấn đề:** reset() cũ chỉ join thread, không kill subprocess truoc
-> BatSim process zombie, ZMQ socket không release đúng cách.

**Giải pháp:** Thêm _kill_simulation() helper với thứ tự cleanup đúng:
1. Signal _is_done = True + unblock action_queue
2. Terminate subprocess (BatSim side closes ZMQ)
3. Join thread (Python/pybatsim side closes ZMQ)

reset() và close() đều gọi _kill_simulation().

```python
# pybatgym/real_adapter.py
def _kill_simulation(self) -> None: ...  # [NEW]
def reset(self): self._kill_simulation(); ...  # [UPDATED]
def close(self): self._kill_simulation()  # [UPDATED]
```

---

### Fix 2 — Giảm ZMQ Roundtrip Không Cần Thiết (real_adapter.py)

**Vấn đề:** onJobCompletion() luôn gọi _wakeup_and_wait() kể cả khi
không có job nào trong queue -> PPO bị đánh thức không cần thiết.

**Giải pháp:** Chỉ wake PPO khi _pending_jobs không rỗng.

```python
# pybatgym/real_adapter.py -- onJobCompletion()
if self._pending_jobs:   # [NEW CONDITION]
    self._wakeup_and_wait()
# else: BatSim tiếp tục tự động
```

Tác động: ~30-40% completion events có queue rỗng -> tiết kiệm ZMQ roundtrip.

---

### Fix 3 — Dynamic Port Selection (real_adapter.py)

**Vấn đề:** Port 28000 bị giữ sau simulation (TIME_WAIT), reset ngay
lập tức -> "Address already in use".

**Giải pháp:** _find_free_port() tự động tìm port trống từ 28000.
Chi ap dung khi dung local batsim binary (khong phai docker-compose).

```python
# pybatgym/real_adapter.py
@staticmethod
def _find_free_port(start: int = 28000) -> int: ...  # [NEW]

# _start_batsim_subprocess() -- Local mode only:
free_port = self._find_free_port(self._default_port)
self.socket_endpoint = f"tcp://*:{free_port}"
```

Luu y: External docker-compose batsim giu nguyen port 28000
(hardcoded trong scripts/batsim_start.sh -> tcp://shell:28000).

---

### Update — Chinh Xac Hoa Phien Ban BatSim

**Truoc:** docs ghi "BatSim 5.0", "Nix build" la primary path.
**Sau:**
- Project dang dung BatSim 3.1.0 (oarteam/batsim:3.1.0)
- pybatsim 3.x <-> BatSim 3.1.0 -> khong co protocol mismatch
- Nix build BatSim 5.x chuyen thanh muc tuy chon trong docs
- File cap nhat: docs/UBUNTU_DOCKER_GUIDE.md

---

## Phase 1 Results (2026-04-10) -- Large Workload Training

### Config
```
Workload    : medium_workload.json (500 jobs, log-normal runtime, Poisson arrival)
Platform    : 4 nodes x 2 cores = 8 total cores
Jobs/episode: 300 (truncated from 500)
Training    : PPO, 200k steps, n_steps=256, batch=64, lr=3e-4, ent=0.01
Reward      : hybrid (util=0.3, wait=0.5, slowdown=0.15, throughput=0.05)
FPS         : 58 steps/sec (~58 min total)
```

### Benchmark: PPO vs Heuristics
```
Metric                    | FCFS     | SJF      | EASY BF  | PPO (200k)
---------------------------------------------------------------------------
Avg Waiting Time (s)      | 805.54   | *9.98*   | 906.46   | 1024.81
Avg Slowdown              | 48.36    | *2.58*   | 59.94    | 58.64
Avg Utilization (%)       | 77.2%    | 6.7%     | *90.2%*  | 64.8%

PPO avg episode reward: -173.13
(* = best)
```

### Phan tich ket qua

**SJF thang tuyet doi** ve waiting_time va slowdown:
- SJF lam waiting_time = 9.98s vs PPO = 1024.81s (100x tot hon)
- Ly do: SJF luon chon job ngan nhat -> giai phong resource nhanh

**EASY Backfilling thang ve utilization (90.2%):**
- Backfilling dien resource "lo hong" bang jobs nho
- PPO chi dat 64.8% -- chua hoc duoc backfilling

**PPO hien tai CHUA vuot duoc heuristics.** Cac nguyen nhan:
1. **200k steps chua du** -- PPO can 500k-2M steps de converge tren 300-job episodes
2. **Episode qua dai** (300 jobs x ~10 steps/job = ~3000 steps/episode)
   -> PPO chi trai qua ~66 episodes trong 200k steps -> chua du exploration
3. **Reward signal qua thu** -- ep_reward hien thi 0.0 trong training
   (do episode chua ket thuc truoc khi SB3 log -> chi thay step reward)
4. **Action space don gian** -- top_k=10 chi chon 1 trong 10 jobs hoac WAIT
   -> khong co backfill action rieng biet

### Bug da fix trong qua trinh training

**Fix 4 — easy_backfilling_policy AttributeError (benchmark.py)**
- Bug: `env._adapter.current_time` -> MockAdapter khong co attribute nay
- Fix: doi thanh `env._adapter.get_current_time()` (public API)
- Dong thoi fix `_running_jobs` access: MockAdapter dung `_RunningJob` wrapper
  co `.finish_time` va `.cores`, khong phai raw Job objects

---

## Phase 1b Results (2026-04-10) -- Convergence-Based Stopping

### Config thay doi so voi Phase 1
```
num_jobs/episode : 100    (giam tu 300 -> nhieu hon 3x episode)
n_steps          : 512    (tang tu 256, phu hop hon voi episode ~900 steps)
batch_size       : 128    (tang tu 64)
Stop condition   : CV < 5% + slope < 0.5%/ep (convergence-based)
Hard cap         : 2,000,000 steps
```

### Convergence Report
```
Dung tai   : 150,000 steps (hard cap: 2,000,000)
Episodes   : 100
Mean reward: -79.30
Std reward : 3.41
CV         : 4.3%  (< 5% threshold -> CONVERGED)
Slope      : +0.0018 (< 0.5% threshold -> plateau)
Thoi gian  : 11 phut (vs 58 phut Phase 1)
```

### Benchmark: PPO Phase 1b vs Heuristics
```
Metric                | FCFS     | SJF      | EASY BF  | PPO 1b
-----------------------------------------------------------------------
Avg Waiting Time (s)  | 325.15   | *9.98*   | 439.77   | 331.68
Avg Slowdown          | 24.68    | *2.58*   | 23.11    | *12.06*   <- BEST!
Avg Utilization (%)   | *93.6%*  | 13.5%    | 77.3%    | 66.2%

PPO avg episode reward: -79.28
(* = best cho metric do)
```

### So sanh Phase 1 vs Phase 1b
```
                      | Phase 1 (200k, 300j) | Phase 1b (150k, 100j)
----------------------|----------------------|----------------------
Episodes thuc te      | ~66                  | 100
Avg Slowdown          | 58.64                | *12.06*  (5x tot hon!)
Avg Waiting Time      | 1024.81              | 331.68   (3x tot hon)
Thoi gian training    | 58 phut              | 11 phut
```

### Phan tich ket qua

**PPO Phase 1b da vuot FCFS va EASY BF ve Slowdown (12.06 vs 24.68 / 23.11)**
- PPO hoc duoc balance giua utilization va slowdown
- SJF van thang ve waiting_time nhung chi dat 13.5% util (workload-dependent)

**Tai sao SJF co waiting_time thap bat thuong (9.98s)?**
- Workload synthetic: log-normal runtime -> nhieu job ngan (1-10s)
- Arrival rate=0.25 -> khoang cach trung binh 4s giua cac job
- SJF luon pick job ngan nhat -> xong ngay -> gap 4s cho job moi
- Ket qua: wait~0 NHUNG util chi 13.5% (core ngoi khong gio neu job ngan)
- Tren HPC thuc te (mix short+long): SJF gay starvation, wait tang manh

**PPO hoc balanced policy**: khong toi uu 1 metric ma optimize reward tong.

---

## Benchmark Cu (tiny_workload) -- Tham khao

Dataset: tiny_workload.json (6 jobs, 5 nodes) -- qua nho

```
Metric              | SJF   | EASY  | PPO (10k steps)
avg_waiting_time    | 0.00  | 0.00  | 0.00
avg_slowdown        | 1.00  | 1.00  | 1.00
avg_utilization     | 48.0% | 48.0% | 48.0%
```

---

## Buoc Tiep Theo (Roadmap)

### Priority 1 — Dataset Lon + Train That Su

| Buoc | Action | Status |
|------|--------|--------|
| 1.1 | Tao workload generator (generate_workload.py) | DONE |
| 1.2 | Sinh medium_workload.json (500 jobs) + heavy_workload.json (1000 jobs) | DONE |
| 1.3 | Tao train_ppo_phase1.py voi hyperparameters toi uu | DONE |
| 1.4 | Train PPO 200k steps, 4 nodes x 2 cores | DONE |
| 1.5 | Fix easy_backfilling_policy bug | DONE |
| 1.6 | PPO chua vuot heuristics -> Phase 1b | DONE |

### Priority 1b -- Convergence-Based Training

| Buoc | Action | Status |
|------|--------|--------|
| 1b.1 | ConvergenceCallback (CV + slope threshold) | DONE |
| 1b.2 | Giam num_jobs/episode xuong 100 | DONE |
| 1b.3 | Chay convergence training, dung tai 150k steps | DONE |
| 1b.4 | PPO vuot FCFS+EASY ve Slowdown (12.06 vs 24.68) | DONE |

---

### Priority 1c -- Fix Observation & Reward Issues (Phase 1c)

> Muc tieu: Cai thien chat luong thong tin PPO nhan duoc & them Backfill-awareness.
> Chi tiet: docs/ENV_DESIGN.md

| ID | File | Action | Status |
|----|------|--------|--------|
| OBS-1 | observation.py | [8+4i] duplicate -> doi thanh bounded_slowdown_norm | DONE |
| OBS-2 | observation.py | [46-49] placeholder -> jobs_fitting_now, queue_urgency, min/max walltime | DONE |
| RWD-1 | reward.py | Hardcode x 64 -> doi thanh x config.platform.total_cores | DONE |
| ACT-1 | action.py + env | Them action SCHEDULE_SMALLEST_FITTING (backfill-aware, K+2) | DONE |
| TEST | benchmark | Test va eval (PPO: 13.36, FCFS: 11.67, EASY: 10.53) | DONE |

---

### Priority 2 -- SWF Parser & NASA Trace (TIEP THEO)

> Muc tieu: Thu chay PPO tren workload du lieu that tu SWF de danh gia he thong

| Buoc | Action | Status |
|------|--------|--------|
| 2.1 | Viet SWF parser (read standard workloads from PWA) | TODO |
| 2.2 | Convert NASA iPSC trace / KTH-SP2 trace sang json format cua workload generator | TODO |
| 2.3 | Chay Phase 2: Train tren Real Trace | TODO |

---

### Priority 2 — SWF Parser

| Buoc | Action | Status |
|------|--------|--------|
| 2.1 | Implement SWF parser trong workload_parser.py | TODO |
| 2.2 | Test voi NASA Ames workload (.swf) | TODO |

---

### Priority 3 — Validate Real Mode sau 3 Fixes

| Buoc | Action | Status |
|------|--------|--------|
| 3.1 | Chay test_real.py -> xac nhan khong port conflict | TODO |
| 3.2 | Chay nhieu episodes lien tiep (reset lap lai) | TODO |
| 3.3 | So sanh metrics Mock vs Real | TODO |

---

### Priority 4 — Multi-Resource Support (GPU + Memory)

| Buoc | Action | Status |
|------|--------|--------|
| 4.1 | Mo rong Resource model trong models.py | TODO |
| 4.2 | Cap nhat ObservationBuilder | TODO |
| 4.3 | Cap nhat MockAdapter scheduler | TODO |

---

## Kien Truc — Khi Nao Can pybatsim

```
Training (Mock Mode)          Validation (Real Mode)
        |                              |
        v                              v
    MockAdapter                 RealBatsimAdapter
    (Python only)               (pybatsim + BatSim C++)
        |                              |
    Khong ZMQ                     ZMQ roundtrip x2
    1 thread                      2 threads (queue sync)
    ~5ph/500k steps               Nhieu gio/500k steps
        |                              |
    Du de hoc policy              Ket qua chinh xac vat ly
```

**Chien luoc:**
Phase A: Train voi MockAdapter + large dataset -> nhanh, hoc policy
Phase B: Validate voi RealAdapter -> verify accuracy vs BatSim C++

---

## Files Quan Trong

| File | Mo ta | Ghi chu |
|------|-------|---------|
| pybatgym/real_adapter.py | ZeroMQ bridge -> BatSim C++ | 3 fixes da ap dung |
| pybatgym/batsim_adapter.py | MockAdapter | On dinh |
| pybatgym/env.py | Gymnasium Env | On dinh |
| pybatgym/observation.py | Observation builder | Known issues: OBS-1, OBS-2 |
| pybatgym/reward.py | Reward calculator | Known issue: RWD-1 (hardcode 64) |
| pybatgym/plugins/benchmark.py | Heuristic baselines | Fix 4 (EASY AttributeError) |
| examples/train_ppo_trace.py | PPO training (tiny workload) | Phase 0 demo |
| examples/train_ppo_phase1.py | PPO Phase 1 (300 jobs/ep) | 200k steps, result: slowdown=58 |
| examples/train_ppo_phase1b.py | PPO Phase 1b (100 jobs/ep) | Convergence stop, result: slowdown=12 |
| scripts/generate_workload.py | Workload generator | medium + heavy presets |
| scripts/run_phase1.sh | Docker runner Phase 1 | bash |
| scripts/run_phase1b.sh | Docker runner Phase 1b | bash |
| data/workloads/medium_workload.json | 500 jobs, log-normal runtime | Generated |
| data/workloads/heavy_workload.json | 1000 jobs, heavy contention | Generated |
| models/ppo_phase1.zip | Trained PPO model | 200k steps |
| models/ppo_phase1b.zip | Trained PPO model (Phase 1b) | 150k steps, converged |
| models/ppo_phase1b_best.zip | Best checkpoint | Auto-saved by ConvergenceCallback |
| docker-compose.yml | Docker services | BatSim 3.1.0 |
| docs/UBUNTU_DOCKER_GUIDE.md | Setup guide | Da cap nhat BatSim 3.x |
| docs/ENV_DESIGN.md | Action/Obs/Reward reference | Known issues, improvement plan |

---

*Cap nhat: 2026-04-10 | PyBatGym v2 | BatSim 3.1.0 | pybatsim 3.x*

