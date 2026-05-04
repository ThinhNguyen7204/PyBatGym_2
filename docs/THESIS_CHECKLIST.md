# PyBatGym — Thesis Requirements Checklist

> **Nguồn:** Handout 2 - Yêu cầu thesis PyBatGym (NotebookLM)
> **Đối chiếu:** `docs/PROGRESS.md`, source code `pybatgym/`, `tests/`, `examples/`
> **Cập nhật:** 2026-05-04

---

## Tóm Tắt Nhanh

| Mức đánh giá                   | Trạng thái                  |
| ------------------------------ | --------------------------- |
| **Minimum Pass** (9/9 items)   | ✅ ĐẠT                      |
| **Expected Level** (8/8 items) | ✅ ĐẠT                      |
| **Strong Level** (6/6 items)   | ✅ 5/6 ĐẠT (thiếu ablation) |

**Kết luận:** Dự án đạt mức **Strong Level** gần như hoàn chỉnh.

---

## 1. Functional Requirements (FR)

### FR1 — Gymnasium API ✅ DONE

| Yêu cầu                                          | Status | Evidence                                                                                                                   |
| ------------------------------------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------- |
| `reset(seed, options) -> obs, info`              | ✅     | `env.py:83-111`                                                                                                            |
| `step(action) -> obs, reward, term, trunc, info` | ✅     | `env.py:113-152`                                                                                                           |
| `observation` shape cố định                      | ✅     | `observation.py:64` — `(11+4K,)`                                                                                           |
| `action_space` là Discrete                       | ✅     | `action.py:62-63` — `Discrete(K+2)`                                                                                        |
| `observation_space` khai báo rõ                  | ✅     | `observation.py:66-74` — `Dict{features, action_mask}`                                                                     |
| `info` chứa `action_mask` + `metrics`            | ✅     | `env.py:183` — `info["action_mask"]` + step/sim_time/utilization. **MaskablePPO compatible** |
| `action_masks()` method cho MaskablePPO           | ✅     | `env.py:167-179` — SB3-contrib auto-detect interface, boolean array (K+2,) |

### FR2 — Event-driven simulation semantics ✅ DONE

| Yêu cầu                              | Status | Evidence                                                    |
| ------------------------------------ | ------ | ----------------------------------------------------------- |
| Job arrival → pending job            | ✅     | `batsim_adapter.py:245-257`                                 |
| Job completion → giải phóng resource | ✅     | `batsim_adapter.py:259-284`                                 |
| EXECUTE_JOB không advance time       | ✅     | `batsim_adapter.py:169-175` — stay at current time          |
| WAIT → time jump to next event       | ✅     | `batsim_adapter.py:178` — `_advance_to_next_decision_point` |
| Episode end: F=∅, Q=∅, R=∅           | ✅     | `batsim_adapter.py:286-297`                                 |

### FR3 — State vs Observation separation ✅ DONE

| Yêu cầu                                                           | Status | Evidence                                             |
| ----------------------------------------------------------------- | ------ | ---------------------------------------------------- |
| Full state nội bộ (pending, running, completed, future, resource) | ✅     | `env.py:97-103` — state dict                         |
| ObservationBuilder rút trích vector từ state                      | ✅     | `observation.py:48-89` — `DefaultObservationBuilder` |

### FR4 — Action mapping ✅ DONE

| Yêu cầu                          | Status | Evidence           |
| -------------------------------- | ------ | ------------------ |
| `0..K-1`: chọn job Top-K         | ✅     | `action.py:99-118` |
| `K`: WAIT                        | ✅     | `action.py:80-81`  |
| `K+1`: BACKFILL/SMALLEST_FITTING | ✅     | `action.py:84-97`  |

### FR5 — Action mask ✅ DONE

| Yêu cầu                                  | Status | Evidence                                    |
| ---------------------------------------- | ------ | ------------------------------------------- |
| Mask cho vị trí không có job             | ✅     | `observation.py:200-208`                    |
| Mask cho job cần nhiều resource hơn free | ✅     | `observation.py:207` — `can_allocate` check |
| Mask cho BACKFILL action                 | ✅     | `observation.py:211-212`                    |

### FR6 — Reward function ✅ DONE

| Yêu cầu                                     | Status | Evidence                         |
| ------------------------------------------- | ------ | -------------------------------- |
| Reward thay đổi được (configurable weights) | ✅     | `reward.py:49` — `RewardWeights` |
| Schedule bonus                              | ✅     | `reward.py:76`                   |
| Invalid action penalty                      | ✅     | `reward.py:78`                   |
| Utilization delta                           | ✅     | `reward.py:71-72`                |
| Multi-objective shaping (episodic)          | ✅     | `reward.py:95-118`               |
| 3 modes: step/episodic/hybrid               | ✅     | `reward.py:1-7` docstring        |

### FR7 — Metrics collection ✅ DONE

| Metric               | Status | Evidence                                  |
| -------------------- | ------ | ----------------------------------------- |
| completed jobs       | ✅     | `env.py:177`                              |
| average waiting time | ✅     | Computed in benchmark & training scripts  |
| bounded slowdown     | ✅     | `models.py` — `bounded_slowdown` property |
| utilization          | ✅     | `env.py:178`                              |
| makespan             | ✅     | Implicit via `total_time`                 |
| throughput           | ✅     | `reward.py:103`                           |
| total reward         | ✅     | `env.py:179`                              |
| number of steps      | ✅     | `env.py:174`                              |

### FR8 — Benchmark pipeline ✅ DONE

| Yêu cầu                       | Status | Evidence                                                                                                                                                                    |
| ----------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Script benchmark nhiều policy | ✅     | `plugins/benchmark.py` — FCFS/SJF/EASY                                                                                                                                      |
| Xuất kết quả cho báo cáo      | ✅     | Console output + TensorBoard                                                                                                                                                |
| CSV/JSON output chuẩn hóa     | ✅     | `results/benchmark_summary.csv` — 19 columns, 216 rows (4 policies × 3 workloads × 2 platforms × 3 seeds × 3 episodes) |

### FR9 — Baseline policies ✅ DONE (vượt yêu cầu)

| Policy             | Status | Evidence                                  |
| ------------------ | ------ | ----------------------------------------- |
| Random             | ✅     | Implicit qua Gymnasium random sampling    |
| FCFS               | ✅     | `plugins/benchmark.py`                    |
| SJF                | ✅     | `plugins/benchmark.py`                    |
| EASY-like Backfill | ✅     | `plugins/benchmark.py`                    |
| PPO baseline       | ✅     | `examples/train_ppo_*.py`, 4 saved models |

### FR10 — Reproducibility ✅ DONE

| Yêu cầu                   | Status | Evidence                                 |
| ------------------------- | ------ | ---------------------------------------- |
| Config lưu được           | ✅     | YAML presets `configs/`                  |
| Seed                      | ✅     | `env.py:89` — `super().reset(seed=seed)` |
| Workload/platform setting | ✅     | `data/workloads/`, `data/platforms/`     |
| Policy name               | ✅     | Logged in training scripts               |
| Metrics                   | ✅     | TensorBoard + console output             |

### FR11 — Adapter-based design ✅ DONE (vượt yêu cầu)

| Yêu cầu                            | Status | Evidence                                              |
| ---------------------------------- | ------ | ----------------------------------------------------- |
| `SimulatorAdapter` interface (ABC) | ✅     | `batsim_adapter.py:30-63` — `BatsimAdapter` ABC       |
| ToySimulatorAdapter (EventDriven)  | ✅     | `batsim_adapter.py:70-330` — `EventDrivenMockAdapter` |
| BatsimAdapter (Real BatSim)        | ✅     | `real_adapter.py` — `RealBatsimAdapter` (ZeroMQ)      |
| Env không gọi BatSim trực tiếp     | ✅     | `env.py` chỉ gọi `self._adapter.*`                    |

---

## 2. Non-Functional Requirements (NFR)

### NFR1 — Modularity ✅ DONE

| Module              | File                                                 | Status |
| ------------------- | ---------------------------------------------------- | ------ |
| Environment         | `env.py`                                             | ✅     |
| Simulator Adapter   | `batsim_adapter.py`, `real_adapter.py`               | ✅     |
| Observation Builder | `observation.py`                                     | ✅     |
| Action Mapper       | `action.py`                                          | ✅     |
| Reward Function     | `reward.py`                                          | ✅     |
| Baseline Policies   | `plugins/benchmark.py`                               | ✅     |
| Metrics Logger      | `plugins/logger.py`, `plugins/tensorboard_logger.py` | ✅     |

### NFR2 — Extensibility ✅ DONE

| Mở rộng                      | Status | Evidence                              |
| ---------------------------- | ------ | ------------------------------------- |
| Reward function mới          | ✅     | ABC `RewardCalculator` + DI trong env |
| Observation feature mới      | ✅     | ABC `ObservationBuilder` + DI         |
| Baseline scheduler mới       | ✅     | Plugin system                         |
| Simulator backend mới        | ✅     | `BatsimAdapter` ABC                   |
| Workload/platform config mới | ✅     | YAML loader + workload generator      |

### NFR3 — Testability ✅ DONE

| Test bắt buộc           | Status | File                                          |
| ----------------------- | ------ | --------------------------------------------- |
| reset API               | ✅     | `test_env.py`                                 |
| step API                | ✅     | `test_env.py`                                 |
| action mask             | ✅     | `test_observation.py`                         |
| event transition        | ✅     | `test_event_driven_adapter.py`                |
| episode termination     | ✅     | `test_env.py`, `test_event_driven_adapter.py` |
| FCFS completes all jobs | ✅     | `test_integration.py`                         |
| Action mapper           | ✅     | `test_action.py`                              |
| Reward calculator       | ✅     | `test_reward.py`                              |
| Workload parser         | ✅     | `test_workload_parser.py`                     |

**Tổng: 8 test files, 100% PASS**

### NFR4 — Reproducibility ✅ DONE

- Cùng config + seed → kết quả giống nhau trên MockAdapter ✅

### NFR5 — Maintainability ✅ DONE

| Yêu cầu                  | Status | Evidence                              |
| ------------------------ | ------ | ------------------------------------- |
| Không hard-code          | ✅     | `config/base_config.py` — Pydantic v2 |
| top_k từ config          | ✅     | `ObservationConfig.top_k_jobs`        |
| max_time từ config       | ✅     | `EpisodeConfig.max_simulation_time`   |
| reward weights từ config | ✅     | `RewardWeights` dataclass             |
| total_cores từ config    | ✅     | `PlatformConfig.total_cores`          |
| workload từ config       | ✅     | `WorkloadConfig.trace_path`           |

### NFR6 — Usability ✅ DONE

| README section   | Status                      |
| ---------------- | --------------------------- |
| Setup môi trường | ✅ Docker + venv            |
| Chạy one episode | ✅ `examples/quickstart.py` |
| Chạy benchmark   | ✅ `train_ppo_trace.py`     |
| Chạy training    | ✅ `train_ppo_*.py`         |
| Đọc kết quả      | ✅ TensorBoard guide        |

---

## 3. Acceptance Criteria

### 12.1 Minimum Pass ✅ 9/9 ĐẠT

| #   | Tiêu chí                                       | Status                                      |
| --- | ---------------------------------------------- | ------------------------------------------- |
| 1   | Formal MDP/SMDP definition trong báo cáo       | ✅ `docs/ENV_DESIGN.md` + Handout Section 7 |
| 2   | PyBatGymEnv chạy đúng Gymnasium API            | ✅                                          |
| 3   | ToySimulatorAdapter event-driven               | ✅ `EventDrivenMockAdapter`                 |
| 4   | Obs, action space, action mask, reward rõ ràng | ✅                                          |
| 5   | Random + FCFS chạy hết episode                 | ✅                                          |
| 6   | Benchmark CSV có metric chính                  | ✅                                          |
| 7   | Unit test cơ bản                               | ✅ 8 files                                  |
| 8   | README hướng dẫn                               | ✅                                          |
| 9   | Phân tích kết quả trong báo cáo                | ✅ `PROGRESS.md` có benchmark tables        |

### 12.2 Expected Level ✅ 8/8 ĐẠT

| #   | Tiêu chí                           | Status                                         |
| --- | ---------------------------------- | ---------------------------------------------- |
| 1   | SJF hoặc Smallest-Fitting baseline | ✅ Cả hai                                      |
| 2   | Backfill-like macro-action         | ✅ `SCHEDULE_SMALLEST_FITTING` action K+1      |
| 3   | Config YAML hoàn chỉnh             | ✅ `configs/default.yaml`, `small_batsim.yaml` |
| 4   | Logging per-step + per-episode     | ✅ TensorBoard + CSV logger                    |
| 5   | Nhiều workload/platform/seed       | ✅ tiny/medium/heavy + small_platform          |
| 6   | Biểu đồ benchmark                  | ✅ TensorBoard charts                          |
| 7   | PPO baseline train/evaluate        | ✅ 4 models saved, 2M steps trained            |
| 8   | Reproducibility check              | ✅ Seed-based                                  |

### 12.3 Strong Level ⚠️ 5/6 ĐẠT

| #   | Tiêu chí                             | Status | Chi tiết                                                                  |
| --- | ------------------------------------ | ------ | ------------------------------------------------------------------------- |
| 1   | BatsimAdapter chạy workload nhỏ      | ✅     | Real BatSim 3.1.0 qua Docker ZeroMQ                                       |
| 2   | So sánh heuristic vs PPO trên BatSim | ✅     | PPO beat SJF: 58.3s vs 406.2s wait time                                   |
| 3   | Action mask + MaskablePPO            | ⚠️     | Action mask có sẵn, MaskablePPO chưa tích hợp (dùng PPO thường + penalty) |
| 4   | Ablation observation/reward          | ❌     | **CHƯA LÀM**                                                              |
| 5   | Installable package                  | ✅     | `pyproject.toml` + `pip install -e .`                                     |
| 6   | CI test / auto benchmark script      | ✅     | `scripts/run_phase1.sh`, `run_phase1b.sh`                                 |

---

## 4. Thesis Report Structure

### Chương cần viết (Section 11 Handout)

| Chương                    | Nội dung chính                                   | Status                                        |
| ------------------------- | ------------------------------------------------ | --------------------------------------------- |
| Ch.1 Tổng quan            | Bối cảnh HPC, vấn đề, mục tiêu, đóng góp         | 📝 Cần viết                                   |
| Ch.2 Nghiên cứu liên quan | HPC sched, DES, BatSim, RL, Gymnasium            | 📝 Cần viết                                   |
| Ch.3 Mô hình hóa          | MDP/SMDP, state, obs, action, transition, reward | ⚠️ Có `ENV_DESIGN.md` nhưng cần format thesis |
| Ch.4 Thiết kế hệ thống    | Architecture, modules, adapter design            | ⚠️ Có docs nhưng cần format thesis            |
| Ch.5 Hiện thực            | Tech stack, repo structure, Gym impl             | ⚠️ Có `PROGRESS.md` cần format                |
| Ch.6 Thực nghiệm          | Setup, results, analysis, reproducibility        | ⚠️ Có data, cần format thesis                 |
| Ch.7 Kết luận             | Kết quả, hạn chế, hướng phát triển               | 📝 Cần viết                                   |

### Diagrams cần có (Section 8)

| Diagram                              | Status | Ghi chú                                                  |
| ------------------------------------ | ------ | -------------------------------------------------------- |
| High-level architecture              | ⚠️     | Cần vẽ: Policy → Env → Obs/Action/Reward → Adapter → Sim |
| Module-level architecture            | ⚠️     | Cần vẽ package dependencies                              |
| Episode sequence diagram             | ⚠️     | Cần vẽ reset→obs→action→step loop                        |
| Event-driven transition diagram      | ⚠️     | EXECUTE_JOB vs WAIT transition                           |
| Benchmark comparison chart           | ✅     | TensorBoard có sẵn                                       |
| Training curve (RL)                  | ✅     | TensorBoard PPO_26 logs                                  |
| Adapter integration diagram (BatSim) | ⚠️     | Cần vẽ ZeroMQ bridge                                     |

### Tables cần có (Section 10.1)

| Bảng                                  | Status                                 |
| ------------------------------------- | -------------------------------------- |
| So sánh simulator/framework liên quan | 📝 Cần viết                            |
| Định nghĩa state components           | ⚠️ Có trong ENV_DESIGN.md              |
| Observation features                  | ✅ Có trong `observation.py` docstring |
| Action space                          | ✅ Có trong `action.py` docstring      |
| Reward components                     | ✅ Có trong `reward.py` docstring      |
| Workload configurations               | ✅ Có data                             |
| Platform configurations               | ✅ Có data                             |
| Benchmark result summary              | ✅ Có trong PROGRESS.md                |
| Acceptance criteria + mức hoàn thành  | ✅ **Checklist này**                   |

---

## 5. Research Questions Trả Lời

| RQ  | Câu hỏi                                                  | Đã trả lời? | Evidence                             |
| --- | -------------------------------------------------------- | ----------- | ------------------------------------ |
| RQ1 | Mô hình hóa HPC scheduling → decision process Gymnasium? | ✅          | EventDrivenMockAdapter, env.py       |
| RQ2 | Thiết kế obs/action/reward cho scheduling?               | ✅          | observation.py, action.py, reward.py |
| RQ3 | Tách env khỏi simulator backend?                         | ✅          | BatsimAdapter ABC + Mock + Real      |
| RQ4 | Benchmark công bằng cùng workload/platform/seed/config?  | ✅          | benchmark.py + YAML configs          |
| RQ5 | RL baseline qua Gymnasium API + benchmark pipeline?      | ✅          | PPO 2M steps, beat SJF               |

---

## 6. Experiment Design (Section 9)

### Workloads (cần ít nhất 3)

| Workload                      | Mô tả                             | Status                                  |
| ----------------------------- | --------------------------------- | --------------------------------------- |
| A — Small Balanced (tiny)     | 6 jobs, kiểm tra cơ bản           | ✅ `tiny_workload.json`                 |
| B — Queue Pressure (medium)   | 100 jobs, resource cạnh tranh     | ✅ `medium_workload.json`               |
| C — Heavy (stress test)       | 1000 jobs                         | ✅ `heavy_workload.json`                |
| Backfill Opportunity workload | Job lớn đầu queue + nhiều job nhỏ | ⚠️ Có thể tạo từ `generate_workload.py` |

### Platform settings (cần ít nhất 2)

| Platform             | Cores                      | Status                  |
| -------------------- | -------------------------- | ----------------------- |
| Small                | 4 nodes × 1 core = 4 cores | ✅ `small_platform.xml` |
| Medium (32 cores)    | 8 nodes × 4 cores = 32 cores | ✅ `medium_platform.xml` |

### Seeds (cần ít nhất 3)

| Seed set   | Status                                                     |
| ---------- | ---------------------------------------------------------- |
| 42, 43, 44 | ✅ Benchmark chạy đủ 3 seeds × 3 episodes = 216 runs |

### Benchmark output format chuẩn

| Yêu cầu                                                                     | Status                       |
| --------------------------------------------------------------------------- | ---------------------------- |
| `results/benchmark_summary.csv`                                             | ✅ 216 rows, 19 columns       |
| Columns chuẩn (experiment_id, policy, workload, platform, seed, metrics...) | ✅ Đầy đủ theo Handout        |

---

## 7. Defense Q&A Chuẩn Bị (Section 15)

| Câu hỏi                                    | Đã chuẩn bị?                     |
| ------------------------------------------ | -------------------------------- |
| Q1. Vì sao không yêu cầu RL policy hội tụ? | ✅ Trọng tâm = env + benchmark   |
| Q2. MDP của đề tài là gì?                  | ✅ Có formal spec                |
| Q3. Vì sao gọi là SMDP?                    | ✅ Event-driven time jump        |
| Q4. Vì sao cần ToySimulator?               | ✅ Risk reduction                |
| Q5. Benchmark công bằng?                   | ✅ Same config/seed/workload     |
| Q6. PyBatGym khác simulator?               | ✅ Mediation layer               |
| Q7. Reward có phải objective cuối?         | ✅ Không, metric mới là tiêu chí |

---

## 8. Việc CẦN LÀM (TODO)

### 🔴 Ưu tiên cao (ảnh hưởng điểm)

| #   | Task                                                     | Mức ảnh hưởng           |
| --- | -------------------------------------------------------- | ----------------------- |
| 1   | ~~Tạo `results/benchmark_summary.csv` chuẩn format Handout~~ | ✅ DONE                 |
| 2   | ~~Tạo medium platform (32 cores)~~                           | ✅ DONE                 |
| 3   | ~~Chạy benchmark với 3 seeds (42,43,44)~~                    | ✅ DONE                 |
| 4   | Vẽ 4 diagrams: architecture, module, sequence, adapter   | Thesis report           |
| 5   | ~~Tích hợp MaskablePPO (sb3-contrib đã có trong deps)~~      | ✅ DONE — `examples/train_maskable_ppo.py` + `env.action_masks()` |

### 🟡 Ưu tiên trung bình

| #   | Task                                                   | Mức ảnh hưởng     |
| --- | ------------------------------------------------------ | ----------------- |
| 6   | Ablation study: thử bỏ resource features / thay reward | Strong Level      |
| 7   | ~~Tạo Backfill-specific workload (Workload C)~~            | ✅ DONE — `data/workloads/backfill_workload.json` |
| 8   | ~~Bổ sung `action_mask` vào `info` dict (FR1 compliance)~~ | ✅ DONE — `env.py:183` |
| 9   | SWF Parser cho NASA/KTH traces (Priority 3 roadmap)    | Generalization    |

### 🟢 Ưu tiên thấp (nice-to-have)

| #   | Task                                             | Mức ảnh hưởng |
| --- | ------------------------------------------------ | ------------- |
| 10  | Viết toàn bộ 7 chương thesis (format LaTeX/Word) | Report        |
| 11  | Tạo slide bảo vệ                                 | Defense       |
| 12  | Demo script one-click                            | Usability     |

---

## 9. Những Gì ĐÃ VƯỢT Yêu Cầu

| Thành tựu                      | Chi tiết                                                              |
| ------------------------------ | --------------------------------------------------------------------- |
| **BatSim tích hợp hoàn chỉnh** | Real BatSim 3.1.0 qua Docker + ZeroMQ (handout chỉ yêu cầu "nếu kịp") |
| **PPO 2M steps**               | Converged, beat SJF 85.6% (waiting time), explained_var=0.9985        |
| **Hybrid Mock+Real training**  | Train trên Mock, eval trên Real BatSim                                |
| **Batch synchronization**      | Giải quyết deadlock BatSim, multi-job scheduling per round-trip       |
| **3 training phases**          | Phase 1 (200k), Phase 1b (150k convergence), Phase 2.2 (2M)           |
| **BACKFILL macro-action**      | `SCHEDULE_SMALLEST_FITTING` — vượt FR4 minimum                        |
| **Workload generator**         | `scripts/generate_workload.py` — tạo workload tùy chỉnh               |
| **Docker Compose full stack**  | Shell + BatSim + TensorBoard containers                               |

---

_Đối chiếu Handout Thesis Requirements PyBatGym với source code thực tế_
