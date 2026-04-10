# PyBatGym — Action Space, Observation Space & Reward

> **File:** `docs/ENV_DESIGN.md`
> **Cập nhật:** 2026-04-10
> **Áp dụng cho:** `PyBatGymEnv` v2 — `pybatgym/env.py`, `observation.py`, `action.py`, `reward.py`

---

## 1. Action Space

### Định nghĩa

```python
action_space = gym.spaces.Discrete(K + 1)
# K = config.observation.top_k_jobs = 10 (default)
# → Discrete(11) : 11 hành động rời rạc
```

### Ý nghĩa từng action

| Action index | Ý nghĩa | Ghi chú |
|---|---|---|
| `0` | Schedule **job 0** (chờ lâu nhất trong top-K) | Hợp lệ nếu đủ cores |
| `1` | Schedule **job 1** | |
| `...` | ... | |
| `9` | Schedule **job 9** | |
| `10` | **WAIT** — không làm gì bước này | Luôn hợp lệ |
| `11` | **SCHEDULE_SMALLEST_FITTING** — backfill | Chọn job tốn ít core nhất nằm trong số cores đang rảnh. Nếu không có → WAIT |

### Top-K Queue — thứ tự sắp xếp

Trong mỗi step, `top_k_jobs` = 10 pending jobs được chọn theo:

```python
# observation.py — _build_job_queue()
sorted_jobs = sorted(pending_jobs,
                     key=lambda j: current_time - j.submit_time,
                     reverse=True)  # chờ lâu nhất → index 0
```

> Job nào **chờ lâu nhất** ở index 0. Đây là FCFS-bias — có thể thay bằng SJF-sort để guide PPO tốt hơn.

### Action Mask

```python
action_mask: Box(shape=(K+2,), low=0, high=1)
# 1.0 = hành động hợp lệ
# 0.0 = hành động không hợp lệ (job cần nhiều cores hơn hiện có)
# mask[K] (WAIT) luôn = 1.0
# mask[K+1] (SMALLEST_FITTING) = 1.0 nếu có ÍT NHẤT 1 job trong queue đủ điều kiện chạy
```

```python
# observation.py — _build_action_mask()
for i, job in enumerate(top_k):
    if resource.can_allocate(job.requested_resources):
        mask[i] = 1.0    # đủ cores → valid
    # else: 0.0          # không đủ cores → invalid
mask[-1] = 1.0            # WAIT luôn valid
```

PPO sử dụng `MultiInputPolicy` — mask được truyền qua `action_mask` key trong observation dict, ngăn agent chọn action không hợp lệ trong forward pass.

### Sơ đồ action flow

```
pending_jobs = [J5, J2, J8, J1, ...]   (30 jobs)
        ↓ sort by wait_time desc
top_10  = [J5(wait=400s), J2(wait=300s), J8(wait=200s), ...]
        ↓ build mask
mask    = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0,  1]
           ^ ^ ^                            ^ WAIT
           valid   J8 cần 4c, chỉ còn 2c → invalid

Agent chọn action=1 → schedule J2 → adapter.step(ScheduleCommand(J2))
```

### Known Issues / Cải tiến tiềm năng

| Vấn đề | Tác động | Giải pháp tiềm năng |
|---|---|---|
| Không có "backfill" action riêng | PPO không thể học EASY Backfilling | Thêm action `SCHEDULE_SMALLEST_FITTING` |
| Top-K sort by wait → FCFS-bias | PPO bị dẫn dắt schedule theo FCFS thứ tự | Sort by runtime ascending (SJF-bias) hoặc multi-sort |
| K=10 cố định | Queue > 10 jobs → mất thông tin | Tăng K hoặc dùng attention mechanism |

---

## 2. Observation Space

### Định nghĩa

```python
observation_space = gym.spaces.Dict({
    "features":     Box(low=0, high=1, shape=(51,), dtype=float32),
    "action_mask":  Box(low=0, high=1, shape=(11,), dtype=float32),
})
# 51 = 5 (global) + 4×10 (jobs) + 6 (resource)
```

### Layout chi tiết — vector `features[51]`

```
Index   | Name                      | Formula                      | Range
--------|---------------------------|------------------------------|-------
GLOBAL (5)
[0]     | time_progress             | current_time / max_sim_time  | [0,1]
[1]     | queue_fill                | len(pending) / max_queue(150)| [0,1]
[2]     | utilization               | busy_cores / total_cores     | [0,1]
### 3.1 Job Features (4 dims per slot, K=10 slots)
For each top-K job (sorted by wait time):
*   `[offset + 0]`: Normalised `waiting_time` (time since submission).
*   `[offset + 1]`: Normalised `requested_walltime`.
*   `[offset + 2]`: Normalised `requested_resources` (cores needed).
*   `[offset + 3]`: Normalised `bounded_slowdown` (proxy for urgency to schedule). **[FIXED in Phase 1c]**

---

### 3.2 Resource Features (6 dims) **[FIXED in Phase 1c]**
Tất cả duplicate/placeholder đã được gỡ bỏ, thay bằng 6 chỉ số có ý nghĩa:
*   `[45]`: Free cores ratio (`free_cores / total_cores`).
*   `[46]`: Fraction of top-K jobs that fit in free cores right now.
*   `[47]`: Queue urgency (Fraction of pending jobs that *can* run but are forced to wait longer than their requested walltime).
*   `[48]`: Min pending walltime (normalized).
*   `[49]`: Max pending walltime (normalized, proxy for expected backfill shadow time).
*   `[50]`: Fragmentation metric (`free_cores % min_cores_requested`).

> **Note:** Việc này cung cấp cho PPO một bức tranh toàn cảnh về độ "nén" của queue và tài nguyên rảnh thực tế để tối ưu Phase Backfilling.

### Cách normalize

Tất cả features được normalize về `[0, 1]` bằng:
```python
def _normalize(value, max_value):
    return min(1.0, max(0.0, value / max_value))
```

Các hằng số normalize:
```python
max_waiting_time  = 3000.0   # giây
max_queue_length  = 150      # số job
max_bounded_sd    = 100.0
max_cores_per_job = 64
max_sim_time      = 30_000.0 # giây (config.episode.max_simulation_time)
```

### Bounded Slowdown (BSD)

Metric quan trọng trong observation và reward:

```
BSD = max(actual_runtime, 10) / (max(actual_runtime, 10) + waiting_time)
                     ^
             tránh chia cho 0 với job cực ngắn (floor = 10s)

BSD = 1.0  → không phải chờ gì cả (ideal)
BSD > 1.0  → đã chờ thêm
BSD = 5.0  → tổng thời gian = 5× thời gian chạy → chờ = 4× runtime
```

---

## 3. Reward Function

### Chế độ `hybrid` (đang dùng)

```
Total reward = Σ(step_rewards) + episode_reward_at_done
```

### Step Reward (mỗi lần gọi `step()`)

```python
r = 0.0

# ① Utilization improvement (quan trọng khi schedule job mới)
delta_util = utilization_now - utilization_prev
r += delta_util × w_util           # w_util = 0.3

# ② Scheduling bonus / WAIT penalty
if action == EXECUTE_JOB:
    r += SCHEDULE_BONUS            # = +0.1
elif action == WAIT and pending_jobs:
    r += INVALID_PENALTY           # = -0.05 (phạt WAIT khi có thể làm)

# ③ Completion penalty (mỗi job kết thúc trong step này)
for job in completed_this_step:
    r -= w_wait × normalize(job.waiting_time, 1000)     # w_wait = 0.5
    r -= w_sd   × normalize(job.bounded_sd - 1, 100)    # w_sd   = 0.15

# ④ Idle penalty (cores trống nhưng có job chờ)
if free_cores > 0 and pending_jobs:
    idle_ratio = free_cores / total_cores
    r -= IDLE_PENALTY_FACTOR × idle_ratio               # factor = 0.01
```

### Episode Reward (chỉ khi `terminated` hoặc `truncated`)

```python
avg_wait = mean(j.waiting_time for j in completed_jobs)
avg_sd   = mean(j.bounded_slowdown for j in completed_jobs)
throughput = len(completed_jobs) / makespan
utilization = total_core_seconds / (makespan × config.platform.total_cores) # [FIXED build-in Phase 1c]

r_ep = (
    + w_util       × utilization              # 0.3
    + w_throughput × normalize(throughput, 1) # 0.05
    - w_wait       × normalize(avg_wait, 1000)# 0.5
    - w_sd         × normalize(avg_sd-1, 100) # 0.15
)
```

> **Note:** Phase 1c đã lấy chuẩn `config.platform.total_cores` thay vì hardcode 64, đảm bảo reward luôn chuẩn xác trên các scale platform khác nhau.

### Cấu hình weights (Phase 1b)

```python
reward_weights = RewardWeights(
    utilization  = 0.3,   # thưởng dùng nhiều cores
    waiting_time = 0.5,   # ← NẶNG NHẤT — signal chính cho PPO
    slowdown     = 0.15,  # thưởng job không bị delay quá nhiều
    throughput   = 0.05,  # thưởng hoàn thành nhiều job
)
reward_type = "hybrid"
```

### Ví dụ episode reward thực tế

```
Phase 1b result: mean_reward = -79.3

Phân tích:
  Avg waiting time = 331.68s → normalize(331.68, 1000) = 0.332
  -wait penalty per step ≈ -0.5 × 0.332 = -0.166 per completed job
  100 jobs/episode → total wait penalty ≈ -16.6

  Avg slowdown = 12.06 → (12.06-1)/100 = 0.111
  -slowdown per job ≈ -0.15 × 0.111 = -0.0167 per job
  100 jobs → total slowdown penalty ≈ -1.67

  Step penalties (idle + WAIT) ≈ tiểu số qua ~900 steps

  → Phần lớn reward âm đến từ waiting_time penalty (đúng với thiết kế)
```

---

## 4. Tổng Quan Config Hiện Tại (Phase 1b)

```python
PyBatGymConfig(
    mode = "mock",

    platform = PlatformConfig(
        total_nodes    = 4,
        cores_per_node = 2,     # 8 total cores
    ),

    workload = WorkloadConfig(
        source     = "trace",
        trace_path = "data/workloads/medium_workload.json",
        num_jobs   = 100,       # per episode
    ),

    episode = EpisodeConfig(
        max_simulation_time = 30_000,  # s
        max_steps           = 1_500,
    ),

    observation = ObservationConfig(
        top_k_jobs       = 10,
        max_queue_length = 150,
        max_waiting_time = 3000,
    ),

    reward_weights = RewardWeights(
        utilization  = 0.30,
        waiting_time = 0.50,   # dominant signal
        slowdown     = 0.15,
        throughput   = 0.05,
    ),
    reward_type = "hybrid",
)

PPO hyperparameters:
    n_steps       = 512      # ~0.5 episode per rollout
    batch_size    = 128
    learning_rate = 3e-4
    ent_coef      = 0.01     # entropy bonus — exploration
    clip_range    = 0.2
    n_epochs      = 10
    gamma         = 0.99     # long-term value
    gae_lambda    = 0.95
```

---

## 5. Cải Tiến Đề Xuất

### Short-term (Priority 1b → 1c)

| ID | Vấn đề | Tình trạng |
|---|---|---|
| OBS-1 | Feature [8+4i] duplicate | **[FIXED]** Đã đổi thành `bounded_slowdown_norm` |
| OBS-2 | Resource placeholders | **[FIXED]** Đã thêm tỷ lệ fitting, queue urgency, min/max walltime pending |
| ACT-1 | Backfill awareness | **[FIXED]** Đã thêm `SCHEDULE_SMALLEST_FITTING` |
| RWD-1 | `max_core_seconds` hardcode 64 | **[FIXED]** Đã sync với `config.platform.total_cores` |
| RWD-2 | Waiting penalty tác động trễ (khi job done) | Thêm per-step incremental wait penalty (Phase 2?) |

### Long-term

| ID | Giải pháp |
|---|---|
| OBS-3 | Thêm job GPU/memory features (multi-resource) |
| ACT-2 | Continuous action space (cho large K) |
| RWD-3 | Curriculum learning: bắt đầu với reward đơn giản, tăng dần complexity |
| ENV-1 | Vectorized environments (N env song song) → tăng fps N× |

---

*Tạo: 2026-04-10 | PyBatGym v2*
