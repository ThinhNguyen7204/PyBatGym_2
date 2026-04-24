# PLAN: Real BatSim Training — Implementation

## Tóm tắt (đã xác nhận)
- BatSim **3.1.0** + pybatsim **3.1.0** — hoàn toàn tương thích, không cần batprotocol
- `RealBatsimAdapter` đã hoạt động với `test_real.py`
- Platform XML: `data/platforms/small_platform.xml` (4 nodes: Jupiter, Fafard, Ginette, Bourassa)

## Kiến trúc: Hybrid Training

```
PPO Training (Mock, nhanh) → 2M steps
  ↓ mỗi eval_freq steps
  Real BatSim eval (1 episode) → log Real/* vào TensorBoard
  ↓ sau khi hội tụ
Final validation (Real BatSim, N episodes)
```

## Công việc

### [x] Tạo batsim_data/platforms/small_platform.xml
→ Done (file đã tạo, nhưng thực ra platform đúng ở data/platforms/)

### [ ] Fix batsim_start.sh — sửa đường dẫn platform
- `/workspace/batsim_data/platforms/small_platform.xml` → `/workspace/data/platforms/small_platform.xml`
- Thêm env var BATSIM_PLATFORM, BATSIM_WORKLOAD để override

### [ ] Tạo RealEvalCallback
- File: `pybatgym/callbacks/real_eval_callback.py`
- Mỗi `eval_freq` steps: chạy 1 episode với RealBatsimAdapter
- Log `Real/avg_waiting_time`, `Real/utilization`, `Real/avg_slowdown`
- Graceful skip nếu BatSim không available

### [ ] Tạo train_ppo_real_eval.py
- File: `examples/train_ppo_real_eval.py`
- Mock training + RealEvalCallback
- Final validation bằng RealBatsimAdapter

### [ ] Export callback trong __init__.py
