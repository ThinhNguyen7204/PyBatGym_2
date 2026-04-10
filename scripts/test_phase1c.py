"""Priority 1c smoke test: verify obs/action space shapes after fixes."""
from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv

cfg = PyBatGymConfig()
cfg.platform.total_nodes = 4
cfg.platform.cores_per_node = 2
cfg.workload.source = "trace"
cfg.workload.trace_path = "/workspace/data/workloads/medium_workload.json"
cfg.workload.num_jobs = 20

env = PyBatGymEnv(config=cfg)
obs, _ = env.reset(seed=42)

K = cfg.observation.top_k_jobs
total_cores = cfg.platform.total_cores

# ── Observation checks ──────────────────────────────────────────────────────
feat_expected = 5 + 4 * K + 6  # global + job_queue + resource = 51
mask_expected = K + 2           # K jobs + WAIT + BACKFILL = 12

assert obs["features"].shape == (feat_expected,), (
    f"features shape {obs['features'].shape} != ({feat_expected},)"
)
assert obs["action_mask"].shape == (mask_expected,), (
    f"action_mask shape {obs['action_mask'].shape} != ({mask_expected},)"
)

# ── Action space check ──────────────────────────────────────────────────────
assert env.action_space.n == K + 2, (
    f"action_space.n={env.action_space.n} != {K+2}"
)

# ── Feature sanity: no NaN / all in [0,1] ──────────────────────────────────
import numpy as np
assert not np.any(np.isnan(obs["features"])), "NaN in features!"
assert np.all(obs["features"] >= 0) and np.all(obs["features"] <= 1), (
    f"features out of [0,1]: min={obs['features'].min()}, max={obs['features'].max()}"
)
assert not np.any(np.isnan(obs["action_mask"])), "NaN in action_mask!"

# ── Job feature index check: [offset+3] is now BSD, not duplicate ───────────
# With pending jobs present, BSD feature (offset+3) should not == wait (offset+0)
# for at least one job slot (they'd only be equal if wait=0 and runtime=0)
# Just verify both exist and are in range
print(f"  Job slot 0 features: {obs['features'][5:9]}")
print(f"    [0]=wait={obs['features'][5]:.3f}  [1]=walltime={obs['features'][6]:.3f}  "
      f"[2]=cores={obs['features'][7]:.3f}  [3]=bsd_norm={obs['features'][8]:.3f}")

# ── Resource features check (OBS-2) ─────────────────────────────────────────
res_start = 5 + 4 * K
print(f"  Resource features: {obs['features'][res_start:res_start+6]}")
print(f"    [0]=free_ratio  [1]=jobs_fitting  [2]=queue_urgency  "
      f"[3]=min_walltime  [4]=max_walltime  [5]=fragmentation")

# ── Step a few times ─────────────────────────────────────────────────────────
for i in range(10):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(int(action))
    if term or trunc:
        break

env.close()

print()
print("=" * 60)
print(f"  K = {K}")
print(f"  total_cores (platform) = {total_cores}")
print(f"  obs['features'].shape  = {feat_expected}  [PASS]")
print(f"  obs['action_mask'].shape = {mask_expected} [PASS]  (K+WAIT+BACKFILL)")
print(f"  action_space.n = {K+2}  [PASS]")
print(f"  No NaN, all in [0,1]    [PASS]")
print("=" * 60)
print("ALL PRIORITY 1c CHECKS PASSED")
