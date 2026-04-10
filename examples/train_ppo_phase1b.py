"""Priority 1b: PPO with Convergence-Based Early Stopping.

Key improvements over Phase 1:
  - num_jobs=100 per episode (shorter episodes -> more episodes -> more exploration)
  - No fixed total_timesteps: training stops when reward has CONVERGED
    (stable sliding window) — learning algorithm is "done"
  - Hard cap of 2M steps to prevent infinite run
  - Best model auto-saved whenever performance improves
  - ConvergenceCallback with Coefficient-of-Variation + trend detection

Convergence definition (all 3 must hold):
  1. CV  = std(rewards[-W:]) / |mean(rewards[-W:])| < threshold   (stable, not noisy)
  2. slope of rewards[-W:] ≈ 0                                     (not still improving)
  3. minimum N episodes seen (avoid premature stop)

Usage (inside Docker):
    bash scripts/run_phase1b.sh
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv
from pybatgym.plugins.tensorboard_logger import TensorBoardLoggerPlugin
from pybatgym.plugins.logger import CSVLoggerPlugin
from pybatgym.plugins.benchmark import run_baseline, sjf_policy, easy_backfilling_policy, fcfs_policy


# ── Convergence Callback ────────────────────────────────────────────────────

class ConvergenceCallback(BaseCallback):
    """Stop training when episode reward has converged to a stable value.

    Convergence is detected when ALL of the following hold for the last
    `window` completed episodes:

    1. Coefficient of Variation (CV) = std / |mean| < `cv_threshold`
       → reward is no longer noisy / jumping around

    2. |linear_slope| / |mean| < `slope_threshold`
       → reward is no longer trending up (agent stopped improving)

    3. Number of completed episodes >= `min_episodes`
       → enough data to make a reliable judgement

    A hard cap at `max_timesteps` prevents runaway training.

    Args:
        window:           Number of recent episodes to evaluate.
        cv_threshold:     Max allowed Coefficient of Variation (default 5%).
        slope_threshold:  Max allowed relative slope per episode (default 0.5%).
        min_episodes:     Minimum episodes before checking convergence.
        max_timesteps:    Hard stop (safety cap).
        check_every:      How many timesteps between convergence checks.
        save_path:        If provided, save best model here on each improvement.
        verbose:          0=silent, 1=convergence events, 2=every check.
    """

    def __init__(
        self,
        window: int = 50,
        cv_threshold: float = 0.05,
        slope_threshold: float = 0.005,
        min_episodes: int = 100,
        max_timesteps: int = 2_000_000,
        check_every: int = 2000,
        save_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.window = window
        self.cv_threshold = cv_threshold
        self.slope_threshold = slope_threshold
        self.min_episodes = min_episodes
        self.max_timesteps = max_timesteps
        self.check_every = check_every
        self.save_path = save_path

        # Internal tracking
        self._episode_rewards: list[float] = []
        self._n_episodes: int = 0
        self._last_check: int = 0
        self._best_mean: float = -np.inf
        self._start_time: float = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # ── 1. Collect episode rewards from step info ──────────────────────
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                # PyBatGymEnv puts cumulative_reward in info
                ep_reward = info.get("cumulative_reward", 0.0)
                self._episode_rewards.append(ep_reward)
                self._n_episodes += 1

                # Track best and save model
                if self.save_path and self._n_episodes >= self.min_episodes:
                    recent_mean = float(np.mean(self._episode_rewards[-self.window:]))
                    if recent_mean > self._best_mean:
                        self._best_mean = recent_mean
                        self.model.save(self.save_path + "_best")
                        if self.verbose >= 2:
                            elapsed = time.time() - self._start_time
                            print(
                                f"  [Best] ep={self._n_episodes}  "
                                f"mean={recent_mean:.4f}  "
                                f"steps={self.num_timesteps:,}  "
                                f"t={elapsed:.0f}s"
                            )

        # ── 2. Hard cap ─────────────────────────────────────────────────── 
        if self.num_timesteps >= self.max_timesteps:
            elapsed = time.time() - self._start_time
            print(
                f"\n[STOP] Hard cap reached: {self.max_timesteps:,} steps. "
                f"Episodes={self._n_episodes}  "
                f"Elapsed={elapsed:.0f}s"
            )
            return False

        # ── 3. Too early to check ─────────────────────────────────────────
        if self._n_episodes < self.min_episodes:
            return True

        # ── 4. Only check every N steps ───────────────────────────────────
        if self.num_timesteps - self._last_check < self.check_every:
            return True
        self._last_check = self.num_timesteps

        # ── 5. Convergence check ──────────────────────────────────────────
        recent = self._episode_rewards[-self.window:]
        if len(recent) < self.window:
            return True

        mean_r = float(np.mean(recent))
        std_r = float(np.std(recent))
        ref = max(abs(mean_r), 1e-8)

        cv = std_r / ref

        # Trend: linear slope over the window (normalized)
        xs = np.arange(len(recent), dtype=float)
        slope = float(np.polyfit(xs, recent, 1)[0])  # reward/episode
        slope_rel = abs(slope) / ref

        converged = cv < self.cv_threshold and slope_rel < self.slope_threshold

        if self.verbose >= 1:
            status = "CONVERGED" if converged else "training"
            elapsed = time.time() - self._start_time
            print(
                f"  [{self.num_timesteps:>9,}] eps={self._n_episodes:>4}  "
                f"mean={mean_r:>8.3f}  cv={cv:.3f}  slope={slope:>+.4f}  "
                f"fps={self.num_timesteps/max(elapsed,1):.0f}  "
                f"[{status}]"
            )

        if converged:
            elapsed = time.time() - self._start_time
            print(
                f"\n{'='*70}\n"
                f"[CONVERGED] Training complete!\n"
                f"  Episodes   : {self._n_episodes}\n"
                f"  Steps      : {self.num_timesteps:,}\n"
                f"  Mean reward: {mean_r:.4f}\n"
                f"  Std reward : {std_r:.4f}\n"
                f"  CV         : {cv:.4f} (threshold={self.cv_threshold})\n"
                f"  Slope      : {slope:+.6f} (threshold={self.slope_threshold*ref:+.6f})\n"
                f"  Elapsed    : {elapsed:.0f}s\n"
                f"{'='*70}"
            )
            return False

        return True


# ── Env factory ─────────────────────────────────────────────────────────────

def make_env(trace_path: str) -> PyBatGymEnv:
    """Phase 1b environment: 100 jobs/episode for dense exploration."""
    config = PyBatGymConfig()

    # Platform: scarce resources create scheduling contention
    config.platform.total_nodes = 4
    config.platform.cores_per_node = 2  # 8 cores total

    # Workload: 100 jobs/episode → ~900 steps/episode → 2M steps = ~2,200 episodes
    config.workload.source = "trace"
    config.workload.trace_path = trace_path
    config.workload.num_jobs = 100

    # Episode limits
    config.episode.max_simulation_time = 30_000.0
    config.episode.max_steps = 1500  # generous per-episode cap

    # Observation
    config.observation.top_k_jobs = 10
    config.observation.max_queue_length = 150
    config.observation.max_waiting_time = 3000.0

    # Reward: strong waiting_time penalty → agent learns to schedule promptly
    config.reward_weights.utilization   = 0.3
    config.reward_weights.waiting_time  = 0.5
    config.reward_weights.slowdown      = 0.15
    config.reward_weights.throughput    = 0.05
    config.reward_type = "hybrid"

    config.plugins = [
        TensorBoardLoggerPlugin(log_dir="logs/tensorboard_phase1b"),
        CSVLoggerPlugin(output_dir="logs/csv_phase1b"),
    ]

    return PyBatGymEnv(config=config)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_agent(env: PyBatGymEnv, model: PPO, num_episodes: int = 5) -> dict:
    """Run trained agent deterministically and collect metrics."""
    totals: dict[str, float] = {k: 0.0 for k in
                                 ("wait", "sd", "util", "reward", "n")}

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=200 + ep)
        done, ep_reward = False, 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        completed = env._adapter.get_completed_jobs()
        if completed:
            makespan   = env._adapter.get_current_time()
            total_cores = env.unwrapped._state["resource"].total_cores

            totals["wait"]   += sum(j.waiting_time for j in completed) / len(completed)
            totals["sd"]     += sum(j.bounded_slowdown for j in completed) / len(completed)
            if makespan > 0 and total_cores > 0:
                busy = sum(j.actual_runtime * j.requested_resources for j in completed)
                totals["util"] += busy / (makespan * total_cores)
            totals["reward"] += ep_reward
            totals["n"] += 1

    n = max(totals["n"], 1)
    return {
        "avg_waiting_time": totals["wait"] / n,
        "avg_slowdown":     totals["sd"]   / n,
        "avg_utilization":  totals["util"] / n,
        "avg_reward":       totals["reward"] / n,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  PyBatGym Priority 1b: PPO with Convergence-Based Early Stopping")
    print("=" * 70)

    trace_path = Path("/workspace/data/workloads/medium_workload.json")
    if not trace_path.exists():
        print(f"[ERROR] Workload not found: {trace_path}")
        print("Run: python scripts/generate_workload.py --preset medium")
        return

    # ── Create environment ─────────────────────────────────────────────
    env = make_env(str(trace_path))
    cfg = env.unwrapped._config
    total_cores = cfg.platform.total_cores

    print(f"\n[Config]")
    print(f"  Workload    : {trace_path.name}")
    print(f"  Platform    : {cfg.platform.total_nodes} nodes x "
          f"{cfg.platform.cores_per_node} cores = {total_cores} total")
    print(f"  Jobs/ep     : {cfg.workload.num_jobs}  (was 300 in Phase 1)")
    print(f"  Reward      : util={cfg.reward_weights.utilization}, "
          f"wait={cfg.reward_weights.waiting_time}, "
          f"slowdown={cfg.reward_weights.slowdown}")
    print()

    # ── Heuristic baselines ────────────────────────────────────────────
    print("[1/4] Heuristic Baselines (3 episodes each)...")
    fcfs_m = run_baseline(env, fcfs_policy,             num_episodes=3)
    sjf_m  = run_baseline(env, sjf_policy,              num_episodes=3)
    easy_m = run_baseline(env, easy_backfilling_policy, num_episodes=3)
    print(f"  FCFS  wait={fcfs_m['avg_waiting_time']:.1f}  "
          f"slowdown={fcfs_m['avg_slowdown']:.2f}  "
          f"util={fcfs_m['avg_utilization']:.1%}")
    print(f"  SJF   wait={sjf_m['avg_waiting_time']:.1f}  "
          f"slowdown={sjf_m['avg_slowdown']:.2f}  "
          f"util={sjf_m['avg_utilization']:.1%}")
    print(f"  EASY  wait={easy_m['avg_waiting_time']:.1f}  "
          f"slowdown={easy_m['avg_slowdown']:.2f}  "
          f"util={easy_m['avg_utilization']:.1%}")

    # ── Build PPO model ────────────────────────────────────────────────
    print("\n[2/4] Building PPO model...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        n_steps=512,          # larger buffer: ~0.5 episodes per rollout
        batch_size=128,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="logs/tensorboard_phase1b",
    )
    print(f"  Policy: MultiInputPolicy  n_steps=512  batch=128  lr=3e-4")

    # ── Convergence-based training ─────────────────────────────────────
    max_steps    = 2_000_000
    window       = 50          # look at last 50 episodes
    cv_thresh    = 0.05        # 5% coefficient of variation → stable
    slope_thresh = 0.005       # slope < 0.5% of |mean| per episode → plateau
    min_eps      = 100         # wait for 100 episodes before checking

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    save_path = str(model_dir / "ppo_phase1b")

    print(f"\n[3/4] Training (convergence-based stopping)...")
    print(f"  Max steps     : {max_steps:,} (hard cap)")
    print(f"  Window        : {window} episodes")
    print(f"  CV threshold  : {cv_thresh:.1%}  (stop when reward variance <5%)")
    print(f"  Slope thresh  : {slope_thresh:.1%}  (stop when trend <0.5%/ep)")
    print(f"  Min episodes  : {min_eps}")
    print(f"  Best model    : {save_path}_best.zip")
    print(f"\n{'─'*70}")
    print(f"  {'Steps':>12}  {'Eps':>4}  {'Mean':>8}  {'CV':>6}  "
          f"{'Slope':>8}  {'FPS':>5}  Status")
    print(f"{'─'*70}")

    callback = ConvergenceCallback(
        window=window,
        cv_threshold=cv_thresh,
        slope_threshold=slope_thresh,
        min_episodes=min_eps,
        max_timesteps=max_steps,
        check_every=5000,
        save_path=save_path,
        verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=max_steps, callback=callback, reset_num_timesteps=True)
    train_time = time.time() - t0

    # Save final model
    model.save(save_path)
    print(f"\n  Final model saved: {save_path}.zip")
    print(f"  Training time  : {train_time:.0f}s ({train_time/60:.1f} min)")
    print(f"  Total steps    : {model.num_timesteps:,}")
    print(f"  Total episodes : {callback._n_episodes}")

    # ── Evaluate trained PPO ───────────────────────────────────────────
    print(f"\n[4/4] Evaluating Trained PPO (5 episodes, deterministic)...")
    ppo_m = evaluate_agent(env, model, num_episodes=5)

    # Try loading best model too
    best_path = save_path + "_best.zip"
    if Path(best_path).exists():
        best_model = PPO.load(best_path, env=env)
        best_m = evaluate_agent(env, best_model, num_episodes=5)
    else:
        best_m = ppo_m

    # ── Final table ────────────────────────────────────────────────────
    print(f"\n{'='*84}")
    print(f"{'Metric':<25} | {'FCFS':<12} | {'SJF':<12} | {'EASY BF':<12} | "
          f"{'PPO final':<12} | {'PPO best':<12}")
    print(f"{'─'*84}")

    rows = [
        ("Avg Waiting Time (s)", "avg_waiting_time", False),  # lower=better
        ("Avg Slowdown",         "avg_slowdown",     False),
        ("Avg Utilization (%)",  "avg_utilization",  True),   # higher=better
    ]

    for label, key, higher_is_better in rows:
        vals = [fcfs_m[key], sjf_m[key], easy_m[key], ppo_m[key], best_m[key]]
        best_idx = vals.index(max(vals) if higher_is_better else min(vals))

        if "Util" in label:
            strs = [f"{v*100:.1f}%" for v in vals]
        else:
            strs = [f"{v:.2f}" for v in vals]

        strs[best_idx] = f"*{strs[best_idx]}*"
        print(f"{label:<25} | {strs[0]:<12} | {strs[1]:<12} | {strs[2]:<12} | "
              f"{strs[3]:<12} | {strs[4]:<12}")

    print(f"\n  PPO final avg reward : {ppo_m['avg_reward']:.4f}")
    print(f"  PPO best  avg reward : {best_m['avg_reward']:.4f}")
    print(f"\n  (* = best for that metric)")
    print(f"  View curves: tensorboard --logdir logs/tensorboard_phase1b")

    env.close()


if __name__ == "__main__":
    main()
