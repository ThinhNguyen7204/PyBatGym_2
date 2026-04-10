"""Phase 1: Train PPO on Large Workload — Mock Mode.

Goal: Demonstrate RL (PPO) can outperform heuristics (SJF, EASY Backfilling)
when scheduling contention is high (many jobs, few nodes).

Key changes from train_ppo_trace.py:
  - 500 jobs  (medium_workload.json) instead of 6
  - 4 nodes x 2 cores = 8 total cores (creates bottleneck: jobs request 1-8 cores)
  - 200k timesteps (enough to converge; increase to 500k for better results)
  - Reward weights tuned: heavy waiting_time penalty forces aggressive scheduling
  - Proper baseline comparison with multiple episodes for statistical significance
"""

import os
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv
from pybatgym.plugins.tensorboard_logger import TensorBoardLoggerPlugin
from pybatgym.plugins.logger import CSVLoggerPlugin
from pybatgym.plugins.benchmark import run_baseline, sjf_policy, easy_backfilling_policy, fcfs_policy


class ProgressCallback(BaseCallback):
    """Print training progress every N steps."""

    def __init__(self, print_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self._print_every = print_every
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self._print_every:
            elapsed = time.time() - self._start_time
            fps = self.num_timesteps / max(elapsed, 0.01)
            print(
                f"  [{self.num_timesteps:>7,} steps] "
                f"elapsed={elapsed:.0f}s  fps={fps:.0f}  "
                f"ep_reward={self._safe_mean('rollout/ep_rew_mean'):.4f}"
            )
            self._last_print = self.num_timesteps
        return True

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _safe_mean(self, key: str) -> float:
        try:
            return self.logger.name_to_value.get(key, 0.0)
        except Exception:
            return 0.0


def make_env(trace_path: str, total_nodes: int = 4, cores_per_node: int = 2):
    """Create a configured PyBatGymEnv for Phase 1 training."""
    config = PyBatGymConfig()

    # Platform: fewer resources than jobs need -> contention
    config.platform.total_nodes = total_nodes
    config.platform.cores_per_node = cores_per_node

    # Workload: large trace
    config.workload.source = "trace"
    config.workload.trace_path = trace_path
    config.workload.num_jobs = 300  # limit per episode to keep episodes reasonable

    # Episode limits
    config.episode.max_simulation_time = 50000.0  # long enough for 300 jobs
    config.episode.max_steps = 3000               # prevent infinite loops

    # Observation
    config.observation.top_k_jobs = 10
    config.observation.max_queue_length = 300
    config.observation.max_waiting_time = 5000.0

    # Reward: heavy penalty on waiting → learn to schedule aggressively
    config.reward_weights.utilization = 0.3
    config.reward_weights.waiting_time = 0.5   # main learning signal
    config.reward_weights.slowdown = 0.15
    config.reward_weights.throughput = 0.05
    config.reward_type = "hybrid"

    # Plugins
    config.plugins = [
        TensorBoardLoggerPlugin(log_dir="logs/tensorboard_phase1"),
        CSVLoggerPlugin(output_dir="logs/csv_phase1"),
    ]

    return PyBatGymEnv(config=config)


def evaluate_agent(env, model, num_episodes: int = 5) -> dict:
    """Run the trained agent and collect averaged metrics."""
    total_wait, total_sd, total_util, total_reward = 0.0, 0.0, 0.0, 0.0
    total_episodes = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep + 100)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        completed = env._adapter.get_completed_jobs()
        if completed:
            makespan = env._adapter.get_current_time()
            total_cores = env.unwrapped._state.get("resource").total_cores

            total_wait += sum(j.waiting_time for j in completed) / len(completed)
            total_sd += sum(j.bounded_slowdown for j in completed) / len(completed)
            if makespan > 0 and total_cores > 0:
                busy = sum(j.actual_runtime * j.requested_resources for j in completed)
                total_util += busy / (makespan * total_cores)
            total_reward += ep_reward
            total_episodes += 1

    n = max(total_episodes, 1)
    return {
        "avg_waiting_time": total_wait / n,
        "avg_slowdown": total_sd / n,
        "avg_utilization": total_util / n,
        "avg_reward": total_reward / n,
    }


def main():
    print("=" * 68)
    print("  PyBatGym Phase 1: PPO vs Heuristics on Large Workload (Mock)")
    print("=" * 68)

    data_dir = Path(__file__).parent.parent / "data"
    trace_path = data_dir / "workloads" / "medium_workload.json"

    if not trace_path.exists():
        print(f"[ERROR] Workload not found: {trace_path}")
        print("Generate it first:  python scripts/generate_workload.py --preset medium")
        return

    # ── 1. Create env ─────────────────────────────────────────────────────
    env = make_env(str(trace_path), total_nodes=4, cores_per_node=2)
    total_cores = env.unwrapped._config.platform.total_cores
    print(f"\n[Config]")
    print(f"  Workload    : {trace_path.name}")
    print(f"  Platform    : {env.unwrapped._config.platform.total_nodes} nodes x "
          f"{env.unwrapped._config.platform.cores_per_node} cores = {total_cores} total")
    print(f"  Jobs/episode: {env.unwrapped._config.workload.num_jobs}")
    print(f"  Reward type : {env.unwrapped._config.reward_type}")
    print(f"  Weights     : util={env.unwrapped._config.reward_weights.utilization}, "
          f"wait={env.unwrapped._config.reward_weights.waiting_time}, "
          f"slowdown={env.unwrapped._config.reward_weights.slowdown}")

    # ── 2. Heuristic baselines ────────────────────────────────────────────
    print("\n[1/4] Running Heuristic Baselines (3 episodes each)...")
    baseline_episodes = 3

    print("  FCFS...")
    fcfs_metrics = run_baseline(env, fcfs_policy, num_episodes=baseline_episodes)
    print("  SJF...")
    sjf_metrics = run_baseline(env, sjf_policy, num_episodes=baseline_episodes)
    print("  EASY Backfilling...")
    easy_metrics = run_baseline(env, easy_backfilling_policy, num_episodes=baseline_episodes)

    # ── 3. Train PPO ──────────────────────────────────────────────────────
    total_timesteps = 200_000

    print(f"\n[2/4] Training PPO Agent ({total_timesteps:,} steps)...")
    print(f"  n_steps=256, batch_size=64, lr=3e-4, ent_coef=0.01")
    print(f"  This may take 5-15 minutes...\n")

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="logs/tensorboard_phase1",
    )

    t_start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=ProgressCallback(print_every=10000),
    )
    train_time = time.time() - t_start

    # Save model
    model_path = Path("models") / "ppo_phase1"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\n  Model saved to: {model_path}")
    print(f"  Training time: {train_time:.0f}s")

    # ── 4. Evaluate trained PPO ───────────────────────────────────────────
    print(f"\n[3/4] Evaluating Trained PPO (5 episodes)...")
    ppo_metrics = evaluate_agent(env, model, num_episodes=5)

    # ── 5. Final comparison ───────────────────────────────────────────────
    print(f"\n[4/4] Results")
    print("=" * 78)
    print(f"{'Metric':<25} | {'FCFS':<12} | {'SJF':<12} | {'EASY BF':<12} | {'PPO (ours)':<12}")
    print("-" * 78)

    rows = [
        ("Avg Waiting Time (s)", "avg_waiting_time"),
        ("Avg Slowdown", "avg_slowdown"),
        ("Avg Utilization (%)", "avg_utilization"),
    ]

    for label, key in rows:
        vals = [fcfs_metrics.get(key, 0), sjf_metrics.get(key, 0),
                easy_metrics.get(key, 0), ppo_metrics.get(key, 0)]

        if "Utilization" in label:
            strs = [f"{v*100:.1f}%" for v in vals]
        else:
            strs = [f"{v:.2f}" for v in vals]

        # Mark best
        if "Waiting" in label or "Slowdown" in label:
            best_idx = vals.index(min(vals))
        else:
            best_idx = vals.index(max(vals))

        for i in range(len(strs)):
            if i == best_idx:
                strs[i] = f"*{strs[i]}*"

        print(f"{label:<25} | {strs[0]:<12} | {strs[1]:<12} | {strs[2]:<12} | {strs[3]:<12}")

    if "avg_reward" in ppo_metrics:
        print(f"\n  PPO avg episode reward: {ppo_metrics['avg_reward']:.4f}")

    print(f"\n  (* = best for that metric)")
    print(f"\n  View learning curves: tensorboard --logdir logs/tensorboard_phase1")
    print(f"  Model checkpoint: {model_path}")

    env.close()


if __name__ == "__main__":
    main()
