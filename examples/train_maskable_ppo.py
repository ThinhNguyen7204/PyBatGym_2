"""MaskablePPO training with action masking for PyBatGym.

Integrates sb3-contrib's MaskablePPO with PyBatGym's action_masks() interface.
Invalid actions (scheduling jobs that don't fit) are automatically masked out,
resulting in faster convergence and better policies compared to standard PPO.

Usage:
    # Quick test (Windows / local)
    python examples/train_maskable_ppo.py --preset small_batsim --timesteps 100000

    # Full training (Docker / Linux)
    python examples/train_maskable_ppo.py --preset medium_batsim --timesteps 500000

    # With TensorBoard
    tensorboard --logdir logs/tensorboard_maskable --port 6006

Key differences from standard PPO:
    - Uses MaskablePPO from sb3-contrib instead of PPO from stable-baselines3
    - Env exposes action_masks() method (auto-detected by MaskablePPO)
    - Zero invalid actions → faster training, better convergence
    - predict() uses action_masks=True for deterministic evaluation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.config.loader import load_preset
from pybatgym.env import PyBatGymEnv
from pybatgym.plugins.benchmark import (
    easy_backfilling_policy,
    fcfs_policy,
    run_baseline,
    sjf_policy,
)


# ── Competitive Callback ────────────────────────────────────────────────────


class CompetitiveStopCallback(BaseCallback):
    """Stop training when MaskablePPO beats SJF win_rate >= threshold."""

    def __init__(
        self,
        sjf_wait: float,
        win_window: int = 50,
        win_rate_threshold: float = 0.80,
        min_episodes: int = 80,
        max_timesteps: int = 500_000,
        check_every: int = 2000,
        save_path: str | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.sjf_wait = sjf_wait
        self.win_window = win_window
        self.win_rate_threshold = win_rate_threshold
        self.min_episodes = min_episodes
        self.max_timesteps = max_timesteps
        self.check_every = check_every
        self.save_path = save_path

        self._wins: list[bool] = []
        self._episode_waits: list[float] = []
        self._n_episodes = 0
        self._last_check = 0
        self._best_win_rate = 0.0
        self._start_time = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if not done:
                continue
            self._n_episodes += 1
            vec_env = self.training_env
            env = vec_env.envs[i] if hasattr(vec_env, "envs") else None
            if env is None:
                continue

            raw = getattr(env, "unwrapped", env)
            adapter = getattr(raw, "_adapter", None)
            if adapter is None:
                continue
            completed = adapter.get_completed_jobs()
            if not completed:
                continue

            avg_wait = sum(j.waiting_time for j in completed) / len(completed)
            won = avg_wait < self.sjf_wait
            self._wins.append(won)
            self._episode_waits.append(avg_wait)

        if self.num_timesteps >= self.max_timesteps:
            print(f"\n[STOP] Hard cap {self.max_timesteps:,} steps. Episodes={self._n_episodes}")
            return False

        if self._n_episodes < self.min_episodes:
            return True
        if self.num_timesteps - self._last_check < self.check_every:
            return True

        self._last_check = self.num_timesteps
        return self._check()

    def _check(self) -> bool:
        if len(self._wins) < self.win_window:
            return True
        recent = self._wins[-self.win_window :]
        win_rate = sum(recent) / len(recent)
        avg_wait = float(np.mean(self._episode_waits[-self.win_window :]))
        elapsed = time.time() - self._start_time

        if self.verbose >= 1:
            print(
                f"  [{self.num_timesteps:>9,}] "
                f"eps={self._n_episodes:>4}  "
                f"win_rate={win_rate:.0%}  "
                f"avg_wait={avg_wait:.1f}  "
                f"SJF_wait={self.sjf_wait:.1f}  "
                f"t={elapsed:.0f}s"
            )

        if win_rate > self._best_win_rate and self.save_path:
            self._best_win_rate = win_rate
            self.model.save(self.save_path + "_best")

        if win_rate >= self.win_rate_threshold:
            print(
                f"\n{'='*60}\n"
                f"[TARGET REACHED] MaskablePPO beats SJF!\n"
                f"  Win rate  : {win_rate:.1%}\n"
                f"  avg_wait  : {avg_wait:.2f}s vs SJF={self.sjf_wait:.2f}s\n"
                f"  Episodes  : {self._n_episodes}\n"
                f"{'='*60}"
            )
            return False
        return True


# ── Env factory ──────────────────────────────────────────────────────────────


def make_env(preset: str, workload_override: str | None = None) -> PyBatGymEnv:
    """Create a PyBatGymEnv from a YAML preset.

    The env exposes action_masks() which MaskablePPO auto-detects.
    No ActionMasker wrapper needed.
    """
    config = load_preset(preset)

    if workload_override:
        config.workload.trace_path = workload_override
    else:
        # Resolve Docker /workspace/ prefix to local project root
        tp = config.workload.trace_path
        if tp and tp.startswith("/workspace/"):
            config.workload.trace_path = str(PROJECT_ROOT / tp.removeprefix("/workspace/"))

    return PyBatGymEnv(config=config)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MaskablePPO training for PyBatGym",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset", type=str, default="small_batsim",
        help="YAML config preset name from configs/",
    )
    parser.add_argument(
        "--workload", type=str, default=None,
        help="Override workload trace path (relative to project root)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Maximum training timesteps",
    )
    parser.add_argument(
        "--win-rate", type=float, default=0.80,
        help="SJF win-rate threshold for early stopping",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-dir", type=str, default="models",
        help="Directory to save trained models",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PyBatGym: MaskablePPO Training (Action Masking)")
    print("=" * 60)
    print(f"  Preset    : {args.preset}")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Win target: {args.win_rate:.0%} vs SJF")
    print(f"  Seed      : {args.seed}")
    print()

    # ── Create env ───────────────────────────────────────────────────────
    workload_path = None
    if args.workload:
        workload_path = str(PROJECT_ROOT / args.workload)

    env = make_env(args.preset, workload_path)
    print(f"  Obs space : {env.observation_space}")
    print(f"  Act space : {env.action_space}")

    # Verify action_masks works
    obs, _ = env.reset(seed=args.seed)
    mask = env.action_masks()
    print(f"  Mask shape: {mask.shape}  valid: {mask.sum()}/{len(mask)}")

    # ── Compute baselines ────────────────────────────────────────────────
    print("\n[1/3] Computing heuristic baselines...")
    sjf_m = run_baseline(env, sjf_policy, num_episodes=5)
    fcfs_m = run_baseline(env, fcfs_policy, num_episodes=5)
    easy_m = run_baseline(env, easy_backfilling_policy, num_episodes=5)

    sjf_wait = sjf_m["avg_waiting_time"]
    print(f"  SJF   wait={sjf_wait:.1f}  util={sjf_m['avg_utilization']:.1%}")
    print(f"  FCFS  wait={fcfs_m['avg_waiting_time']:.1f}  util={fcfs_m['avg_utilization']:.1%}")
    print(f"  EASY  wait={easy_m['avg_waiting_time']:.1f}  util={easy_m['avg_utilization']:.1%}")
    print(f"\n  Target: MaskablePPO win_rate > {args.win_rate:.0%} vs SJF wait={sjf_wait:.1f}s")

    # ── Build MaskablePPO ────────────────────────────────────────────────
    print("\n[2/3] Building MaskablePPO model...")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    save_path = str(save_dir / f"maskable_ppo_{args.preset}")

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        seed=args.seed,
        n_steps=512,
        batch_size=128,
        learning_rate=3e-4,
        ent_coef=0.02,
        clip_range=0.2,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="logs/tensorboard_maskable",
    )
    print("  Policy    : MultiInputPolicy (MaskablePPO)")
    print("  n_steps=512  batch=128  lr=3e-4  ent_coef=0.02")

    # ── Callbacks ────────────────────────────────────────────────────────
    competitive_cb = CompetitiveStopCallback(
        sjf_wait=sjf_wait,
        win_window=50,
        win_rate_threshold=args.win_rate,
        min_episodes=80,
        max_timesteps=args.timesteps,
        check_every=2000,
        save_path=save_path,
        verbose=1,
    )

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\n[3/3] Training MaskablePPO (max {args.timesteps:,} steps)...")
    print("-" * 60)

    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=competitive_cb,
        reset_num_timesteps=True,
    )
    train_time = time.time() - t0

    model.save(save_path)
    print(f"\n  Model saved: {save_path}.zip")
    print(f"  Training   : {train_time:.0f}s ({train_time/60:.1f} min)")
    print(f"  Steps      : {model.num_timesteps:,}")
    print(f"  Episodes   : {competitive_cb._n_episodes}")
    print(f"  Best win   : {competitive_cb._best_win_rate:.1%}")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final Evaluation (10 episodes, deterministic)")
    print("-" * 60)

    best_path = save_path + "_best.zip"
    eval_model = (
        MaskablePPO.load(best_path, env=env)
        if Path(best_path).exists()
        else model
    )

    waits, utils, rewards = [], [], []
    for ep in range(10):
        obs, _ = env.reset(seed=100 + ep)
        done, ep_reward = False, 0.0
        while not done:
            action, _ = eval_model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        completed = env._adapter.get_completed_jobs()
        if completed:
            n = len(completed)
            avg_w = sum(j.waiting_time for j in completed) / n
            makespan = env._adapter.get_current_time()
            tc = env._config.platform.total_cores
            busy = sum(j.actual_runtime * j.requested_resources for j in completed)
            util = min(1.0, busy / (makespan * tc)) if makespan > 0 else 0.0
            waits.append(avg_w)
            utils.append(util)
            rewards.append(ep_reward)

    if waits:
        ppo_wait = float(np.mean(waits))
        ppo_util = float(np.mean(utils))
        print(f"  {'Metric':<28s} | {'SJF':>10s} | {'MaskPPO':>10s}")
        print("-" * 60)
        print(f"  {'Avg Waiting Time (s)':<28s} | {sjf_wait:>10.2f} | {ppo_wait:>10.2f}")
        print(f"  {'Avg Utilization':<28s} | {sjf_m['avg_utilization']:>9.1%}  | {ppo_util:>9.1%}")
        print(f"  {'Avg Reward':<28s} | {'n/a':>10s} | {np.mean(rewards):>+10.2f}")
        print("=" * 60)

        if ppo_wait < sjf_wait:
            improvement = (sjf_wait - ppo_wait) / sjf_wait * 100
            print(f"\n  [OK] MaskablePPO beats SJF by {improvement:.1f}% on waiting time!")
        else:
            print(f"\n  [INFO] MaskablePPO did not beat SJF. More training may help.")
    else:
        print("  No completed jobs in evaluation.")

    env.close()
    print(f"\n  TensorBoard: tensorboard --logdir logs/tensorboard_maskable --port 6006")


if __name__ == "__main__":
    main()
