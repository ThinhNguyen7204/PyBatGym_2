"""RealEvalCallback — evaluate PPO policy against real BatSim during Mock training.

Every `eval_freq` training steps:
  1. Create a PyBatGymEnv with mode="real" (RealBatsimAdapter via ZMQ)
  2. Run `eval_episodes` episodes with current PPO model (deterministic)
  3. Log Real/* metrics to TensorBoard
  4. Gracefully skip if BatSim is unavailable (training continues)

Requirements:
  - docker-compose up batsim   must be running before training starts
  - BATSIM_SOCKET env var (default: tcp://batsim:28000)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from pybatgym.config.base_config import PyBatGymConfig


class RealEvalCallback(BaseCallback):
    """Evaluate current PPO policy on real BatSim every N training steps.

    Args:
        real_config:    PyBatGymConfig with mode="real" pointing to BatSim service.
        eval_freq:      Evaluate every this many *training* steps.
        eval_episodes:  Number of real BatSim episodes per evaluation.
        sjf_wait:       SJF baseline avg_waiting_time for comparison.
        verbose:        0=silent, 1=key events.
    """

    def __init__(
        self,
        real_config: PyBatGymConfig,
        eval_freq: int = 50_000,
        eval_episodes: int = 1,
        sjf_wait: float = 0.0,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.real_config = real_config
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.sjf_wait = sjf_wait
        self._last_eval: int = 0
        self._eval_count: int = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        self._run_real_eval()
        return True

    def _run_real_eval(self) -> None:
        """Run eval_episodes with real BatSim and log Real/* metrics."""
        from pybatgym.env import PyBatGymEnv

        self._eval_count += 1
        t0 = time.time()
        if self.verbose >= 1:
            print(
                f"\n  [RealEval #{self._eval_count} @ {self.num_timesteps:,} steps] "
                f"Running {self.eval_episodes} real episode(s)..."
            )

        waits, utils, sds, rewards = [], [], [], []

        env = None
        try:
            env = PyBatGymEnv(config=self.real_config)

            for ep in range(self.eval_episodes):
                # Vary seed per eval call so consecutive evals test different conditions
                seed = 100 + self._eval_count * self.eval_episodes + ep
                obs, _ = env.reset(seed=seed)
                done = False
                ep_reward = 0.0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(int(action))
                    ep_reward += reward
                    done = terminated or truncated

                raw = getattr(env, "unwrapped", env)
                adapter = getattr(raw, "_adapter", None)
                if adapter is None:
                    continue

                completed = adapter.get_completed_jobs()
                if not completed:
                    continue

                n = len(completed)
                makespan = adapter.get_current_time()
                total_cores = raw._config.platform.total_cores

                avg_wait = sum(j.waiting_time for j in completed) / n
                avg_sd = sum(j.bounded_slowdown for j in completed) / n
                util = 0.0
                if makespan > 0 and total_cores > 0:
                    busy = sum(j.actual_runtime * j.requested_resources for j in completed)
                    util = min(1.0, busy / (makespan * total_cores))  # clamp to [0,1]

                waits.append(avg_wait)
                utils.append(util)
                sds.append(avg_sd)
                rewards.append(ep_reward)

        except Exception as exc:
            if self.verbose >= 1:
                print(f"  [RealEval] Skipped — BatSim unavailable: {exc}")
            return
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            # Allow ZMQ port to be released before BatSim restarts
            time.sleep(3)

        if not waits:
            return

        avg_wait = float(np.mean(waits))
        avg_util = float(np.mean(utils))
        avg_sd = float(np.mean(sds))
        avg_rew = float(np.mean(rewards))
        advantage = (self.sjf_wait - avg_wait) / max(self.sjf_wait, 1e-8)
        elapsed = time.time() - t0

        self.logger.record("Real/avg_waiting_time", avg_wait)
        self.logger.record("Real/utilization", avg_util)
        self.logger.record("Real/avg_slowdown", avg_sd)
        self.logger.record("Real/avg_reward", avg_rew)
        self.logger.record("Real/advantage_over_SJF", advantage)

        if self.verbose >= 1:
            verdict = "BEAT SJF ✓" if avg_wait < self.sjf_wait else "behind SJF ✗"
            print(
                f"  [RealEval #{self._eval_count}] "
                f"wait={avg_wait:.1f}s (SJF={self.sjf_wait:.1f})  "
                f"util={avg_util:.1%}  sd={avg_sd:.2f}  "
                f"reward={avg_rew:.2f}  [{verdict}]  t={elapsed:.0f}s"
            )
