"""Tests for PyBatGymEnv."""

import gymnasium as gym
import numpy as np

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv


def _make_env() -> PyBatGymEnv:
    config = PyBatGymConfig()
    config.workload.num_jobs = 20
    config.workload.seed = 42
    config.episode.max_steps = 500
    return PyBatGymEnv(config=config)


class TestPyBatGymEnv:
    def test_reset_returns_obs_info(self):
        env = _make_env()
        obs, info = env.reset(seed=42)
        assert "features" in obs
        assert "action_mask" in obs
        assert "step" in info
        env.close()

    def test_step_returns_correct_tuple(self):
        env = _make_env()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_observation_space_contains_obs(self):
        env = _make_env()
        obs, _ = env.reset(seed=42)
        assert env.observation_space["features"].contains(obs["features"])
        assert env.observation_space["action_mask"].contains(obs["action_mask"])
        env.close()

    def test_full_episode(self):
        env = _make_env()
        obs, info = env.reset(seed=42)
        total_reward = 0.0
        steps = 0

        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        assert steps > 0
        assert isinstance(total_reward, float)
        env.close()

    def test_gymnasium_make(self):
        import pybatgym  # noqa: F401 - triggers registration
        env = gym.make("PyBatGym-v0")
        obs, info = env.reset()
        assert "features" in obs
        env.close()

    def test_render_ansi(self):
        env = _make_env()
        env.render_mode = "ansi"
        env.reset(seed=42)
        env.step(0)
        output = env.render()
        assert isinstance(output, str)
        assert "PyBatGym" in output
        env.close()

    def test_multiple_resets(self):
        env = _make_env()
        for _ in range(3):
            obs, info = env.reset(seed=42)
            assert obs["features"].shape[0] > 0
        env.close()

    def test_info_keys(self):
        env = _make_env()
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)
        expected_keys = {"step", "sim_time", "pending", "completed", "utilization", "cumulative_reward", "num_events"}
        assert expected_keys.issubset(info.keys())
        env.close()
