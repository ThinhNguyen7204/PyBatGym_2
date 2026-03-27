"""Integration test: PPO training on PyBatGym."""

import pytest

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv


def _sb3_available() -> bool:
    try:
        import stable_baselines3  # noqa: F401
        return True
    except ImportError:
        return False


def test_random_agent_runs():
    """Verify environment works for many episodes with random actions."""
    config = PyBatGymConfig()
    config.workload.num_jobs = 10
    config.episode.max_steps = 200
    env = PyBatGymEnv(config=config)

    total_reward = 0.0
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

    env.close()
    assert isinstance(total_reward, float)


@pytest.mark.skipif(
    not _sb3_available(),
    reason="stable-baselines3 not installed",
)
def test_ppo_training():
    """Train PPO for 1000 steps — validates full SB3 compatibility."""
    from stable_baselines3 import PPO

    config = PyBatGymConfig()
    config.workload.num_jobs = 20
    config.episode.max_steps = 300
    env = PyBatGymEnv(config=config)

    model = PPO("MultiInputPolicy", env, verbose=0, n_steps=128, batch_size=64)
    model.learn(total_timesteps=1000)
    env.close()
