"""PyBatGym - A Gymnasium-compatible RL environment for HPC scheduling."""

__version__ = "0.1.0"

from pybatgym.env import PyBatGymEnv  # noqa: F401

from gymnasium.envs.registration import register

register(
    id="PyBatGym-v0",
    entry_point="pybatgym.env:PyBatGymEnv",
    kwargs={"config_path": None},
)
