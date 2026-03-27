"""Observation builder for PyBatGym.

Converts raw simulation state into fixed-size, normalized observation vectors
compatible with PPO/A2C policy networks.

Observation layout: [global(5) | job_queue(4*K) | resource(6)] = 11 + 4K dims
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from pybatgym.config.base_config import ObservationConfig
from pybatgym.models import Job, Resource


class ObservationBuilder(ABC):
    """Abstract base class for observation construction."""

    @abstractmethod
    def build(self, state: dict[str, Any]) -> dict[str, NDArray[np.float32]]:
        """Convert raw state into observation dict with 'features' and 'action_mask'."""

    @abstractmethod
    def get_observation_space(self) -> gym.spaces.Dict:
        """Return the Gymnasium observation space definition."""


class DefaultObservationBuilder(ObservationBuilder):
    """PPO-friendly fixed-size observation builder.

    Produces a flat vector of shape (11 + 4*K,) where K = top_k_jobs.
    All values normalized to [0, 1].
    """

    GLOBAL_FEATURES = 5
    JOB_FEATURES = 4
    RESOURCE_FEATURES = 6

    def __init__(self, config: ObservationConfig) -> None:
        self._k = config.top_k_jobs
        self._max_queue = config.max_queue_length
        self._max_wait = config.max_waiting_time
        self._max_bsd = config.max_bounded_slowdown
        self._obs_dim = self.GLOBAL_FEATURES + self.JOB_FEATURES * self._k + self.RESOURCE_FEATURES

    def get_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            "features": gym.spaces.Box(
                low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32,
            ),
            "action_mask": gym.spaces.Box(
                low=0.0, high=1.0, shape=(self._k + 1,), dtype=np.float32,
            ),
        })

    def build(self, state: dict[str, Any]) -> dict[str, NDArray[np.float32]]:
        current_time: float = state.get("current_time", 0.0)
        max_time: float = state.get("max_time", 1.0)
        pending_jobs: list[Job] = state.get("pending_jobs", [])
        resource: Resource = state["resource"]

        global_vec = self._build_global(current_time, max_time, pending_jobs, resource)
        job_vec = self._build_job_queue(pending_jobs, current_time)
        resource_vec = self._build_resource(resource)

        features = np.concatenate([global_vec, job_vec, resource_vec]).astype(np.float32)
        action_mask = self._build_action_mask(pending_jobs, resource)

        return {"features": features, "action_mask": action_mask}

    def _build_global(
        self,
        current_time: float,
        max_time: float,
        pending_jobs: list[Job],
        resource: Resource,
    ) -> NDArray[np.float32]:
        avg_wait = 0.0
        avg_bsd = 0.0
        if pending_jobs:
            waits = [current_time - j.submit_time for j in pending_jobs]
            avg_wait = sum(waits) / len(waits)
            avg_bsd = sum(j.bounded_slowdown for j in pending_jobs) / len(pending_jobs)

        return np.array([
            _normalize(current_time, max_time),
            _normalize(len(pending_jobs), self._max_queue),
            resource.utilization,
            _normalize(avg_wait, self._max_wait),
            _normalize(avg_bsd, self._max_bsd),
        ], dtype=np.float32)

    def _build_job_queue(
        self, pending_jobs: list[Job], current_time: float,
    ) -> NDArray[np.float32]:
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: current_time - j.submit_time,
            reverse=True,
        )
        top_k = sorted_jobs[: self._k]

        vec = np.zeros(self.JOB_FEATURES * self._k, dtype=np.float32)
        for i, job in enumerate(top_k):
            offset = i * self.JOB_FEATURES
            vec[offset] = _normalize(current_time - job.submit_time, self._max_wait)
            vec[offset + 1] = _normalize(job.requested_walltime, self._max_wait)
            vec[offset + 2] = _normalize(job.requested_resources, 64)
            vec[offset + 3] = _normalize(current_time - job.submit_time, self._max_wait)
        return vec

    def _build_resource(self, resource: Resource) -> NDArray[np.float32]:
        return np.array([
            _normalize(resource.free_cores, resource.total_cores),
            _normalize(resource.total_nodes, resource.total_nodes),  # free_nodes proxy
            resource.utilization,
            resource.utilization,  # max_node_util proxy
            0.0,  # std_node_util placeholder
            _normalize(resource.free_cores % 4, 4),  # fragmentation proxy
        ], dtype=np.float32)

    def _build_action_mask(
        self, pending_jobs: list[Job], resource: Resource,
    ) -> NDArray[np.float32]:
        """Mask: 1.0 = valid, 0.0 = invalid. Last slot = WAIT action."""
        mask = np.zeros(self._k + 1, dtype=np.float32)
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: j.submit_time,
        )
        top_k = sorted_jobs[: self._k]
        for i, job in enumerate(top_k):
            if resource.can_allocate(job.requested_resources):
                mask[i] = 1.0
        mask[-1] = 1.0  # WAIT is always valid
        return mask


def _normalize(value: float, max_value: float) -> float:
    """Normalize a value to [0, 1]."""
    if max_value <= 0:
        return 0.0
    return min(1.0, max(0.0, value / max_value))
