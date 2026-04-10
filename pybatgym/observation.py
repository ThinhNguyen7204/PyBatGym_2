"""Observation builder for PyBatGym.

Converts raw simulation state into fixed-size, normalized observation vectors
compatible with PPO/A2C policy networks.

Observation layout: [global(5) | job_queue(4*K) | resource(6)] = 11 + 4K dims

Job features per slot (4 dims):
  [0] wait_time_norm       : how long the job has been waiting
  [1] walltime_norm        : estimated runtime
  [2] cores_norm           : resource demand
  [3] bounded_slowdown_norm: current slowdown (urge to schedule) -- was duplicate in v1

Resource features (6 dims):
  [0] free_cores_ratio     : fraction of cores available
  [1] jobs_fitting_now     : fraction of top-K jobs that fit right now
  [2] queue_urgency        : fraction of pending jobs with BSD > 2.0
  [3] min_walltime_pending : shortest walltime in queue (normalized)
  [4] max_walltime_pending : longest walltime in queue (normalized)
  [5] fragmentation_proxy  : free_cores % cores_per_node_proxy
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
                low=0.0, high=1.0, shape=(self._k + 2,), dtype=np.float32,
            ),  # K jobs + WAIT + SCHEDULE_SMALLEST_FITTING
        })

    def build(self, state: dict[str, Any]) -> dict[str, NDArray[np.float32]]:
        current_time: float = state.get("current_time", 0.0)
        max_time: float = state.get("max_time", 1.0)
        pending_jobs: list[Job] = state.get("pending_jobs", [])
        resource: Resource = state["resource"]

        global_vec = self._build_global(current_time, max_time, pending_jobs, resource)
        job_vec = self._build_job_queue(pending_jobs, current_time)
        resource_vec = self._build_resource(resource, pending_jobs)

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
            wait = current_time - job.submit_time
            vec[offset]     = _normalize(wait, self._max_wait)             # [0] wait_time
            vec[offset + 1] = _normalize(job.requested_walltime, self._max_wait)  # [1] walltime
            vec[offset + 2] = _normalize(job.requested_resources, 64)     # [2] cores
            # [3] bounded_slowdown_norm — urge to schedule (was duplicate of [0] in v1)
            in_queue_bsd = max(1.0, (wait + max(job.requested_walltime, 10.0)) /
                               max(job.requested_walltime, 1.0))
            vec[offset + 3] = _normalize(in_queue_bsd - 1.0, self._max_bsd - 1.0)
        return vec

    def _build_resource(
        self, resource: Resource, pending_jobs: list[Job],
    ) -> NDArray[np.float32]:
        """6 resource features — all meaningful (no duplicates)."""
        # [0] free cores ratio
        free_ratio = _normalize(resource.free_cores, resource.total_cores)

        # [1] fraction of pending top-K jobs that fit right now
        top_k = pending_jobs[: self._k] if pending_jobs else []
        fitting = sum(1 for j in top_k if resource.can_allocate(j.requested_resources))
        jobs_fitting = _normalize(fitting, max(len(top_k), 1))

        # [2] queue urgency: fraction of pending jobs with queue-BSD > 2.0
        if pending_jobs:
            urgent = sum(1 for j in pending_jobs
                         if j.requested_walltime > 0
                         and (j.requested_walltime + j.submit_time) < j.submit_time + 2 * max(j.requested_walltime, 10.0))
            # simpler: jobs that have waited longer than their own walltime
            urgent = sum(
                1 for j in pending_jobs
                if (resource.free_cores >= j.requested_resources  # could run
                    and j.requested_walltime > 0)
            )
            queue_urgency = _normalize(urgent, max(len(pending_jobs), 1))
        else:
            queue_urgency = 0.0

        # [3] min walltime among pending jobs (guide short-job scheduling)
        if pending_jobs:
            min_wt = min(j.requested_walltime for j in pending_jobs)
            max_wt = max(j.requested_walltime for j in pending_jobs)
        else:
            min_wt = max_wt = 0.0
        min_walltime_norm = _normalize(min_wt, self._max_wait)

        # [4] max walltime among pending jobs (backfill shadow time estimate)
        max_walltime_norm = _normalize(max_wt, self._max_wait)

        # [5] fragmentation proxy: how many complete "min_core" chunks are free
        min_cores_in_queue = min((j.requested_resources for j in pending_jobs), default=1)
        fragmentation = _normalize(
            resource.free_cores % max(min_cores_in_queue, 1),
            max(min_cores_in_queue, 1)
        )

        return np.array([
            free_ratio,
            jobs_fitting,
            queue_urgency,
            min_walltime_norm,
            max_walltime_norm,
            fragmentation,
        ], dtype=np.float32)

    def _build_action_mask(
        self, pending_jobs: list[Job], resource: Resource,
    ) -> NDArray[np.float32]:
        """Mask: 1.0 = valid, 0.0 = invalid.

        Layout:
          [0..K-1] : top-K individual job selections
          [K]      : WAIT (always valid)
          [K+1]    : SCHEDULE_SMALLEST_FITTING (valid if any pending job fits)
        """
        mask = np.zeros(self._k + 2, dtype=np.float32)  # K + WAIT + BACKFILL
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: j.submit_time,
        )
        top_k = sorted_jobs[: self._k]
        for i, job in enumerate(top_k):
            if resource.can_allocate(job.requested_resources):
                mask[i] = 1.0
        mask[self._k] = 1.0       # WAIT always valid
        # BACKFILL valid if any pending job fits right now
        if any(resource.can_allocate(j.requested_resources) for j in pending_jobs):
            mask[self._k + 1] = 1.0
        return mask


def _normalize(value: float, max_value: float) -> float:
    """Normalize a value to [0, 1]."""
    if max_value <= 0:
        return 0.0
    return min(1.0, max(0.0, value / max_value))
