"""Action mapper for PyBatGym.

Maps discrete RL agent actions to BatSim scheduling commands.

Action space: Discrete(K+2) where:
  0..K-1 : select job i from top-K wait-sorted queue (EXECUTE_JOB)
  K      : WAIT (do nothing, advance simulation)
  K+1    : SCHEDULE_SMALLEST_FITTING — backfill-aware action: pick the
           smallest-core-demand job that fits in current free resources.
           If no job fits, falls back to WAIT.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import gymnasium as gym

from pybatgym.models import (
    Job,
    Resource,
    ScheduleCommand,
    ScheduleCommandType,
)


class ActionMapper(ABC):
    """Abstract base class for action mapping."""

    @abstractmethod
    def map(self, action: int, state: dict[str, Any]) -> Optional[ScheduleCommand]:
        """Map an integer action to a ScheduleCommand. Returns None for WAIT."""

    @abstractmethod
    def get_action_space(self) -> gym.spaces.Discrete:
        """Return the Gymnasium action space."""


class DefaultActionMapper(ActionMapper):
    """Maps Discrete(K+2) actions to scheduling commands.

    Actions 0..K-1 : select the i-th job from top-K pending queue
                     (sorted by waiting_time descending).
    Action K       : WAIT — do nothing, advance simulation.
    Action K+1     : SCHEDULE_SMALLEST_FITTING — backfill-aware:
                     schedule the pending job with the fewest requested cores
                     that currently fits in free resources. Falls back to WAIT
                     if no job fits.

    If selected job (actions 0..K-1) cannot be allocated due to insufficient
    resources, the action is treated as invalid and WAIT is returned.
    """

    # Sentinel action indices
    _WAIT = None  # determined dynamically via _max_jobs

    def __init__(self, max_jobs: int = 10) -> None:
        self._max_jobs = max_jobs
        # K+1 = WAIT index, K+2 total (0..K-1, K=WAIT, K+1=SMALLEST_FITTING)

    def get_action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self._max_jobs + 2)  # K + WAIT + SMALLEST_FITTING

    @property
    def wait_action(self) -> int:
        """Index of the WAIT action."""
        return self._max_jobs

    @property
    def backfill_action(self) -> int:
        """Index of SCHEDULE_SMALLEST_FITTING."""
        return self._max_jobs + 1

    def map(self, action: int, state: dict[str, Any]) -> Optional[ScheduleCommand]:
        pending_jobs: list[Job] = state.get("pending_jobs", [])
        resource: Resource = state["resource"]

        # ── WAIT ─────────────────────────────────────────────────────────
        if action == self.wait_action:
            return ScheduleCommand(command_type=ScheduleCommandType.WAIT)

        # ── SCHEDULE_SMALLEST_FITTING (backfill-aware) ────────────────────
        if action == self.backfill_action:
            fitting = [
                j for j in pending_jobs
                if resource.can_allocate(j.requested_resources)
            ]
            if not fitting:
                return ScheduleCommand(command_type=ScheduleCommandType.WAIT)
            # Pick job with minimum core demand (greedy backfill)
            selected = min(fitting, key=lambda j: j.requested_resources)
            return ScheduleCommand(
                command_type=ScheduleCommandType.EXECUTE_JOB,
                job=selected,
                allocated_cores=selected.requested_resources,
            )

        # ── SCHEDULE JOB i from top-K ──────────────────────────────────────
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: j.submit_time,
        )
        top_k = sorted_jobs[: self._max_jobs]

        if action >= len(top_k):
            return ScheduleCommand(command_type=ScheduleCommandType.WAIT)

        selected_job = top_k[action]

        if not resource.can_allocate(selected_job.requested_resources):
            return ScheduleCommand(command_type=ScheduleCommandType.WAIT)

        return ScheduleCommand(
            command_type=ScheduleCommandType.EXECUTE_JOB,
            job=selected_job,
            allocated_cores=selected_job.requested_resources,
        )
