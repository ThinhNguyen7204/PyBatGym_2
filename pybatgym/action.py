"""Action mapper for PyBatGym.

Maps discrete RL agent actions to BatSim scheduling commands.
Action space: Discrete(K+1) where indices 0..K-1 select jobs, K = WAIT.
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
    """Maps Discrete(K+1) actions to scheduling commands.

    Actions 0..K-1: select the i-th job from top-K pending queue.
    Action K: WAIT (do nothing, advance simulation).

    If selected job cannot be allocated (insufficient resources), the action
    is treated as invalid and a WAIT is returned instead.
    """

    def __init__(self, max_jobs: int = 10) -> None:
        self._max_jobs = max_jobs

    def get_action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self._max_jobs + 1)

    def map(self, action: int, state: dict[str, Any]) -> Optional[ScheduleCommand]:
        if action == self._max_jobs:
            return ScheduleCommand(command_type=ScheduleCommandType.WAIT)

        pending_jobs: list[Job] = state.get("pending_jobs", [])
        resource: Resource = state["resource"]

        sorted_jobs = sorted(pending_jobs, key=lambda j: j.submit_time)
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
