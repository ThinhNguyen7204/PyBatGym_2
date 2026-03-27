"""BatSim adapter layer for PyBatGym.

Provides an abstraction to decouple RL environment from the simulator.
- MockAdapter: fast, no BatSim needed, synthetic workload generation
- (Future) RealAdapter: actual BatSim subprocess or Docker communication
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.models import (
    Event,
    EventType,
    Job,
    JobStatus,
    Resource,
    ScheduleCommand,
    ScheduleCommandType,
)


class BatsimAdapter(ABC):
    """Abstract adapter for BatSim communication."""

    @abstractmethod
    def start(self) -> None:
        """Initialize the simulation."""

    @abstractmethod
    def reset(self) -> tuple[list[Event], Resource]:
        """Reset simulation, return initial events and resource state."""

    @abstractmethod
    def step(self, command: Optional[ScheduleCommand]) -> tuple[list[Event], bool]:
        """Execute a command, return resulting events and done flag."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""

    @abstractmethod
    def get_current_time(self) -> float:
        """Return current simulation time."""


class MockAdapter(BatsimAdapter):
    """Fast mock simulator for development and testing.

    Generates synthetic workload and simulates job lifecycle
    without requiring BatSim installation.
    """

    def __init__(self, config: PyBatGymConfig) -> None:
        self._config = config
        self._rng = random.Random(config.workload.seed)
        self._current_time = 0.0
        self._resource = Resource(
            total_nodes=config.platform.total_nodes,
            total_cores_per_node=config.platform.cores_per_node,
        )
        self._all_jobs: list[Job] = []
        self._pending_jobs: list[Job] = []
        self._running_jobs: list[_RunningJob] = []
        self._completed_jobs: list[Job] = []
        self._next_submit_time = 0.0

    def start(self) -> None:
        pass

    def reset(self) -> tuple[list[Event], Resource]:
        self._current_time = 0.0
        self._resource = Resource(
            total_nodes=self._config.platform.total_nodes,
            total_cores_per_node=self._config.platform.cores_per_node,
        )
        self._all_jobs = self._generate_workload()
        self._pending_jobs = []
        self._running_jobs = []
        self._completed_jobs = []
        self._next_submit_time = 0.0

        initial_events = self._process_submissions()
        return initial_events, self._resource

    def step(self, command: Optional[ScheduleCommand]) -> tuple[list[Event], bool]:
        events: list[Event] = []

        if command and command.command_type == ScheduleCommandType.EXECUTE_JOB and command.job:
            job = command.job
            if self._resource.can_allocate(command.allocated_cores):
                self._resource.allocate(command.allocated_cores)
                job.status = JobStatus.RUNNING
                job.start_time = self._current_time

                finish_time = self._current_time + job.actual_runtime
                self._running_jobs.append(
                    _RunningJob(job=job, finish_time=finish_time, cores=command.allocated_cores),
                )
                self._pending_jobs = [j for j in self._pending_jobs if j.job_id != job.job_id]

                events.append(Event(
                    event_type=EventType.JOB_STARTED,
                    timestamp=self._current_time,
                    job=job,
                ))

        # Advance time
        self._current_time += 1.0

        # Process completions
        events.extend(self._process_completions())

        # Process new submissions
        events.extend(self._process_submissions())

        done = self._is_done()
        if done:
            events.append(Event(event_type=EventType.SIMULATION_ENDED, timestamp=self._current_time))

        return events, done

    def close(self) -> None:
        self._all_jobs.clear()
        self._pending_jobs.clear()
        self._running_jobs.clear()

    def get_current_time(self) -> float:
        return self._current_time

    def get_pending_jobs(self) -> list[Job]:
        return list(self._pending_jobs)

    def get_completed_jobs(self) -> list[Job]:
        return list(self._completed_jobs)

    # -- Private methods --

    def _generate_workload(self) -> list[Job]:
        """Generate synthetic workload or load from trace based on config."""
        wl = self._config.workload
        
        if wl.source == "trace" and wl.trace_path:
            from pybatgym.workload_parser import parse_workload
            jobs = parse_workload(wl.trace_path)
            # Limit to num_jobs if requested
            if wl.num_jobs > 0 and len(jobs) > wl.num_jobs:
                jobs = jobs[:wl.num_jobs]
            return jobs

        # Synthetic generation fallback
        jobs: list[Job] = []
        current_submit = 0.0

        for i in range(wl.num_jobs):
            inter_arrival = self._rng.expovariate(1.0 / 5.0)
            current_submit += inter_arrival

            requested_cores = self._rng.randint(1, min(wl.max_job_cores, self._resource.total_cores))
            walltime = self._rng.uniform(1.0, wl.max_job_runtime)
            actual = walltime * self._rng.uniform(0.5, 1.0)

            jobs.append(Job(
                job_id=i,
                submit_time=current_submit,
                requested_walltime=walltime,
                actual_runtime=actual,
                requested_resources=requested_cores,
            ))
        return jobs

    def _process_submissions(self) -> list[Event]:
        """Submit jobs whose submit_time <= current_time."""
        events: list[Event] = []
        newly_submitted: list[Job] = []

        for job in self._all_jobs:
            if job.submit_time <= self._current_time and job.status == JobStatus.PENDING:
                if job not in self._pending_jobs:
                    newly_submitted.append(job)
                    events.append(Event(
                        event_type=EventType.JOB_SUBMITTED,
                        timestamp=self._current_time,
                        job=job,
                    ))

        self._pending_jobs.extend(newly_submitted)
        return events

    def _process_completions(self) -> list[Event]:
        """Complete jobs whose finish_time <= current_time."""
        events: list[Event] = []
        still_running: list[_RunningJob] = []

        for rj in self._running_jobs:
            if rj.finish_time <= self._current_time:
                rj.job.status = JobStatus.COMPLETED
                rj.job.finish_time = rj.finish_time
                self._resource.release(rj.cores)
                self._completed_jobs.append(rj.job)
                events.append(Event(
                    event_type=EventType.JOB_COMPLETED,
                    timestamp=self._current_time,
                    job=rj.job,
                ))
                events.append(Event(
                    event_type=EventType.RESOURCE_FREED,
                    timestamp=self._current_time,
                    data={"freed_cores": rj.cores},
                ))
            else:
                still_running.append(rj)

        self._running_jobs = still_running
        return events

    def _is_done(self) -> bool:
        all_submitted = all(j.submit_time <= self._current_time for j in self._all_jobs)
        no_pending = len(self._pending_jobs) == 0
        no_running = len(self._running_jobs) == 0
        time_exceeded = self._current_time >= self._config.episode.max_simulation_time

        return (all_submitted and no_pending and no_running) or time_exceeded


class _RunningJob:
    """Internal tracking for jobs currently executing."""

    __slots__ = ("job", "finish_time", "cores")

    def __init__(self, job: Job, finish_time: float, cores: int) -> None:
        self.job = job
        self.finish_time = finish_time
        self.cores = cores
