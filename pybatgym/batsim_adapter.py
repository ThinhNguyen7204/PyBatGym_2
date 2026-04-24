"""BatSim adapter layer for PyBatGym.

Provides an abstraction to decouple RL environment from the simulator.
- MockAdapter (EventDrivenMockAdapter): event-driven, matches real BatSim protocol
- RealAdapter: actual BatSim subprocess via pybatsim (see real_adapter.py)
"""

from __future__ import annotations

import heapq
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
    SimEvent,
    SimEventType,
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

    @abstractmethod
    def get_pending_jobs(self) -> list[Job]:
        """Return currently pending jobs."""

    @abstractmethod
    def get_completed_jobs(self) -> list[Job]:
        """Return currently completed jobs."""

    @abstractmethod
    def get_resource(self) -> Resource:
        """Return the current resource state."""


# ==========================================================================
# Event-Driven Mock Adapter (new default — matches real BatSim protocol)
# ==========================================================================

class EventDrivenMockAdapter(BatsimAdapter):
    """Event-driven mock simulator mirroring real BatSim protocol.

    Instead of advancing time by +1.0s each step, this adapter maintains
    a priority queue of future events (submissions, completions) and jumps
    time directly to the next event — exactly like real BatSim does over ZMQ.

    Decision loop (mirrors BatSim REQ/REP):
    1. RL agent calls step(command)
    2. Adapter processes command (allocate resources, schedule completion)
    3. Adapter pops next event batch from queue (all events at same timestamp)
    4. Time jumps to that timestamp
    5. Returns batch events to RL agent → next decision point
    """

    def __init__(self, config: PyBatGymConfig) -> None:
        self._config = config
        self._rng = random.Random(config.workload.seed)
        self._current_time = 0.0
        self._resource = Resource(
            total_nodes=config.platform.total_nodes,
            total_cores_per_node=config.platform.cores_per_node,
        )
        self._pending_jobs: list[Job] = []
        self._running_jobs: dict[int, _RunningJobInfo] = {}
        self._completed_jobs: list[Job] = []

        # Priority queue of future events
        self._event_queue: list[SimEvent] = []
        self._tiebreaker = 0

    def start(self) -> None:
        pass

    def reset(self) -> tuple[list[Event], Resource]:
        self._current_time = 0.0
        self._resource = Resource(
            total_nodes=self._config.platform.total_nodes,
            total_cores_per_node=self._config.platform.cores_per_node,
        )
        self._pending_jobs = []
        self._running_jobs = {}
        self._completed_jobs = []
        self._event_queue = []
        self._tiebreaker = 0

        # Load workload and seed ALL submission events into the queue
        all_jobs = self._generate_workload()
        for job in all_jobs:
            self._push_event(SimEvent(
                timestamp=job.submit_time,
                event_type=SimEventType.JOB_SUBMISSION,
                job=job,
            ))

        # Seed simulation-end sentinel
        self._push_event(SimEvent(
            timestamp=self._config.episode.max_simulation_time,
            event_type=SimEventType.SIMULATION_END,
        ))

        # Advance to first event batch (initial submissions at t=0 or near-0)
        initial_events = self._advance_to_next_decision_point()
        return initial_events, self._resource

    def step(self, command: Optional[ScheduleCommand]) -> tuple[list[Event], bool]:
        events: list[Event] = []
        did_schedule = False

        # 1. Process scheduling command
        if command and command.command_type == ScheduleCommandType.EXECUTE_JOB and command.job:
            job = command.job
            if self._resource.can_allocate(command.allocated_cores):
                self._resource.allocate(command.allocated_cores)
                job.status = JobStatus.RUNNING
                job.start_time = self._current_time

                # Schedule completion event
                finish_time = self._current_time + job.actual_runtime
                self._push_event(SimEvent(
                    timestamp=finish_time,
                    event_type=SimEventType.JOB_COMPLETION,
                    job=job,
                ))
                self._running_jobs[job.job_id] = _RunningJobInfo(
                    job=job, finish_time=finish_time, cores=command.allocated_cores,
                )
                self._pending_jobs = [j for j in self._pending_jobs if j.job_id != job.job_id]

                events.append(Event(
                    event_type=EventType.JOB_STARTED,
                    timestamp=self._current_time,
                    job=job,
                ))
                did_schedule = True

        # 2. If agent just scheduled a job and there are MORE schedulable
        #    pending jobs, stay at current time (like real BatSim batch reply).
        #    If agent sent WAIT/None → always advance to next event.
        if did_schedule and self._pending_jobs and any(
            self._resource.can_allocate(j.requested_resources) for j in self._pending_jobs
        ):
            done = self._check_done()
            if done:
                events.append(Event(event_type=EventType.SIMULATION_ENDED, timestamp=self._current_time))
            return events, done

        # 3. Advance to next event batch
        batch_events = self._advance_to_next_decision_point()
        events.extend(batch_events)

        done = self._check_done()
        if done:
            events.append(Event(event_type=EventType.SIMULATION_ENDED, timestamp=self._current_time))

        return events, done

    def close(self) -> None:
        self._pending_jobs.clear()
        self._running_jobs.clear()
        self._event_queue.clear()

    def get_current_time(self) -> float:
        return self._current_time

    def get_pending_jobs(self) -> list[Job]:
        return list(self._pending_jobs)

    def get_completed_jobs(self) -> list[Job]:
        return list(self._completed_jobs)

    def get_resource(self) -> Resource:
        return self._resource

    # -- Private methods --

    def _push_event(self, event: SimEvent) -> None:
        """Push event into priority queue with monotonic tiebreaker."""
        event._tiebreaker = self._tiebreaker
        self._tiebreaker += 1
        heapq.heappush(self._event_queue, event)

    def _advance_to_next_decision_point(self) -> list[Event]:
        """Pop all events at the next timestamp from the queue.

        This is the core of the event-driven model: time jumps directly
        to the next event instead of ticking +1.0s.
        """
        if not self._event_queue:
            return []

        next_time = self._event_queue[0].timestamp
        # Don't go backward in time
        if next_time < self._current_time:
            next_time = self._current_time

        # Pop all events at this timestamp (batch processing)
        batch: list[SimEvent] = []
        while self._event_queue and self._event_queue[0].timestamp <= next_time:
            batch.append(heapq.heappop(self._event_queue))

        self._current_time = next_time

        # Process batch → produce RL-facing events
        rl_events: list[Event] = []
        for sim_evt in batch:
            if sim_evt.event_type == SimEventType.JOB_SUBMISSION:
                rl_events.extend(self._handle_submission(sim_evt))
            elif sim_evt.event_type == SimEventType.JOB_COMPLETION:
                rl_events.extend(self._handle_completion(sim_evt))
            elif sim_evt.event_type == SimEventType.SIMULATION_END:
                pass  # handled by _check_done

        return rl_events

    def _handle_submission(self, sim_evt: SimEvent) -> list[Event]:
        """Process a job submission event."""
        job = sim_evt.job
        if job is None or job.status != JobStatus.PENDING:
            return []
        if job in self._pending_jobs:
            return []
        self._pending_jobs.append(job)
        return [Event(
            event_type=EventType.JOB_SUBMITTED,
            timestamp=self._current_time,
            job=job,
        )]

    def _handle_completion(self, sim_evt: SimEvent) -> list[Event]:
        """Process a job completion event."""
        job = sim_evt.job
        if job is None:
            return []
        info = self._running_jobs.pop(job.job_id, None)
        if info is None:
            return []

        job.status = JobStatus.COMPLETED
        job.finish_time = self._current_time
        self._resource.release(info.cores)
        self._completed_jobs.append(job)

        return [
            Event(
                event_type=EventType.JOB_COMPLETED,
                timestamp=self._current_time,
                job=job,
            ),
            Event(
                event_type=EventType.RESOURCE_FREED,
                timestamp=self._current_time,
                data={"freed_cores": info.cores},
            ),
        ]

    def _check_done(self) -> bool:
        """Check if simulation should end."""
        if self._current_time >= self._config.episode.max_simulation_time:
            return True
        # All events processed, nothing pending or running
        no_pending = len(self._pending_jobs) == 0
        no_running = len(self._running_jobs) == 0
        no_future = len(self._event_queue) == 0 or (
            len(self._event_queue) == 1
            and self._event_queue[0].event_type == SimEventType.SIMULATION_END
        )
        return no_pending and no_running and no_future

    def _generate_workload(self) -> list[Job]:
        """Generate synthetic workload or load from trace based on config."""
        wl = self._config.workload

        if wl.source == "trace" and wl.trace_path:
            from pybatgym.workload_parser import parse_workload
            jobs = parse_workload(wl.trace_path, max_cores=self._resource.total_cores)
            if wl.num_jobs > 0 and len(jobs) > wl.num_jobs:
                jobs = jobs[:wl.num_jobs]
            return jobs

        # Synthetic generation
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


class _RunningJobInfo:
    """Internal tracking for a job currently executing."""

    __slots__ = ("job", "finish_time", "cores")

    def __init__(self, job: Job, finish_time: float, cores: int) -> None:
        self.job = job
        self.finish_time = finish_time
        self.cores = cores



# Backward-compatible alias: MockAdapter → EventDrivenMockAdapter
MockAdapter = EventDrivenMockAdapter
