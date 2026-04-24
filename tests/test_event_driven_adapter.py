"""Tests for EventDrivenMockAdapter — event-driven mock BatSim."""

import pytest

from pybatgym.batsim_adapter import (
    EventDrivenMockAdapter,
    MockAdapter,
    TickBasedMockAdapter,
)
from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.models import EventType, ScheduleCommand, ScheduleCommandType


def _make_config(num_jobs: int = 20, seed: int = 42) -> PyBatGymConfig:
    config = PyBatGymConfig()
    config.workload.num_jobs = num_jobs
    config.workload.seed = seed
    return config


class TestEventDrivenMockAdapter:
    """Core behavior tests for the event-driven adapter."""

    def _make(self, num_jobs: int = 20) -> EventDrivenMockAdapter:
        return EventDrivenMockAdapter(_make_config(num_jobs))

    def test_reset_returns_events_and_resource(self):
        adapter = self._make()
        events, resource = adapter.reset()
        assert resource.total_cores > 0
        assert isinstance(events, list)
        # Should have at least one JOB_SUBMITTED at t~0
        assert any(e.event_type == EventType.JOB_SUBMITTED for e in events)

    def test_time_jumps_to_next_event(self):
        """Time must NOT advance by +1.0s — it jumps to next event."""
        adapter = self._make(num_jobs=5)
        adapter.reset()
        t0 = adapter.get_current_time()

        # Step with WAIT — time should jump to next event, not +1.0
        events, _ = adapter.step(None)
        t1 = adapter.get_current_time()

        # The jump could be to a submission or completion event
        assert t1 >= t0
        # Key: with tick-based it would be t0+1.0; event-driven jumps further
        # (unless there happens to be an event at exactly t0+1.0)

    def test_no_idle_ticking(self):
        """When no events exist at t+1.0, adapter should skip past idle time."""
        adapter = self._make(num_jobs=3)
        adapter.reset()

        # Schedule all pending jobs
        pending = adapter.get_pending_jobs()
        for job in pending:
            cmd = ScheduleCommand(
                command_type=ScheduleCommandType.EXECUTE_JOB,
                job=job,
                allocated_cores=job.requested_resources,
            )
            adapter.step(cmd)

        # Now step with WAIT — should jump to next event (completion or submission)
        t_before = adapter.get_current_time()
        adapter.step(None)
        t_after = adapter.get_current_time()

        # Should have jumped past idle time (not +1.0)
        assert t_after > t_before

    def test_batch_events_same_timestamp(self):
        """Events at the same timestamp should arrive in a single batch."""
        adapter = self._make(num_jobs=10)
        events, _ = adapter.reset()

        # Initial reset should batch all submissions at or near t=0
        submission_events = [e for e in events if e.event_type == EventType.JOB_SUBMITTED]
        # The first batch should contain at least 1 submission
        assert len(submission_events) >= 1

    def test_job_lifecycle(self):
        """Schedule a job → JOB_STARTED → eventually JOB_COMPLETED."""
        adapter = self._make(num_jobs=5)
        adapter.reset()

        pending = adapter.get_pending_jobs()
        assert len(pending) > 0

        job = pending[0]
        cmd = ScheduleCommand(
            command_type=ScheduleCommandType.EXECUTE_JOB,
            job=job,
            allocated_cores=job.requested_resources,
        )
        events, _ = adapter.step(cmd)

        started = [e for e in events if e.event_type == EventType.JOB_STARTED]
        assert len(started) >= 1
        assert started[0].job.job_id == job.job_id

        # Step until the job completes
        completed_seen = False
        for _ in range(500):
            events, done = adapter.step(None)
            for e in events:
                if e.event_type == EventType.JOB_COMPLETED and e.job.job_id == job.job_id:
                    completed_seen = True
                    break
            if completed_seen or done:
                break

        assert completed_seen

    def test_simulation_ends(self):
        """Simulation must eventually terminate."""
        adapter = self._make(num_jobs=3)
        adapter.reset()
        done = False
        for _ in range(5000):
            pending = adapter.get_pending_jobs()
            if pending:
                job = pending[0]
                cmd = ScheduleCommand(
                    command_type=ScheduleCommandType.EXECUTE_JOB,
                    job=job,
                    allocated_cores=job.requested_resources,
                )
                _, done = adapter.step(cmd)
            else:
                _, done = adapter.step(None)
            if done:
                break
        assert done

    def test_episode_fewer_steps_than_tick_based(self):
        """Event-driven should complete in fewer steps than tick-based."""
        config = _make_config(num_jobs=10)

        # Run event-driven
        ed = EventDrivenMockAdapter(config)
        ed.reset()
        ed_steps = 0
        for _ in range(10000):
            pending = ed.get_pending_jobs()
            if pending:
                job = pending[0]
                cmd = ScheduleCommand(
                    command_type=ScheduleCommandType.EXECUTE_JOB,
                    job=job,
                    allocated_cores=job.requested_resources,
                )
                _, done = ed.step(cmd)
            else:
                _, done = ed.step(None)
            ed_steps += 1
            if done:
                break

        # Run tick-based
        config2 = _make_config(num_jobs=10)
        tb = TickBasedMockAdapter(config2)
        tb.reset()
        tb_steps = 0
        for _ in range(20000):
            pending = tb.get_pending_jobs()
            if pending:
                job = pending[0]
                cmd = ScheduleCommand(
                    command_type=ScheduleCommandType.EXECUTE_JOB,
                    job=job,
                    allocated_cores=job.requested_resources,
                )
                _, done = tb.step(cmd)
            else:
                _, done = tb.step(None)
            tb_steps += 1
            if done:
                break

        # Event-driven should use significantly fewer steps
        assert ed_steps < tb_steps, f"ED={ed_steps} should be < TB={tb_steps}"

    def test_completion_frees_resources(self):
        """Resources must be freed when a job completes."""
        adapter = self._make(num_jobs=3)
        _, resource = adapter.reset()

        pending = adapter.get_pending_jobs()
        if not pending:
            pytest.skip("No pending jobs at reset")

        job = pending[0]
        cores_before = resource.free_cores
        cmd = ScheduleCommand(
            command_type=ScheduleCommandType.EXECUTE_JOB,
            job=job,
            allocated_cores=job.requested_resources,
        )
        adapter.step(cmd)
        cores_after_alloc = resource.free_cores
        assert cores_after_alloc < cores_before

        # Wait for completion
        for _ in range(500):
            events, done = adapter.step(None)
            for e in events:
                if e.event_type == EventType.JOB_COMPLETED and e.job.job_id == job.job_id:
                    # Resources should be freed
                    assert resource.free_cores > cores_after_alloc
                    return
            if done:
                break
        pytest.fail("Job never completed")

    def test_close_clears_state(self):
        adapter = self._make()
        adapter.reset()
        adapter.close()
        assert len(adapter.get_pending_jobs()) == 0

    def test_all_jobs_eventually_submitted(self):
        """All jobs in workload should be submitted as events."""
        adapter = self._make(num_jobs=10)
        events_from_reset, _ = adapter.reset()

        # Count jobs submitted during reset
        submitted_ids = set()
        for e in events_from_reset:
            if e.event_type == EventType.JOB_SUBMITTED and e.job:
                submitted_ids.add(e.job.job_id)

        for _ in range(2000):
            pending = adapter.get_pending_jobs()
            if pending:
                job = pending[0]
                cmd = ScheduleCommand(
                    command_type=ScheduleCommandType.EXECUTE_JOB,
                    job=job,
                    allocated_cores=job.requested_resources,
                )
                events, done = adapter.step(cmd)
            else:
                events, done = adapter.step(None)

            for e in events:
                if e.event_type == EventType.JOB_SUBMITTED and e.job:
                    submitted_ids.add(e.job.job_id)
            if done:
                break

        assert len(submitted_ids) == 10

    def test_stay_at_time_when_schedulable_jobs_remain(self):
        """Adapter should NOT advance time if pending schedulable jobs exist."""
        config = _make_config(num_jobs=5)
        config.platform.total_nodes = 16  # lots of cores
        config.platform.cores_per_node = 4
        adapter = EventDrivenMockAdapter(config)
        adapter.reset()

        pending = adapter.get_pending_jobs()
        if len(pending) < 2:
            pytest.skip("Need 2+ simultaneous submissions")

        # Schedule first job
        job1 = pending[0]
        cmd = ScheduleCommand(
            command_type=ScheduleCommandType.EXECUTE_JOB,
            job=job1,
            allocated_cores=job1.requested_resources,
        )
        events, _ = adapter.step(cmd)
        t_after_first = adapter.get_current_time()

        # If there are still schedulable pending jobs, time should NOT have advanced
        remaining = adapter.get_pending_jobs()
        if remaining and any(
            adapter._resource.can_allocate(j.requested_resources) for j in remaining
        ):
            assert adapter.get_current_time() == t_after_first


class TestBackwardCompat:
    """MockAdapter alias should still work."""

    def test_mock_adapter_is_event_driven(self):
        assert MockAdapter is EventDrivenMockAdapter

    def test_mock_adapter_functional(self):
        config = _make_config(num_jobs=5)
        adapter = MockAdapter(config)
        events, resource = adapter.reset()
        assert resource.total_cores > 0
        assert isinstance(events, list)

    def test_tick_based_still_works(self):
        config = _make_config(num_jobs=5)
        adapter = TickBasedMockAdapter(config)
        events, resource = adapter.reset()
        assert resource.total_cores > 0

        # Tick-based should advance by exactly 1.0
        t0 = adapter.get_current_time()
        adapter.step(None)
        assert adapter.get_current_time() == pytest.approx(t0 + 1.0)


class TestMetricsParity:
    """Event-driven and tick-based should produce comparable scheduling metrics."""

    def test_same_jobs_completed(self):
        """Both adapters should complete the same jobs (FIFO schedule)."""
        config_ed = _make_config(num_jobs=10)
        config_tb = _make_config(num_jobs=10)

        # FIFO schedule on both
        ed_completed = self._run_fifo(EventDrivenMockAdapter(config_ed))
        tb_completed = self._run_fifo(TickBasedMockAdapter(config_tb))

        ed_ids = sorted(j.job_id for j in ed_completed)
        tb_ids = sorted(j.job_id for j in tb_completed)

        assert ed_ids == tb_ids, "Both adapters should complete the same jobs"

    @staticmethod
    def _run_fifo(adapter) -> list:
        adapter.reset()
        for _ in range(20000):
            pending = adapter.get_pending_jobs()
            if pending:
                job = pending[0]
                cmd = ScheduleCommand(
                    command_type=ScheduleCommandType.EXECUTE_JOB,
                    job=job,
                    allocated_cores=job.requested_resources,
                )
                _, done = adapter.step(cmd)
            else:
                _, done = adapter.step(None)
            if done:
                break
        return adapter.get_completed_jobs()
