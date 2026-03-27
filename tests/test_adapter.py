"""Tests for BatsimAdapter (MockAdapter)."""

from pybatgym.batsim_adapter import MockAdapter
from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.models import EventType, ScheduleCommand, ScheduleCommandType


class TestMockAdapter:
    def _make_adapter(self, num_jobs: int = 20) -> MockAdapter:
        config = PyBatGymConfig()
        config.workload.num_jobs = num_jobs
        config.workload.seed = 42
        return MockAdapter(config)

    def test_reset_returns_events_and_resource(self):
        adapter = self._make_adapter()
        events, resource = adapter.reset()
        assert resource.total_cores > 0
        assert isinstance(events, list)

    def test_step_advances_time(self):
        adapter = self._make_adapter()
        adapter.reset()
        t0 = adapter.get_current_time()
        adapter.step(None)
        assert adapter.get_current_time() > t0

    def test_job_lifecycle(self):
        adapter = self._make_adapter(num_jobs=5)
        adapter.reset()

        # Run until we have pending jobs
        for _ in range(50):
            adapter.step(None)
            if adapter.get_pending_jobs():
                break

        pending = adapter.get_pending_jobs()
        if pending:
            job = pending[0]
            cmd = ScheduleCommand(
                command_type=ScheduleCommandType.EXECUTE_JOB,
                job=job,
                allocated_cores=job.requested_resources,
            )
            events, _ = adapter.step(cmd)
            started = [e for e in events if e.event_type == EventType.JOB_STARTED]
            assert len(started) >= 1

    def test_simulation_ends(self):
        adapter = self._make_adapter(num_jobs=3)
        adapter.reset()
        done = False
        for _ in range(20000):
            _, done = adapter.step(None)
            if done:
                break
        assert done

    def test_close_clears_state(self):
        adapter = self._make_adapter()
        adapter.reset()
        adapter.close()
        assert len(adapter.get_pending_jobs()) == 0
