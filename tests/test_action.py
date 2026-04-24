"""Tests for ActionMapper."""

from pybatgym.action import DefaultActionMapper
from pybatgym.models import Job, Resource, ScheduleCommandType


def _make_state(num_jobs: int = 5) -> dict:
    resource = Resource(total_nodes=4, total_cores_per_node=4, used_cores=0)
    pending = [
        Job(
            job_id=i,
            submit_time=float(i),
            requested_walltime=10.0,
            actual_runtime=8.0,
            requested_resources=2,
        )
        for i in range(num_jobs)
    ]
    return {"pending_jobs": pending, "resource": resource}


class TestDefaultActionMapper:
    def test_wait_action(self):
        mapper = DefaultActionMapper(max_jobs=5)
        cmd = mapper.map(5, _make_state())  # action=K means WAIT
        assert cmd.command_type == ScheduleCommandType.WAIT

    def test_valid_job_selection(self):
        mapper = DefaultActionMapper(max_jobs=5)
        cmd = mapper.map(0, _make_state())
        assert cmd.command_type == ScheduleCommandType.EXECUTE_JOB
        assert cmd.job is not None
        assert cmd.job.job_id == 0

    def test_out_of_range_action(self):
        mapper = DefaultActionMapper(max_jobs=10)
        state = _make_state(num_jobs=3)
        cmd = mapper.map(7, state)  # only 3 jobs, index 7 invalid
        assert cmd.command_type == ScheduleCommandType.WAIT

    def test_insufficient_resources(self):
        mapper = DefaultActionMapper(max_jobs=5)
        state = _make_state()
        state["resource"] = Resource(total_nodes=1, total_cores_per_node=1, used_cores=1)
        cmd = mapper.map(0, state)  # job wants 2 cores, 0 free
        assert cmd.command_type == ScheduleCommandType.WAIT

    def test_action_space_size(self):
        mapper = DefaultActionMapper(max_jobs=10)
        space = mapper.get_action_space()
        assert space.n == 12  # K + WAIT + SCHEDULE_SMALLEST_FITTING
