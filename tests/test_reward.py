"""Tests for RewardCalculator."""

from pybatgym.config.base_config import RewardWeights
from pybatgym.models import (
    Event,
    EventType,
    Job,
    JobStatus,
    Resource,
    ScheduleCommandType,
)
from pybatgym.reward import DefaultRewardCalculator


def _make_resource() -> Resource:
    return Resource(total_nodes=4, total_cores_per_node=4, used_cores=8)


def _make_completed_jobs() -> list[Job]:
    jobs = []
    for i in range(5):
        j = Job(
            job_id=i,
            submit_time=float(i),
            requested_walltime=10.0,
            actual_runtime=8.0,
            requested_resources=2,
            status=JobStatus.COMPLETED,
            start_time=float(i + 2),
            finish_time=float(i + 10),
        )
        jobs.append(j)
    return jobs


class TestDefaultRewardCalculator:
    def test_step_reward_schedule(self):
        calc = DefaultRewardCalculator(RewardWeights(), reward_type="step")
        state = {"resource": _make_resource(), "pending_jobs": []}
        reward = calc.compute_step_reward([], ScheduleCommandType.EXECUTE_JOB, state)
        assert reward > 0  # schedule bonus

    def test_step_reward_idle_penalty(self):
        calc = DefaultRewardCalculator(RewardWeights(), reward_type="step")
        pending = [Job(0, 0.0, 10.0, 8.0, 2)]
        state = {"resource": Resource(4, 4, 0), "pending_jobs": pending}
        reward = calc.compute_step_reward([], ScheduleCommandType.WAIT, state)
        assert reward < 0  # idle penalty + WAIT with pending

    def test_episodic_mode_zero_step(self):
        calc = DefaultRewardCalculator(RewardWeights(), reward_type="episodic")
        state = {"resource": _make_resource(), "pending_jobs": []}
        reward = calc.compute_step_reward([], ScheduleCommandType.EXECUTE_JOB, state)
        assert reward == 0.0

    def test_episode_reward(self):
        calc = DefaultRewardCalculator(RewardWeights())
        jobs = _make_completed_jobs()
        reward = calc.compute_episode_reward(jobs, 100.0)
        assert isinstance(reward, float)

    def test_episode_reward_no_jobs(self):
        calc = DefaultRewardCalculator(RewardWeights())
        assert calc.compute_episode_reward([], 100.0) == -1.0

    def test_reset_clears_state(self):
        calc = DefaultRewardCalculator(RewardWeights(), reward_type="step")
        state = {"resource": Resource(4, 4, 8), "pending_jobs": []}
        calc.compute_step_reward([], ScheduleCommandType.EXECUTE_JOB, state)
        calc.reset()
        assert calc._prev_utilization == 0.0
