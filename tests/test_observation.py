"""Tests for ObservationBuilder."""

import numpy as np

from pybatgym.config.base_config import ObservationConfig
from pybatgym.models import Job, Resource
from pybatgym.observation import DefaultObservationBuilder


def _make_state(num_jobs: int = 5, current_time: float = 50.0) -> dict:
    resource = Resource(total_nodes=16, total_cores_per_node=4, used_cores=20)
    pending = [
        Job(
            job_id=i,
            submit_time=float(i * 5),
            requested_walltime=10.0,
            actual_runtime=8.0,
            requested_resources=2,
        )
        for i in range(num_jobs)
    ]
    return {
        "current_time": current_time,
        "max_time": 10000.0,
        "pending_jobs": pending,
        "resource": resource,
    }


class TestDefaultObservationBuilder:
    def test_output_shape(self):
        config = ObservationConfig(top_k_jobs=10)
        builder = DefaultObservationBuilder(config)
        state = _make_state()
        obs = builder.build(state)

        expected_dim = 5 + 4 * 10 + 6  # 51
        assert obs["features"].shape == (expected_dim,)
        assert obs["action_mask"].shape == (11,)

    def test_values_normalized(self):
        config = ObservationConfig(top_k_jobs=5)
        builder = DefaultObservationBuilder(config)
        obs = builder.build(_make_state())

        assert np.all(obs["features"] >= 0.0)
        assert np.all(obs["features"] <= 1.0)

    def test_action_mask_valid(self):
        config = ObservationConfig(top_k_jobs=5)
        builder = DefaultObservationBuilder(config)
        obs = builder.build(_make_state())

        # WAIT is always valid
        assert obs["action_mask"][-1] == 1.0
        # At least some jobs schedulable (20 used / 64 total, jobs want 2)
        assert np.sum(obs["action_mask"][:-1]) > 0

    def test_empty_queue(self):
        config = ObservationConfig(top_k_jobs=5)
        builder = DefaultObservationBuilder(config)
        state = _make_state(num_jobs=0)
        obs = builder.build(state)

        # Job features should be zero
        job_features = obs["features"][5:5 + 4 * 5]
        assert np.all(job_features == 0.0)

    def test_observation_space_matches(self):
        config = ObservationConfig(top_k_jobs=10)
        builder = DefaultObservationBuilder(config)
        space = builder.get_observation_space()
        obs = builder.build(_make_state())

        assert space["features"].contains(obs["features"])
        assert space["action_mask"].contains(obs["action_mask"])
