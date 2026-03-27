"""Tests for workload_parser module."""

from pathlib import Path

from pybatgym.workload_parser import parse_workload


def test_parse_json_workload():
    """Verify JSON workload parser works correctly."""
    path = Path("D:/ThinhProject/data/workloads/tiny_workload.json")
    if not path.exists():
        return  # test skips if file is missing
        
    jobs = parse_workload(path)
    assert len(jobs) == 6
    assert jobs[0].job_id == 0
    assert jobs[0].submit_time == 0.0
    assert jobs[0].requested_walltime == 100.0
    assert jobs[0].actual_runtime == 10.0  # From DelayProfile
    assert jobs[0].requested_resources == 1
    
    assert jobs[-1].job_id == 5
    assert jobs[-1].submit_time == 15.0
