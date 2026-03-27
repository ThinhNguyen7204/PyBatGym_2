"""Workload parser for PyBatGym.

Parses BatSim JSON workload descriptions or SWF traces into Job objects.
"""

from __future__ import annotations

import json
from pathlib import Path

from pybatgym.models import Job


def parse_workload(file_path: str | Path) -> list[Job]:
    """Parse a workload file and return a list of Jobs.

    Supports:
    - BatSim JSON workloads (.json)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Workload file not found: {path}")

    if path.suffix == ".json":
        return _parse_json_workload(path)
    # SWF placeholder
    elif path.suffix == ".swf":
        raise NotImplementedError("SWF parsing not yet implemented.")
    else:
        raise ValueError(f"Unsupported workload format: {path.suffix}")


def _parse_json_workload(path: Path) -> list[Job]:
    """Parse BatSim JSON format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_jobs = data.get("jobs", [])
    profiles = data.get("profiles", {})

    jobs: list[Job] = []
    for r in raw_jobs:
        job_id = int(r["id"]) if isinstance(r["id"], int) else int(r["id"].split("!")[-1])
        subtime = float(r["subtime"])
        walltime = float(r["walltime"])
        res = int(r["res"])

        # Estimate actual runtime from profile if it's a delay, else use walltime
        actual_runtime = walltime
        profile_name = str(r["profile"])
        if profile_name in profiles:
            prof = profiles[profile_name]
            if prof.get("type") == "delay" or prof.get("type") == "DelayProfile":
                actual_runtime = float(prof.get("delay", walltime))

        jobs.append(Job(
            job_id=job_id,
            submit_time=subtime,
            requested_walltime=walltime,
            actual_runtime=actual_runtime,
            requested_resources=res,
        ))

    return sorted(jobs, key=lambda j: j.submit_time)
