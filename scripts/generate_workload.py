#!/usr/bin/env python3
"""Generate realistic HPC workloads in BatSim JSON format.

Workload characteristics are based on observations from the Parallel
Workloads Archive (Feitelson et al., 2005) to create scheduling contention
that exposes the difference between heuristics and learned policies.

Usage:
    python scripts/generate_workload.py                         # 500 jobs, default
    python scripts/generate_workload.py --num-jobs 1000 --seed 7
    python scripts/generate_workload.py --preset heavy          # heavy contention
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# ── Preset configurations ────────────────────────────────────────────────────

PRESETS = {
    "light": dict(
        num_jobs=200,
        max_cores=4,
        arrival_rate=0.08,   # low arrival → little queueing
        runtime_mu=3.0,
        runtime_sigma=1.0,
    ),
    "medium": dict(
        num_jobs=500,
        max_cores=8,
        arrival_rate=0.25,   # moderate overlap
        runtime_mu=3.5,
        runtime_sigma=1.2,
    ),
    "heavy": dict(
        num_jobs=1000,
        max_cores=16,
        arrival_rate=0.5,    # heavy overlap → big backlog
        runtime_mu=4.0,
        runtime_sigma=1.5,
    ),
}


def generate_workload(
    num_jobs: int = 500,
    max_cores: int = 8,
    arrival_rate: float = 0.25,
    runtime_mu: float = 3.5,
    runtime_sigma: float = 1.2,
    seed: int = 42,
) -> dict:
    """Generate a BatSim‑compatible JSON workload.

    Parameters
    ----------
    num_jobs : int
        Number of jobs to generate.
    max_cores : int
        Maximum cores a single job can request.
    arrival_rate : float
        Mean arrivals per time unit (Poisson). Higher = more contention.
    runtime_mu, runtime_sigma : float
        Parameters for log‑normal runtime distribution (in ln‑seconds).
    seed : int
        Random seed for reproducibility.
    """
    rng = random.Random(seed)

    jobs = []
    profiles = {}
    current_time = 0.0

    for i in range(num_jobs):
        # ── Inter‑arrival time (exponential) ─────────────────────────────
        if arrival_rate > 0:
            inter = rng.expovariate(arrival_rate)
        else:
            inter = rng.uniform(1, 10)
        current_time += inter

        # ── Runtime (log‑normal → heavy tail like real HPC) ──────────────
        actual_runtime = math.exp(rng.gauss(runtime_mu, runtime_sigma))
        actual_runtime = max(1.0, min(actual_runtime, 5000.0))

        # walltime ≈ user estimate (overestimate by 10%–80%)
        overestimate = rng.uniform(1.1, 1.8)
        walltime = actual_runtime * overestimate

        # ── Resource request (power‑of‑2 biased, like real HPC) ──────────
        max_pow = max(0, int(math.log2(max_cores)))
        power = rng.randint(0, max_pow)
        cores = min(2 ** power, max_cores)

        # ── Create profile ───────────────────────────────────────────────
        profile_name = f"delay_{i}"
        profiles[profile_name] = {
            "type": "delay",
            "delay": round(actual_runtime, 2),
        }

        jobs.append({
            "id": i,
            "subtime": round(current_time, 2),
            "walltime": round(walltime, 2),
            "res": cores,
            "profile": profile_name,
        })

    return {
        "description": (
            f"Generated workload: {num_jobs} jobs, "
            f"arrival_rate={arrival_rate}, "
            f"max_cores={max_cores}, seed={seed}"
        ),
        "nb_res": max_cores,
        "jobs": jobs,
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate HPC workloads for PyBatGym")
    parser.add_argument("--num-jobs", type=int, default=500, help="Number of jobs")
    parser.add_argument("--max-cores", type=int, default=8, help="Max cores per job")
    parser.add_argument("--arrival-rate", type=float, default=0.25, help="Poisson arrival rate")
    parser.add_argument("--runtime-mu", type=float, default=3.5, help="Log-normal mu")
    parser.add_argument("--runtime-sigma", type=float, default=1.2, help="Log-normal sigma")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Use a preset config")
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "workloads" / "generated_workload.json"),
        help="Output file path",
    )
    args = parser.parse_args()

    if args.preset:
        params = PRESETS[args.preset].copy()
        params["seed"] = args.seed
    else:
        params = dict(
            num_jobs=args.num_jobs,
            max_cores=args.max_cores,
            arrival_rate=args.arrival_rate,
            runtime_mu=args.runtime_mu,
            runtime_sigma=args.runtime_sigma,
            seed=args.seed,
        )

    workload = generate_workload(**params)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(workload, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────
    jobs = workload["jobs"]
    runtimes = [workload["profiles"][j["profile"]]["delay"] for j in jobs]
    cores = [j["res"] for j in jobs]
    print(f"[OK] Generated {len(jobs)} jobs -> {out_path}")
    print(f"  Runtime : min={min(runtimes):.1f}  median={sorted(runtimes)[len(runtimes)//2]:.1f}  max={max(runtimes):.1f}")
    print(f"  Cores   : min={min(cores)}  median={sorted(cores)[len(cores)//2]}  max={max(cores)}")
    print(f"  Timespan: {jobs[0]['subtime']:.1f} -> {jobs[-1]['subtime']:.1f}")


if __name__ == "__main__":
    main()
