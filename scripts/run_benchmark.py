"""Thesis benchmark runner — produces results/benchmark_summary.csv.

Runs all policies × workloads × platforms × seeds and exports a standardized
CSV conforming to the Handout Thesis Requirements (FR8, FR10).

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --seeds 42 43 44 --episodes 5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import uuid
from datetime import datetime, timezone

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv
from pybatgym.plugins.benchmark import (
    easy_backfilling_policy,
    fcfs_policy,
    sjf_policy,
)

# ── Configuration ────────────────────────────────────────────────────────────

POLICIES = {
    "Random": None,  # handled separately
    "FCFS": fcfs_policy,
    "SJF": sjf_policy,
    "EASY_Backfill": easy_backfilling_policy,
}

# Workload registry — paths are relative to PROJECT_ROOT
WORKLOADS = [
    {"name": "tiny",     "path": "data/workloads/tiny_workload.json",     "num_jobs": 6},
    {"name": "medium",   "path": "data/workloads/medium_workload.json",   "num_jobs": 100},
    {"name": "heavy",    "path": "data/workloads/heavy_workload.json",    "num_jobs": 1000},
    {"name": "backfill", "path": "data/workloads/backfill_workload.json", "num_jobs": 50},
]

# Platform registry — maps to YAML preset names in configs/
PLATFORM_PRESETS = {
    "small":  "small_batsim",    # configs/small_batsim.yaml
    "medium": "medium_batsim",   # configs/medium_batsim.yaml
}

CSV_COLUMNS = [
    "experiment_id",
    "timestamp",
    "policy",
    "workload",
    "platform",
    "total_cores",
    "seed",
    "episode",
    "steps",
    "terminated",
    "truncated",
    "total_reward",
    "completed_jobs",
    "total_jobs",
    "avg_waiting_time",
    "avg_bounded_slowdown",
    "utilization",
    "makespan",
    "throughput",
]

# ── Helpers ──────────────────────────────────────────────────────────────────


def make_config(
    workload: dict, platform_preset: str, seed: int,
) -> PyBatGymConfig:
    """Build config by loading YAML preset and overriding workload/seed.

    All platform, episode, observation, and reward parameters come from the
    YAML file — nothing is hardcoded here.
    """
    from pybatgym.config.loader import load_preset

    config = load_preset(platform_preset)

    # Override workload trace + seed
    config.workload.source = "trace"
    config.workload.trace_path = str(PROJECT_ROOT / workload["path"])
    config.workload.num_jobs = workload["num_jobs"]
    config.workload.seed = seed

    # Ensure max_job_cores matches the platform (auto-derive from YAML)
    config.workload.max_job_cores = config.platform.total_cores

    # Ensure observation queue length is at least as large as num_jobs
    if config.observation.max_queue_length < workload["num_jobs"]:
        config.observation.max_queue_length = workload["num_jobs"]

    return config


def run_single_episode(
    env: PyBatGymEnv, policy_fn, seed: int,
) -> dict:
    """Run one episode and return metrics dict."""
    obs, info = env.reset(seed=seed)
    done = False
    ep_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not done:
        if policy_fn is None:
            action = env.action_space.sample()
        else:
            action = policy_fn(env)
            action = min(action, env.action_space.n - 1)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1

    completed = env._adapter.get_completed_jobs()
    makespan = env._adapter.get_current_time()
    total_cores = env._state.get("resource").total_cores

    if completed:
        avg_wait = sum(j.waiting_time for j in completed) / len(completed)
        avg_slowdown = sum(j.bounded_slowdown for j in completed) / len(completed)
        busy = sum(j.actual_runtime * j.requested_resources for j in completed)
        utilization = min(1.0, busy / (makespan * total_cores)) if makespan > 0 else 0.0
        throughput = len(completed) / makespan if makespan > 0 else 0.0
    else:
        avg_wait = avg_slowdown = utilization = throughput = 0.0

    return {
        "steps": steps,
        "terminated": int(terminated),
        "truncated": int(truncated),
        "total_reward": round(ep_reward, 6),
        "completed_jobs": len(completed),
        "avg_waiting_time": round(avg_wait, 4),
        "avg_bounded_slowdown": round(avg_slowdown, 4),
        "utilization": round(utilization, 4),
        "makespan": round(makespan, 2),
        "throughput": round(throughput, 6),
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="PyBatGym Thesis Benchmark Runner")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/benchmark_summary.csv")
    parser.add_argument("--policies", nargs="+", default=list(POLICIES.keys()))
    parser.add_argument("--workloads", nargs="+", default=[w["name"] for w in WORKLOADS])
    parser.add_argument("--platforms", nargs="+", default=list(PLATFORM_PRESETS.keys()))
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_workloads = [w for w in WORKLOADS if w["name"] in args.workloads]
    selected_platform_names = [p for p in args.platforms if p in PLATFORM_PRESETS]
    selected_policies = {k: v for k, v in POLICIES.items() if k in args.policies}

    total_runs = (
        len(selected_policies) * len(selected_workloads)
        * len(selected_platform_names) * len(args.seeds) * args.episodes
    )

    print("=" * 50)
    print("       PyBatGym Thesis Benchmark Runner")
    print("=" * 50)
    print(f" Policies:  {', '.join(selected_policies.keys())}")
    print(f" Workloads: {', '.join(args.workloads)}")
    print(f" Platforms: {', '.join(selected_platform_names)}")
    print(f" Seeds:     {args.seeds}")
    print(f" Episodes:  {args.episodes}")
    print(f" Total runs: {total_runs}")
    print("=" * 50)
    print()

    rows: list[dict] = []
    run_count = 0
    start_time = time.time()

    for workload in selected_workloads:
        for platform_name in selected_platform_names:
            preset_name = PLATFORM_PRESETS[platform_name]

            for policy_name, policy_fn in selected_policies.items():
                for seed in args.seeds:
                    config = make_config(workload, preset_name, seed)
                    total_cores = config.platform.total_cores
                    env = PyBatGymEnv(config=config)
                    experiment_id = uuid.uuid4().hex[:12]
                    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

                    for ep in range(args.episodes):
                        run_count += 1
                        ep_seed = seed + ep

                        metrics = run_single_episode(env, policy_fn, ep_seed)

                        row = {
                            "experiment_id": experiment_id,
                            "timestamp": timestamp,
                            "policy": policy_name,
                            "workload": workload["name"],
                            "platform": platform_name,
                            "total_cores": total_cores,
                            "seed": ep_seed,
                            "episode": ep + 1,
                            "total_jobs": workload["num_jobs"],
                            **metrics,
                        }
                        rows.append(row)

                        elapsed = time.time() - start_time
                        pct = run_count / total_runs * 100
                        print(
                            f"  [{run_count}/{total_runs}] {pct:5.1f}% | "
                            f"{policy_name:16s} | {workload['name']:6s} | "
                            f"{platform_name:6s} | seed={ep_seed} | "
                            f"ep={ep+1} | reward={metrics['total_reward']:+.4f} | "
                            f"wait={metrics['avg_waiting_time']:.1f} | "
                            f"util={metrics['utilization']:.3f} | "
                            f"{elapsed:.1f}s"
                        )

                    env.close()

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    elapsed_total = time.time() - start_time
    print()
    print(f"[DONE] Benchmark complete: {run_count} runs in {elapsed_total:.1f}s")
    print(f"[FILE] Results saved to: {output_path}")

    # Print summary table
    print()
    print("=" * 90)
    print(f"{'Policy':16s} | {'Workload':8s} | {'Platform':8s} | "
          f"{'AvgReward':>10s} | {'AvgWait':>8s} | {'AvgUtil':>8s} | {'AvgSlowdown':>12s}")
    print("-" * 90)

    from itertools import groupby

    def group_key(r):
        return (r["policy"], r["workload"], r["platform"])

    sorted_rows = sorted(rows, key=group_key)
    for key, group_rows in groupby(sorted_rows, key=group_key):
        group_list = list(group_rows)
        n = len(group_list)
        avg_reward = sum(r["total_reward"] for r in group_list) / n
        avg_wait = sum(r["avg_waiting_time"] for r in group_list) / n
        avg_util = sum(r["utilization"] for r in group_list) / n
        avg_slow = sum(r["avg_bounded_slowdown"] for r in group_list) / n
        print(
            f"{key[0]:16s} | {key[1]:8s} | {key[2]:8s} | "
            f"{avg_reward:>+10.4f} | {avg_wait:>8.2f} | {avg_util:>8.4f} | {avg_slow:>12.4f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
