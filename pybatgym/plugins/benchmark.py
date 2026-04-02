"""Baseline benchmark plugins for PyBatGym.

Implements classic HPC scheduling heuristics (FCFS, SJF, EASY) as reference baselines
for comparing RL agent performance.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from pybatgym.env import PyBatGymEnv
from pybatgym.plugins.registry import Plugin


def fcfs_policy(env: PyBatGymEnv) -> int:
    """First-Come-First-Served: schedule the oldest pending job."""
    state = env._state
    pending_jobs = state.get("pending_jobs", [])
    if not pending_jobs:
        return env.action_space.n - 1  # WAIT

    sorted_jobs = sorted(pending_jobs, key=lambda j: j.submit_time)
    resource = state.get("resource")
    
    for i, job in enumerate(sorted_jobs):
        if resource.can_allocate(job.requested_resources):
            return i

    return len(sorted_jobs)  # WAIT


def sjf_policy(env: PyBatGymEnv) -> int:
    """Shortest-Job-First: schedule the job with smallest requested walltime."""
    state = env._state
    pending_jobs = state.get("pending_jobs", [])
    if not pending_jobs:
        return env.action_space.n - 1

    resource = state.get("resource")
    sorted_by_submit = sorted(pending_jobs, key=lambda j: j.submit_time)
    
    candidates = [
        (i, job) for i, job in enumerate(sorted_by_submit)
        if resource.can_allocate(job.requested_resources)
    ]

    if not candidates:
        return len(sorted_by_submit)

    best_idx, _ = min(candidates, key=lambda pair: pair[1].requested_walltime)
    return best_idx


def easy_backfilling_policy(env: PyBatGymEnv) -> int:
    """EASY Backfilling heuristic policy."""
    state = env._state
    pending_jobs = state.get("pending_jobs", [])
    if not pending_jobs:
        return env.action_space.n - 1
        
    resource = state.get("resource")
    sorted_jobs = sorted(pending_jobs, key=lambda j: j.submit_time)
    first_job = sorted_jobs[0]
    
    if resource.can_allocate(first_job.requested_resources):
        return 0  # Schedule the first job
        
    running_jobs = getattr(env._adapter, "_running_jobs", [])
    if not running_jobs:
        return len(sorted_jobs)
        
    sim_time = env._adapter.current_time
    expected_completions = [
        (rj.submit_time + rj.waiting_time + rj.requested_walltime, rj.requested_resources) 
        for rj in running_jobs
    ]
    expected_completions.sort()
    
    freed_cores = resource.free_cores
    shadow_time = sim_time
    
    for comp_time, cores in expected_completions:
        shadow_time = comp_time
        freed_cores += cores
        if freed_cores >= first_job.requested_resources:
            break
            
    # Backfill
    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        if job.requested_resources <= resource.free_cores:
            expected_finish = sim_time + job.requested_walltime
            if expected_finish <= shadow_time:
                return i

    return len(sorted_jobs)


def run_baseline(
    env: PyBatGymEnv,
    policy_fn: Callable[[PyBatGymEnv], int],
    num_episodes: int = 10,
) -> dict[str, float]:
    """Run a baseline policy and collect aggregate metrics."""
    total_reward = 0.0
    total_util = 0.0
    total_wait = 0.0
    total_slowdown = 0.0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action = policy_fn(env)
            action = min(action, env.action_space.n - 1)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        total_reward += ep_reward

        completed = env._adapter.get_completed_jobs()
        if completed:
            makespan = env._adapter.get_current_time()
            total_cores = env._state.get("resource").total_cores
            
            wait_sum = sum(j.waiting_time for j in completed)
            slowdown_sum = sum(j.bounded_slowdown for j in completed)
            busy_time_sum = sum(j.actual_runtime * j.requested_resources for j in completed)

            total_wait += wait_sum / len(completed)
            total_slowdown += slowdown_sum / len(completed)
            if makespan > 0 and total_cores > 0:
                total_util += busy_time_sum / (makespan * total_cores)

    n = max(num_episodes, 1)
    return {
        "avg_reward": total_reward / n,
        "avg_utilization": total_util / n,
        "avg_waiting_time": total_wait / n,
        "avg_slowdown": total_slowdown / n,
    }


class BenchmarkPlugin(Plugin):
    """Plugin to automatically run heuristics and compare with RL agent logs."""

    @property
    def name(self) -> str:
        return "benchmark"

    def __init__(self, run_on_close: bool = True):
        self.run_on_close = run_on_close
        
    def on_close(self) -> None:
        if self.run_on_close:
            print("[BenchmarkPlugin] Environment closed. Run `run_baseline()` manually for comparisons.")
