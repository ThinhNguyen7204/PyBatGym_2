"""Test script for running PyBatGym with the RealBatsimAdapter.

NOTE: Run this inside the Docker/Ubuntu container where batsim and pybatsim
are natively installed.

Usage (inside container):
    $ export PATH=/workspace/batsim_data/result/bin:$PATH
    $ python examples/test_real.py
"""

from pathlib import Path

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv
from pybatgym.plugins.benchmark import run_baseline, sjf_policy

# Resolve paths relative to project root (works on Windows and inside Docker)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BATSIM_DATA = _PROJECT_ROOT / "batsim_data"
_WORKLOAD = _PROJECT_ROOT / "data" / "workloads" / "tiny_workload.json"


def main() -> None:
    print("--- Testing PyBatGym with REAL BatSim (ZeroMQ) ---")

    config = PyBatGymConfig()
    config.mode = "real"
    config.workload.source = "trace"
    config.workload.trace_path = str(_WORKLOAD)
    config.platform.total_nodes = 5
    config.platform.cores_per_node = 1

    print(f"[Init] Workload : {_WORKLOAD}")
    print(f"[Init] BatSim   : {_BATSIM_DATA / 'result/bin/batsim'}")

    env = PyBatGymEnv(config=config)

    print("\n[Run] Running Shortest-Job-First Baseline...")
    try:
        metrics = run_baseline(env, sjf_policy, num_episodes=1)
        print("\n--- RESULTS ---")
        print(metrics)
    except Exception as exc:
        print(f"\n[Error] {exc}")
        print("Hint: ensure 'batsim' is in PATH and pyzmq/pybatsim are installed.")
    finally:
        env.close()

    print("Test finished.")


if __name__ == "__main__":
    main()
