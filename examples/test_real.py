"""Test script for running PyBatGym with the RealBatsimAdapter.

NOTE: This script is intended to be run inside an environment where `batsim`
and `pybatsim` are natively installed (e.g., WSL Ubuntu or Linux).

Usage (inside WSL):
$ pip install pybatsim
$ python examples/test_real.py
"""

from pathlib import Path
from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv
from pybatgym.plugins.benchmark import run_baseline, sjf_policy


def main():
    print("--- Testing PyBatGym with REAL BatSim (ZeroMQ) ---")
    data_dir = Path("D:/ThinhProject/data") # Adjust to Linux path if mounted, e.g. /mnt/d/...
    
    # In WSL, Windows drives are mounted under /mnt/
    linux_data_dir = Path("/mnt/d/ThinhProject/data")
    if linux_data_dir.exists():
        data_dir = linux_data_dir

    config = PyBatGymConfig()
    config.mode = "real"  # Switch to RealBatsimAdapter
    config.workload.source = "trace"
    config.workload.trace_path = str(data_dir / "workloads" / "tiny_workload.json")
    
    # Ensure platform syncs with BatSim args (it uses small_platform.xml)
    config.platform.total_nodes = 5
    config.platform.cores_per_node = 1
    
    print(f"[Init] Creating PyBatGymEnv in REAL mode...")
    print(f"       Trace: {config.workload.trace_path}")
    env = PyBatGymEnv(config=config)
    
    print(f"\n[Run] Running Shortest-Job-First Baseline...")
    try:
        metrics = run_baseline(env, sjf_policy, num_episodes=1)
        print("\n--- RESULTS ---")
        print(metrics)
    except Exception as e:
        print(f"\n[Error] {e}")
        print("Note: Ensure 'batsim' is in your PATH and 'pybatsim' is installed.")
    
    env.close()
    print("Test finished.")


if __name__ == "__main__":
    main()
