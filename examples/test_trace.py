"""Quick check script for loading custom traces into PyBatGym."""

from pathlib import Path

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv
from pybatgym.plugins.benchmark import run_baseline, sjf_policy, easy_backfilling_policy


def main():
    print("--- Testing PyBatGym with custom JSON trace ---")
    data_dir = Path("D:/ThinhProject/data")
    
    config = PyBatGymConfig()
    config.platform.total_nodes = 5
    config.platform.cores_per_node = 1
    config.workload.source = "trace"
    config.workload.trace_path = str(data_dir / "workloads" / "tiny_workload.json")
    
    env = PyBatGymEnv(config=config)
    
    print(f"\nRunning SJF Baseline on trace {config.workload.trace_path}...")
    metrics_sjf = run_baseline(env, sjf_policy, num_episodes=1)
    print("\nMetrics (SJF):")
    print(metrics_sjf)

    print(f"\nRunning EASY Backfilling Baseline on trace {config.workload.trace_path}...")
    metrics_easy = run_baseline(env, easy_backfilling_policy, num_episodes=1)
    print("\nMetrics (EASY):")
    print(metrics_easy)
    
    env.close()

if __name__ == "__main__":
    main()
