"""PyBatGym Quickstart Example.

Demonstrates:
1. Creating the environment
2. Running an episode with random actions
3. Running baselines (FCFS, SJF)
4. Basic PPO training (if stable-baselines3 installed)
"""

from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv
from pybatgym.plugins.benchmark import fcfs_policy, run_baseline, sjf_policy
from pybatgym.plugins.logger import CSVLoggerPlugin


def main() -> None:
    config = PyBatGymConfig()
    config.workload.num_jobs = 50
    config.workload.seed = 42
    config.episode.max_steps = 1000

    # --- 1. Random Agent ---
    print("=" * 50)
    print("🎲 Random Agent")
    print("=" * 50)

    env = PyBatGymEnv(config=config, render_mode="human")
    env.register_plugin(CSVLoggerPlugin(output_dir="logs"))

    obs, info = env.reset(seed=42)
    total_reward = 0.0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if steps % 100 == 0:
            env.render()

        if terminated or truncated:
            break

    print(f"\n✅ Episode finished: {steps} steps, reward={total_reward:.4f}")
    print(f"   Completed: {info['completed']} jobs, Utilization: {info['utilization']:.1%}")
    env.close()

    # --- 2. Baselines ---
    print("\n" + "=" * 50)
    print("📊 Baseline Comparison")
    print("=" * 50)

    env = PyBatGymEnv(config=config)

    fcfs_results = run_baseline(env, fcfs_policy, num_episodes=5)
    print(f"\nFCFS: reward={fcfs_results['avg_reward']:.4f}, "
          f"util={fcfs_results['avg_utilization']:.1%}, "
          f"wait={fcfs_results['avg_waiting_time']:.1f}")

    sjf_results = run_baseline(env, sjf_policy, num_episodes=5)
    print(f"SJF:  reward={sjf_results['avg_reward']:.4f}, "
          f"util={sjf_results['avg_utilization']:.1%}, "
          f"wait={sjf_results['avg_waiting_time']:.1f}")

    env.close()

    # --- 3. PPO Training (optional) ---
    try:
        from stable_baselines3 import PPO

        print("\n" + "=" * 50)
        print("🤖 PPO Training (2000 steps)")
        print("=" * 50)

        env = PyBatGymEnv(config=config)
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=256, batch_size=64)
        model.learn(total_timesteps=2000)

        # Evaluate
        obs, info = env.reset(seed=0)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        print(f"\n🏆 PPO eval: reward={total_reward:.4f}, "
              f"completed={info['completed']}, util={info['utilization']:.1%}")
        env.close()

    except ImportError:
        print("\n⏭️  Skipping PPO (install: pip install pybatgym[rl])")


if __name__ == "__main__":
    main()
