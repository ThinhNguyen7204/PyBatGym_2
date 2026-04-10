from pybatgym.config.base_config import PyBatGymConfig
from pybatgym.env import PyBatGymEnv
from stable_baselines3 import PPO
from examples.train_ppo_phase1b import make_env, evaluate_agent, run_baseline, fcfs_policy, easy_backfilling_policy

print('Evaluating models/ppo_phase1b.zip...')
env = make_env('/workspace/data/workloads/medium_workload.json')
model = PPO.load('models/ppo_phase1b.zip', env=env)

print('Running PPO...')
ppo_m = evaluate_agent(env, model, num_episodes=5)
print(f'PPO : wait={ppo_m["avg_waiting_time"]:.1f} slowdown={ppo_m["avg_slowdown"]:.2f}')

print('Running FCFS...')
fcfs_m = run_baseline(env, fcfs_policy, num_episodes=3)
print(f'FCFS: wait={fcfs_m["avg_waiting_time"]:.1f} slowdown={fcfs_m["avg_slowdown"]:.2f}')

print('Running EASY...')
easy_m = run_baseline(env, easy_backfilling_policy, num_episodes=3)
print(f'EASY: wait={easy_m["avg_waiting_time"]:.1f} slowdown={easy_m["avg_slowdown"]:.2f}')
env.close()
