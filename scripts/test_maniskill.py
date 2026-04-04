#!/usr/bin/env python3
"""Quick test of ManiSkill3 YCB environment."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gymnasium as gym
import mani_skill.envs

print("Testing PickSingleYCB-v1...")
env = gym.make('PickSingleYCB-v1', render_mode='rgb_array', obs_mode='rgbd', num_envs=1, sim_backend='gpu')
obs, info = env.reset()
print("YCB OK, sensor keys:", list(obs['sensor_data'].keys()))
for k, v in obs['sensor_data'].items():
    if 'rgb' in v:
        print(f"  {k} rgb: {v['rgb'].shape}")
env.close()
print("DONE")
