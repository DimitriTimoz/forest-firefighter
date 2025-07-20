#!/usr/bin/env python3
"""Simple demo script for Forest Fire RL Environment"""

import numpy as np
from forest_fire_rl import ForestFireEnv

def random_policy(obs):
    """Random policy for demonstration"""
    return np.random.randint(0, 5)  # 5 actions: up, down, left, right, wait

def demo():
    """Run a simple demo"""
    print("Forest Fire RL Environment Demo")
    
    env = ForestFireEnv(grid_size=15, fire_spread_prob=0.15)
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = random_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Render every few steps
        if step % 5 == 0:
            env.render()
            print(f"Step {step}: Fire={info['fire_count']}, Forest={info['forest_count']}, Reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    demo()
