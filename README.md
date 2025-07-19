# Forest Fire RL Environment

A minimalist Gymnasium-compatible reinforcement learning environment for forest fire simulation.

## Features

- **Simple 2D Grid**: Forest fire spreading with basic dynamics
- **Gymnasium Compatible**: Standard RL environment interface  
- **Basic Visualization**: Matplotlib-based rendering
- **Minimal Dependencies**: Only numpy, gymnasium, matplotlib

## Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

## Usage

```python
from forest_fire_rl import ForestFireEnv

env = ForestFireEnv(grid_size=15, fire_spread_prob=0.1)
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # 0-5: up, down, left, right, suppress, wait
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        break

env.close()
```

## Environment

- **Grid**: 0=empty, 1=forest, 2=fire, 3=firefighter
- **Actions**: Move (0-3), Suppress fire (4), Wait (5)  
- **Reward**: +10 for suppressing fire, +0.1 per forest cell, -0.2 per fire
- **Goal**: Extinguish all fires while preserving forest

Total code: ~200 lines
