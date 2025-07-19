# Forest Fire RL Environment

A high-performance Gymnasium-compatible reinforcement learning environment for forest fire simulation and firefighting agent training.

## Features

- **Fast 2D Grid Simulation**: Numpy-based operations with 1,000-3,000+ steps/second
- **Realistic Fire Dynamics**: Configurable spread patterns with wind effects
- **Multiple Firefighters**: Support for coordinated multi-agent scenarios
- **Pre-configured Scenarios**: Easy to hard difficulty levels
- **Vectorized Training**: Parallel environment execution

## Quick Start

```bash
pip install -r requirements.txt
python setup.py  # Validate installation
```

```python
from forest_fire_rl.core.environment import ForestFireEnv
from forest_fire_rl.configs.env_configs import get_config

# Create environment
config = get_config('medium')
env = ForestFireEnv(**config.to_dict())

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
env.close()
```

## Environment Details

**State Space:** 2D grid with cells: Empty(0), Forest(1), Fire(2), Firefighter(3)

**Action Space:** Move Up/Down/Left/Right, Suppress Fire (3x3 area), Do Nothing

**Observation:**
```python
{
    'grid': Box(shape=(grid_size, grid_size), dtype=int32),
    'firefighter_pos': Box(shape=(2,), dtype=int32), 
    'steps_remaining': Box(shape=(), dtype=int32)
}
```

**Rewards:** Fire Suppressed(+10), Forest Saved(+1), Forest Burned(-2), Time Penalty(-0.1)

## Configuration

```python
from forest_fire_rl.configs.env_configs import get_config, create_custom_config

# Pre-defined scenarios
config = get_config('easy')     # Small fires, slow spread
config = get_config('medium')   # Moderate challenge  
config = get_config('hard')     # Multiple fires, wind effects

# Custom configuration
config = create_custom_config(
    grid_size=25,
    num_fires=3,
    num_firefighters=2,  # Multi-agent support
    fire_spread_prob=0.15,
    wind_direction=(1, 1),
    max_steps=300
)
```

## Training & Integration

```python
# Stable Baselines3 integration
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return ForestFireEnv(**config.to_dict())

env = DummyVecEnv([make_env for _ in range(4)])
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

**Performance:** 1,000-3,000 steps/second (single env), 3,000-8,000 effective steps/second (vectorized)

## CLI Commands

```bash
# Demo with visualization
python -m forest_fire_rl.cli demo --firefighters 2

# Benchmark performance  
python -m forest_fire_rl.cli benchmark --config medium

# Quick test
python setup.py
```
