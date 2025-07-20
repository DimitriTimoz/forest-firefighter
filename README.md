# Forest Fire Fighting with Reinforcement Learning 🔥🚁

## Objective

This project models forest fire fighting using reinforcement learning techniques. The goal is to train an intelligent agent (firefighter) to effectively combat and extinguish forest fires in a simulated 2D grid environment.

## Overview

The project implements a Deep Q-Network (DQN) agent that learns to navigate a forest environment and strategically suppress fires before they spread uncontrollably. The agent must balance between moving efficiently toward fires and maximizing fire suppression effectiveness.

## Environment

### Forest Fire Environment Features
- **Grid-based simulation**: 2D grid representing forest terrain
- **Dynamic fire spreading**: Fires spread probabilistically to neighboring forest cells
- **Firefighter agent**: Controllable agent that can move and suppress fires
- **Real-time visualization**: Visual rendering of the environment state

### State Space
- **4-channel observation**: One-hot encoded representation
  - Channel 0: Empty/burned areas
  - Channel 1: Forest areas
  - Channel 2: Active fires
  - Channel 3: Firefighter position

### Action Space
- **5 discrete actions**:
  - 0: Move up
  - 1: Move down
  - 2: Move left
  - 3: Move right
  - 4: Wait/stay in place


### Training Algorithm
- **Algorithm**: Deep Q-Learning with Experience Replay
- **Target Network**: Soft updates with τ = 0.01
- **Experience Replay**: Buffer size of 10,000 transitions
- **Exploration**: ε-greedy with decay from 0.9 to 0.05

## Key Features

### Fire Suppression Mechanics
- **Suppression radius**: 5×5 area around firefighter
- **Immediate effect**: Fires are extinguished to burned areas
- **Strategic positioning**: Agent must position itself optimally

### Fire Spreading Dynamics
- **Kernel-based spreading**: Probabilistic fire propagation using convolution
- **Balanced gameplay**: Fire spreads every 2 steps (not every step)
- **Realistic behavior**: Higher spread probability near existing fires

## Future Improvements

- [ ] Multi-agent coordination (multiple firefighters)
- [ ] Heterogeneous terrain (water bodies, roads, urban areas)
- [ ] Wind effects on fire spreading
- [ ] Resource constraints (water/fuel limitations)
- [ ] Real-world fire behavior modeling
- [ ] Climate change effects on fire dynamics

---

*Simulating intelligent forest fire fighting to protect our natural environments* 🌲🔥🚁
