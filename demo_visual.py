#!/usr/bin/env python3
"""
Visual demo of the Forest Fire RL Environment
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from forest_fire_rl import (
    ForestFireEnv, 
    EasyConfig, 
    TorchRandomAgent, 
    SimpleVisualizer, 
    TorchHeuristicAgent
)
import torch

def demo_simple_visualization():
    """Demo with simple visualizer"""
    print("🎨 Simple Visualization Demo")
    print("=" * 40)
    
    # Create environment and agent
    config = EasyConfig()
    env = ForestFireEnv(
        grid_size=config.grid_size,
        initial_fire_positions=config.initial_fire_positions,
        max_steps=config.max_steps,
        fire_spread_prob=config.fire_spread_prob,
        render_mode=None  # We'll use external visualization
    )
    
    # Use heuristic agent for more interesting behavior
    agent = TorchHeuristicAgent(env.action_space)
    
    # Create visualizer
    visualizer = SimpleVisualizer(figsize=(10, 8))
    
    # Run episode
    total_reward = visualizer.run_episode(
        env=env,
        agent=agent,
        max_steps=50,
        delay=0.8,
        verbose=True
    )
    
    print(f"\n🎯 Demo completed with total reward: {total_reward:.2f}")
    
    # Keep plot open for a moment
    input("Press Enter to close...")
    visualizer.close()
    env.close()

def main():
    """Main demo function"""
    print("🔥 Forest Fire RL - Visualization Demo")
    print("🎮 Demonstrating improved rendering and PyTorch integration")
    print("🤖 Using heuristic agent for intelligent behavior")
    print()
    
    try:
        demo_simple_visualization()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
