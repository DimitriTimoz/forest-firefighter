#!/usr/bin/env python3
"""
Command-line interface for the Forest Fire RL environment
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from forest_fire_rl import ForestFireEnv, EasyConfig, MediumConfig, HardConfig
from forest_fire_rl.agents import TorchRandomAgent
from forest_fire_rl.visualization import AdvancedVisualizer
import torch


def demo_environment(config_name="easy", steps=100, render=True, num_firefighters=1):
    """Run a demonstration of the environment"""
    
    # Select configuration
    config_map = {
        "easy": EasyConfig(),
        "medium": MediumConfig(), 
        "hard": HardConfig()
    }
    
    config = config_map.get(config_name, EasyConfig())
    
    print(f"Forest Fire RL Environment Demo")
    print(f"Configuration: {config_name.upper()}")
    print(f"Grid size: {config.grid_size}x{config.grid_size}")
    print(f"Max steps: {config.max_steps}")
    print(f"Fire spread prob: {config.fire_spread_prob}")
    print(f"Firefighters: {num_firefighters}")
    print("-" * 50)
    
    # Create environment
    env = ForestFireEnv(
        grid_size=config.grid_size,
        initial_fire_positions=config.initial_fire_positions,
        max_steps=config.max_steps,
        fire_spread_prob=config.fire_spread_prob,
        wind_direction=config.wind_direction,
        wind_strength=config.wind_strength,
        render_mode="human" if render else None,
        reward_structure=config.reward_structure,
        num_firefighters=num_firefighters
    )
    
    # Create agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = TorchRandomAgent(
        action_space=env.action_space,
        device=device
    )
    
    # Run episode
    observation, info = env.reset()
    total_reward = 0
    episode_steps = 0
    
    print("Starting episode...")
    
    try:
        for step in range(steps):
            # Get action from agent
            action = agent.act(observation)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_steps += 1
            
            # Render if enabled
            if render:
                env.render()
            
            # Print progress
            if step % 20 == 0:
                fire_count = info.get('fire_count', 0)
                forest_count = info.get('forest_count', 0)
                print(f"Step {step:3d}: Fire={fire_count:3d}, Forest={forest_count:3d}, Reward={reward:6.2f}")
            
            # Check if episode ended
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        print(f"\nEpisode completed!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final fire count: {info.get('fire_count', 0)}")
        print(f"Final forest count: {info.get('forest_count', 0)}")
        print(f"Forest preservation: {info.get('forest_preservation_ratio', 0):.2%}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        env.close()


def benchmark_environment(config_name="easy", num_episodes=100):
    """Benchmark environment performance"""
    
    config_map = {
        "easy": EasyConfig(),
        "medium": MediumConfig(),
        "hard": HardConfig()
    }
    
    config = config_map.get(config_name, EasyConfig())
    
    print(f"Benchmarking Forest Fire Environment")
    print(f"Configuration: {config_name.upper()}")
    print(f"Episodes: {num_episodes}")
    print("-" * 50)
    
    from time import time
    
    # Create environment without rendering
    env = ForestFireEnv(
        grid_size=config.grid_size,
        initial_fire_positions=config.initial_fire_positions,
        max_steps=config.max_steps,
        fire_spread_prob=config.fire_spread_prob,
        wind_direction=config.wind_direction,
        wind_strength=config.wind_strength,
        render_mode=None,
        reward_structure=config.reward_structure
    )
    
    # Create agent
    agent = TorchRandomAgent(env.action_space)
    
    start_time = time()
    total_steps = 0
    total_reward = 0
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
        
        if (episode + 1) % 20 == 0:
            avg_reward = total_reward / (episode + 1)
            print(f"Episode {episode + 1:3d}: Avg reward = {avg_reward:6.2f}")
    
    elapsed_time = time() - start_time
    steps_per_second = total_steps / elapsed_time
    episodes_per_second = num_episodes / elapsed_time
    
    print(f"\nPerformance Results:")
    print(f"Total steps: {total_steps:,}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Steps per second: {steps_per_second:,.0f}")
    print(f"Episodes per second: {episodes_per_second:.1f}")
    print(f"Average reward: {total_reward/num_episodes:.2f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Forest Fire RL Environment CLI")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run environment demonstration')
    demo_parser.add_argument('--config', choices=['easy', 'medium', 'hard'], 
                           default='easy', help='Environment configuration')
    demo_parser.add_argument('--steps', type=int, default=100, 
                           help='Maximum steps to run')
    demo_parser.add_argument('--no-render', action='store_true', 
                           help='Disable visualization')
    demo_parser.add_argument('--firefighters', type=int, default=1,
                           help='Number of firefighters (1-5)')                         
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark environment performance')
    bench_parser.add_argument('--config', choices=['easy', 'medium', 'hard'], 
                            default='easy', help='Environment configuration')
    bench_parser.add_argument('--episodes', type=int, default=100, 
                            help='Number of episodes to run')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        demo_environment(
            config_name=args.config, 
            steps=args.steps, 
            render=not args.no_render,
            num_firefighters=min(args.firefighters, 5)  # Limit to 5 firefighters
        )
    elif args.command == 'benchmark':
        benchmark_environment(
            config_name=args.config,
            num_episodes=args.episodes
        )
    elif args.command == 'version':
        print("Forest Fire RL Environment v1.0.0")
        print("PyTorch version:", torch.__version__)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
