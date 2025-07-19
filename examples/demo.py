#!/usr/bin/env python3
"""
Forest Fire Environment Demo
Demonstrates the usage of the forest fire environment with different configurations
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Import our custom modules
from forest_fire_env import ForestFireEnv
from env_configs import get_config, create_custom_config
from training_utils import (
    VectorizedForestFireEnv, 
    EnvironmentBenchmark, 
    EpisodeLogger,
    random_policy, 
    heuristic_policy,
    run_episode,
    evaluate_policy
)


def demo_basic_usage():
    """Demonstrate basic environment usage"""
    print("=== Basic Environment Usage ===")
    
    # Create environment with default settings
    env = ForestFireEnv(grid_size=15, render_mode="human")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Initial forest count: {info['forest_count']}")
    print(f"Initial fire count: {info['fire_count']}")
    
    # Run a few steps
    for step in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, "
              f"Fires={info['fire_count']}, Forest={info['forest_count']}")
        
        if terminated:
            print("Episode terminated!")
            break
    
    env.close()


def demo_configurations():
    """Demonstrate different environment configurations"""
    print("\n=== Environment Configurations ===")
    
    configs = ['easy', 'medium', 'hard']
    
    for config_name in configs:
        print(f"\n--- {config_name.upper()} Configuration ---")
        config = get_config(config_name)
        env = ForestFireEnv(**config.to_dict())
        
        obs, info = env.reset(seed=42)
        print(f"Grid size: {config.grid_size}x{config.grid_size}")
        print(f"Initial fires: {len(config.initial_fire_positions)}")
        print(f"Fire spread probability: {config.fire_spread_prob}")
        print(f"Wind: {config.wind_direction} (strength: {config.wind_strength})")
        print(f"Max steps: {config.max_steps}")
        
        env.close()


def demo_custom_scenario():
    """Demonstrate creating custom scenarios"""
    print("\n=== Custom Scenario ===")
    
    # Create a custom scenario with specific fire positions
    fire_positions = [(5, 5), (10, 10), (5, 15)]
    config = create_custom_config(
        grid_size=20,
        fire_positions=fire_positions,
        fire_spread_prob=0.12,
        wind_direction=(1, 0),  # Wind blowing right
        wind_strength=0.4,
        max_steps=250
    )
    
    env = ForestFireEnv(**config.to_dict())
    obs, info = env.reset(seed=42)
    
    print(f"Custom scenario created with {len(fire_positions)} fires")
    print(f"Fire positions: {fire_positions}")
    print(f"Initial state - Fires: {info['fire_count']}, Forest: {info['forest_count']}")
    
    env.close()


def demo_policies():
    """Demonstrate different policies"""
    print("\n=== Policy Comparison ===")
    
    config = get_config('medium')
    
    # Test random policy
    print("Testing random policy...")
    random_stats = evaluate_policy(config, random_policy, num_episodes=20)
    print(f"Random Policy - Mean Reward: {random_stats['mean_reward']:.2f}, "
          f"Success Rate: {random_stats['success_rate']:.2%}")
    
    # Test heuristic policy
    print("Testing heuristic policy...")
    heuristic_stats = evaluate_policy(config, heuristic_policy, num_episodes=20)
    print(f"Heuristic Policy - Mean Reward: {heuristic_stats['mean_reward']:.2f}, "
          f"Success Rate: {heuristic_stats['success_rate']:.2%}")
    
    # Compare policies
    improvement = (heuristic_stats['mean_reward'] - random_stats['mean_reward']) / abs(random_stats['mean_reward']) * 100
    print(f"Heuristic policy improvement: {improvement:.1f}%")


def demo_vectorized_environment():
    """Demonstrate vectorized environment for parallel training"""
    print("\n=== Vectorized Environment ===")
    
    config = get_config('easy')
    num_envs = 4
    
    vec_env = VectorizedForestFireEnv(num_envs, config, seed=42)
    observations, infos = vec_env.reset()
    
    print(f"Created {num_envs} parallel environments")
    print(f"Initial fire counts: {[info['fire_count'] for info in infos]}")
    
    # Run some steps
    num_steps = 50
    start_time = time.time()
    
    for step in range(num_steps):
        # Random actions for all environments
        actions = [vec_env.action_space.sample() for _ in range(num_envs)]
        observations, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    
    end_time = time.time()
    
    # Performance stats
    stats = vec_env.get_performance_stats()
    print(f"Completed {num_steps} steps in {end_time - start_time:.2f}s")
    print(f"Steps per second: {stats['steps_per_second']:.1f}")
    print(f"Total episodes completed: {stats['total_episodes']}")


def demo_benchmark():
    """Demonstrate environment benchmarking"""
    print("\n=== Performance Benchmark ===")
    
    config = get_config('medium')
    benchmark = EnvironmentBenchmark(config)
    
    print("Benchmarking single environment...")
    single_stats = benchmark.benchmark_single_env(num_steps=500, num_runs=3)
    print(f"Single env: {single_stats['steps_per_second']:.1f} steps/sec")
    
    print("Benchmarking vectorized environment...")
    vec_stats = benchmark.benchmark_vectorized_env(num_envs=4, num_steps=500)
    print(f"Vectorized (4 envs): {vec_stats['effective_steps_per_second']:.1f} effective steps/sec")
    
    if 'speedup_vs_single' in vec_stats:
        print(f"Speedup: {vec_stats['speedup_vs_single']:.1f}x")


def demo_episode_logging():
    """Demonstrate episode logging during training"""
    print("\n=== Episode Logging ===")
    
    config = get_config('easy')
    env = ForestFireEnv(**config.to_dict())
    logger = EpisodeLogger(log_frequency=10)
    
    print("Running 25 episodes with logging...")
    
    for episode in range(25):
        result = run_episode(env, random_policy)
        logger.log_episode(
            result['total_reward'],
            result['steps'],
            result['final_info']
        )
    
    # Final statistics
    final_stats = logger.get_stats()
    print(f"Final stats after {final_stats['total_episodes']} episodes:")
    print(f"  Average reward: {final_stats['avg_reward']:.2f}")
    print(f"  Average forest saved: {final_stats['avg_forest_saved']:.2%}")
    
    env.close()


def demo_visualization():
    """Demonstrate environment visualization"""
    print("\n=== Visualization Demo ===")
    print("Running environment with visualization...")
    print("Close the plot window to continue the demo.")
    
    config = get_config('easy')
    env = ForestFireEnv(**config.to_dict(), render_mode="human")
    
    obs, info = env.reset(seed=42)
    env.render()
    
    # Run episode with heuristic policy and visualization
    for step in range(30):
        action = heuristic_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        time.sleep(0.2)  # Slow down for visualization
        
        if terminated:
            print(f"Episode finished at step {step+1}")
            break
    
    env.close()
    plt.close('all')  # Clean up matplotlib


def main():
    """Run all demonstrations"""
    print("Forest Fire Environment Demonstration")
    print("====================================")
    
    try:
        demo_basic_usage()
        demo_configurations()
        demo_custom_scenario()
        demo_policies()
        demo_vectorized_environment()
        demo_benchmark()
        demo_episode_logging()
        
        # Ask user if they want to see visualization
        try:
            response = input("\nWould you like to see the visualization demo? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                demo_visualization()
        except KeyboardInterrupt:
            print("\nSkipping visualization demo.")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install gymnasium numpy matplotlib")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
