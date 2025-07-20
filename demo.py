#!/usr/bin/env python3
"""Optimized demo script for Forest Fire RL Environment"""

import numpy as np
import torch
import time
from forest_fire_rl import ForestFireEnv
from models import DeeQModel, DuelingDQN

def random_policy(obs):
    """Random policy for demonstration"""
    return np.random.randint(0, 5)  # 5 actions: up, down, left, right, wait

def smart_policy(obs):
    """Smarter heuristic policy that moves toward fires"""
    # Extract fire locations from observation
    fire_channel = obs[2]  # Fire channel
    firefighter_channel = obs[3]  # Firefighter channel
    
    # Find firefighter position
    ff_pos = np.where(firefighter_channel == 1.0)
    if len(ff_pos[0]) == 0:
        return 4  # Wait if no firefighter found
    
    ff_x, ff_y = ff_pos[0][0], ff_pos[1][0]
    
    # Find closest fire
    fire_positions = np.where(fire_channel == 1.0)
    if len(fire_positions[0]) == 0:
        return 4  # Wait if no fires
    
    # Calculate distances to all fires
    distances = []
    for i in range(len(fire_positions[0])):
        fire_x, fire_y = fire_positions[0][i], fire_positions[1][i]
        dist = abs(ff_x - fire_x) + abs(ff_y - fire_y)  # Manhattan distance
        distances.append((dist, fire_x, fire_y))
    
    # Move toward closest fire
    _, target_x, target_y = min(distances)
    
    # Decide movement
    if ff_x > target_x:
        return 0  # Up
    elif ff_x < target_x:
        return 1  # Down
    elif ff_y > target_y:
        return 2  # Left
    elif ff_y < target_y:
        return 3  # Right
    else:
        return 4  # Wait (already at fire location)

def performance_demo():
    """Run performance demonstration"""
    print("Forest Fire RL Environment - Performance Demo")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test different grid sizes
    grid_sizes = [15, 25, 35]
    policies = [
        ("Random Policy", random_policy),
        ("Smart Heuristic", smart_policy)
    ]
    
    results = {}
    
    for grid_size in grid_sizes:
        print(f"\n--- Grid Size: {grid_size}x{grid_size} ---")
        results[grid_size] = {}
        
        for policy_name, policy_func in policies:
            print(f"\nTesting {policy_name}...")
            
            env = ForestFireEnv(grid_size=grid_size, fire_spread_prob=0.15, max_steps=200)
            
            total_rewards = []
            episode_lengths = []
            start_time = time.time()
            
            num_episodes = 10
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                total_reward = 0
                step_count = 0
                
                for step in range(200):  # Max steps per episode
                    action = policy_func(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    total_reward += reward
                    step_count += 1
                    
                    if terminated or truncated:
                        break
                
                total_rewards.append(total_reward)
                episode_lengths.append(step_count)
            
            end_time = time.time()
            env.close()
            
            # Calculate statistics
            avg_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            avg_length = np.mean(episode_lengths)
            fps = (sum(episode_lengths)) / (end_time - start_time)
            
            results[grid_size][policy_name] = {
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'avg_length': avg_length,
                'fps': fps
            }
            
            print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
            print(f"  Average Length: {avg_length:.1f} steps")
            print(f"  Performance: {fps:.2f} steps/second")
    
    # Summary
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    
    for grid_size in grid_sizes:
        print(f"\nGrid {grid_size}x{grid_size}:")
        for policy_name in ["Random Policy", "Smart Heuristic"]:
            data = results[grid_size][policy_name]
            print(f"  {policy_name:15s}: {data['avg_reward']:6.1f} reward, {data['fps']:6.1f} fps")
    
    return results

def neural_network_demo():
    """Demonstrate neural network inference performance"""
    print(f"\n{'='*50}")
    print("NEURAL NETWORK PERFORMANCE DEMO")
    print(f"{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    grid_size = 25
    batch_size = 32
    
    # Create models
    models = {
        'Standard DQN': DeeQModel(grid_size=grid_size, n_state=4, ac=5).to(device),
        'Dueling DQN': DuelingDQN(grid_size=grid_size, n_state=4, ac=5).to(device)
    }
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 4, grid_size, grid_size, device=device)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Batch size: {batch_size}")
    
    for name, model in models.items():
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        iterations_per_second = num_iterations / total_time
        samples_per_second = (num_iterations * batch_size) / total_time
        
        # Model parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n{name}:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Output shape: {output.shape}")
        print(f"  Inference speed: {iterations_per_second:.2f} batches/second")
        print(f"  Throughput: {samples_per_second:.2f} samples/second")

def interactive_demo():
    """Run an interactive demo where user can watch the agent"""
    print(f"\n{'='*50}")
    print("INTERACTIVE DEMO")
    print(f"{'='*50}")
    
    env = ForestFireEnv(grid_size=20, fire_spread_prob=0.12, max_steps=150)
    
    print("Choose a policy:")
    print("1. Random")
    print("2. Smart Heuristic")
    print("3. Both (comparison)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        policies = [("Random", random_policy)]
    elif choice == "2":
        policies = [("Smart Heuristic", smart_policy)]
    else:
        policies = [("Random", random_policy), ("Smart Heuristic", smart_policy)]
    
    for policy_name, policy_func in policies:
        print(f"\n--- Running {policy_name} Policy ---")
        input("Press Enter to start...")
        
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(150):
            action = policy_func(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # Render every few steps
            if step % 3 == 0:  # Render every 3rd step for better visibility
                env.render()
                print(f"Step {step:3d}: Action={action}, Reward={reward:5.1f}, "
                      f"Fires={info['fire_count']:2d}, Forest={info['forest_count']:3d}")
                time.sleep(0.2)  # Slow down for visibility
            
            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                break
        
        env.render()  # Final render
        print(f"Final total reward: {total_reward:.2f}")
        print(f"Final state - Fires: {info['fire_count']}, Forest: {info['forest_count']}")
        
        if len(policies) > 1:
            input("Press Enter for next policy...")
    
    env.close()

def main():
    """Main demo function"""
    print("Forest Fire RL Environment - Comprehensive Demo")
    print("Choose demo type:")
    print("1. Performance Benchmark")
    print("2. Neural Network Performance")
    print("3. Interactive Demo")
    print("4. All Demos")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1" or choice == "4":
        performance_demo()
    
    if choice == "2" or choice == "4":
        neural_network_demo()
    
    if choice == "3" or choice == "4":
        interactive_demo()
    
    print(f"\n{'='*50}")
    print("Demo completed!")
    print("Key optimizations applied:")
    print("✓ Vectorized environment operations")
    print("✓ Optimized neural network architectures")
    print("✓ Efficient observation encoding")
    print("✓ Smart heuristic policies")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
