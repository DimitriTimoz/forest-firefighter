#!/usr/bin/env python3
"""
Setup script for Forest Fire RL Environment
Installs dependencies and validates the environment
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Successfully installed all requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False


def validate_environment():
    """Validate that the environment works correctly"""
    print("\nValidating environment...")
    
    try:
        # Test basic imports
        import numpy as np
        import gymnasium as gym
        import matplotlib.pyplot as plt
        print("✓ All imports successful")
        
        # Test environment creation
        from forest_fire_env import ForestFireEnv
        from env_configs import get_config
        
        config = get_config('easy')
        env = ForestFireEnv(**config.to_dict())
        print("✓ Environment created successfully")
        
        # Test environment reset and step
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Environment reset and step successful")
        
        # Test policy functions
        from training_utils import random_policy, heuristic_policy
        
        random_action = random_policy(obs)
        heuristic_action = heuristic_policy(obs)
        print("✓ Policy functions working")
        
        env.close()
        print("✓ Environment validation complete")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def run_quick_demo():
    """Run a quick demonstration"""
    print("\nRunning quick demonstration...")
    
    try:
        from forest_fire_env import ForestFireEnv
        from env_configs import get_config
        from training_utils import heuristic_policy, run_episode
        
        # Create environment
        config = get_config('easy')
        env = ForestFireEnv(**config.to_dict())
        
        # Run one episode
        result = run_episode(env, heuristic_policy)
        
        print(f"✓ Demo episode completed:")
        print(f"  - Total reward: {result['total_reward']:.2f}")
        print(f"  - Episode length: {result['steps']} steps")
        print(f"  - Forest preserved: {result['final_info']['forest_preservation_ratio']:.1%}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        return False


def main():
    """Main setup function"""
    print("Forest Fire RL Environment Setup")
    print("=================================")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not install_requirements():
        print("\nSetup failed at requirements installation.")
        sys.exit(1)
    
    # Validate environment
    if not validate_environment():
        print("\nSetup failed at environment validation.")
        sys.exit(1)
    
    # Run quick demo
    if not run_quick_demo():
        print("\nSetup completed but demo failed. Environment may still work.")
    
    print("\n" + "="*50)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python demo.py' to see a full demonstration")
    print("2. Check the examples in the demo script for usage patterns")
    print("3. Start developing your RL agent!")
    print("\nEnvironment features:")
    print("- Efficient numpy-based grid operations")
    print("- Multiple difficulty configurations")
    print("- Vectorized environments for parallel training")
    print("- Built-in benchmarking and logging tools")
    print("- Customizable fire spread, wind effects, and rewards")


if __name__ == "__main__":
    main()
