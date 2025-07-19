#!/usr/bin/env python3
"""Setup and validation script for Forest Fire RL Environment"""

import subprocess
import sys

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed!")

def test_environment():
    """Test the environment"""
    try:
        from forest_fire_rl import ForestFireEnv
        
        print("Creating environment...")
        env = ForestFireEnv(grid_size=10)
        
        print("Testing reset...")
        obs, info = env.reset()
        print(f"✓ Reset successful - Grid shape: {obs.shape}")
        
        print("Testing step...")
        obs, reward, terminated, truncated, info = env.step(4)  # suppress action
        print(f"✓ Step successful - Reward: {reward:.2f}")
        
        env.close()
        print("✓ Environment test passed!")
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        install_dependencies()
        if test_environment():
            print("\n🎉 Setup complete! Run 'python demo.py' to see it in action.")
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Setup error: {e}")
        sys.exit(1)
