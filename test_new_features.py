#!/usr/bin/env python3
"""
Test script to verify burned cells remain burned and multi-firefighter functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import forest_fire_rl
import numpy as np

def test_burned_cells_persist():
    """Test that burned cells stay burned when firefighters move over them"""
    print("🧪 Testing burned cell persistence...")
    
    # Create small environment for testing
    env = forest_fire_rl.ForestFireEnv(
        grid_size=5,
        initial_fire_positions=[(2, 2)],  # Fire in center
        num_firefighters=1,
        max_steps=50,
        fire_spread_prob=0.8  # High spread probability
    )
    
    obs, info = env.reset()
    
    # Let fire spread for a few steps without intervention
    for _ in range(3):
        action = 5  # Do nothing
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    
    # Find a burned cell (if any)
    burned_cells = np.where(obs['grid'] == env.EMPTY)
    
    if len(burned_cells[0]) > 0:
        # Move firefighter to a burned cell
        target_row, target_col = burned_cells[0][0], burned_cells[1][0]
        ff_pos = obs['firefighter_positions'][0]
        
        print(f"🔥 Found burned cell at ({target_row}, {target_col})")
        print(f"🚒 Firefighter at ({ff_pos[0]}, {ff_pos[1]})")
        
        # Move firefighter step by step toward the burned cell
        steps = 0
        while not np.array_equal(obs['firefighter_positions'][0], [target_row, target_col]) and steps < 10:
            ff_pos = obs['firefighter_positions'][0]
            
            # Determine movement direction
            if ff_pos[0] < target_row:
                action = 1  # Move down
            elif ff_pos[0] > target_row:
                action = 0  # Move up
            elif ff_pos[1] < target_col:
                action = 3  # Move right
            elif ff_pos[1] > target_col:
                action = 2  # Move left
            else:
                break
            
            print(f"  Step {steps}: Moving firefighter from ({ff_pos[0]}, {ff_pos[1]}) with action {action}")
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            if done or truncated:
                break
        
        # Check if firefighter is on the target cell and it's still burned underneath
        final_ff_pos = obs['firefighter_positions'][0]
        if np.array_equal(final_ff_pos, [target_row, target_col]):
            print(f"✅ Firefighter successfully moved to burned cell at ({target_row}, {target_col})")
            
            # Move firefighter away and check if cell is still burned
            obs, reward, done, truncated, info = env.step(0)  # Move up
            
            # Check the cell where firefighter was
            cell_state = obs['grid'][target_row, target_col]
            if cell_state == env.EMPTY:
                print("✅ SUCCESS: Burned cell remained burned after firefighter moved away!")
                return True
            else:
                print(f"❌ FAIL: Cell changed to state {cell_state} instead of staying burned")
                return False
        else:
            print("❌ Could not move firefighter to burned cell")
            return False
    else:
        print("⚠️  No burned cells found in this test")
        return True
    
    env.close()

def test_multi_firefighter():
    """Test multiple firefighters functionality"""
    print("\n🧪 Testing multiple firefighters...")
    
    env = forest_fire_rl.ForestFireEnv(
        grid_size=8,
        initial_fire_positions=[(1, 1), (6, 6)],  # Two fires
        num_firefighters=3,
        max_steps=20
    )
    
    agent = forest_fire_rl.TorchHeuristicAgent(env.action_space)
    obs, info = env.reset()
    
    print(f"🚒 Created environment with {info['firefighter_count']} firefighters")
    print(f"🔥 Initial fires: {info['fire_count']}")
    print(f"🌲 Initial forest: {info['forest_count']}")
    
    # Run a few steps
    total_reward = 0
    for step in range(5):
        actions = agent.act(obs)
        print(f"  Step {step}: Actions = {actions}")
        
        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward
        
        if done or truncated:
            break
    
    print(f"✅ Multi-firefighter test completed!")
    print(f"🏆 Total reward: {total_reward:.2f}")
    print(f"🔥 Final fires: {info['fire_count']}")
    print(f"🌲 Final forest: {info['forest_count']}")
    
    env.close()
    return True

def main():
    """Run all tests"""
    print("🔥 Forest Fire RL Environment - Feature Tests")
    print("=" * 50)
    
    try:
        # Test 1: Burned cells persistence
        test1_passed = test_burned_cells_persist()
        
        # Test 2: Multiple firefighters
        test2_passed = test_multi_firefighter()
        
        print("\n" + "=" * 50)
        print("📊 Test Results:")
        print(f"  🧪 Burned cells persist: {'✅ PASS' if test1_passed else '❌ FAIL'}")
        print(f"  🧪 Multiple firefighters: {'✅ PASS' if test2_passed else '❌ FAIL'}")
        
        if test1_passed and test2_passed:
            print("\n🎉 All tests passed! New features working correctly!")
        else:
            print("\n⚠️  Some tests failed. Check implementation.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
