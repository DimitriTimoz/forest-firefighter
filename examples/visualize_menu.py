#!/usr/bin/env python3
"""
Forest Fire Environment - Visualization Menu
Choose between different visualization options
"""

import sys

def show_menu():
    """Display the visualization menu"""
    print("🔥 Forest Fire RL Environment - PyTorch Visualization 🔥")
    print("=" * 60)
    print("Choose your visualization option:")
    print()
    print("1. 🎬 Simple Visualization (Auto-play one episode)")
    print("2. 🎮 Interactive Visualization (Full controls)")  
    print("3. 📊 Advanced Analysis Visualization")
    print("4. 📝 Show environment info")
    print("5. ❌ Exit")
    print()

def run_simple():
    """Run simple visualization"""
    print("Starting simple visualization...")
    import subprocess
    subprocess.run([
        "/Users/dimitri/Documents/Projects/AI/RL/forest-firefighter/.venv/bin/python", 
        "simple_visual.py"
    ])

def run_interactive():
    """Run interactive visualization"""
    print("Starting interactive visualization...")
    print("Use the buttons and sliders to control the environment!")
    import subprocess
    subprocess.run([
        "/Users/dimitri/Documents/Projects/AI/RL/forest-firefighter/.venv/bin/python", 
        "interactive_visual.py"
    ])

def run_advanced():
    """Run advanced visualization"""
    print("Starting advanced visualization...")
    import subprocess
    subprocess.run([
        "/Users/dimitri/Documents/Projects/AI/RL/forest-firefighter/.venv/bin/python", 
        "visualize_torch.py"
    ])

def show_info():
    """Show environment information"""
    print("\n📊 Environment Information")
    print("=" * 30)
    
    import torch
    from forest_fire_env import ForestFireEnv
    from env_configs import get_config
    from interactive_visual import InteractiveRandomAgent
    
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🎯 Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Test different configurations
    configs = ['easy', 'medium', 'hard']
    
    for config_name in configs:
        config = get_config(config_name)
        print(f"\n--- {config_name.upper()} Configuration ---")
        print(f"Grid size: {config.grid_size}x{config.grid_size}")
        print(f"Initial fires: {len(config.initial_fire_positions)}")
        print(f"Fire spread probability: {config.fire_spread_prob}")
        print(f"Wind: {config.wind_direction} (strength: {config.wind_strength})")
        print(f"Max steps: {config.max_steps}")
    
    # Test agent
    agent = InteractiveRandomAgent()
    print(f"\n🤖 PyTorch Agent Info:")
    print(f"Parameters: {sum(p.numel() for p in agent.parameters())}")
    print(f"Action space: 6 discrete actions")
    print(f"Ready for RL training: ✅")
    
    print(f"\n🚀 Visualization Features:")
    print(f"• Real-time fire spread simulation")
    print(f"• Interactive controls and sliders") 
    print(f"• Multiple difficulty levels")
    print(f"• Performance monitoring")
    print(f"• PyTorch integration ready")

def main():
    """Main menu loop"""
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                run_simple()
            elif choice == '2':
                run_interactive()
            elif choice == '3':
                run_advanced()
            elif choice == '4':
                show_info()
            elif choice == '5':
                print("👋 Thanks for using Forest Fire RL Environment!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure all dependencies are installed.")
        
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
