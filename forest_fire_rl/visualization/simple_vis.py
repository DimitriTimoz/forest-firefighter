"""
Simple visualizer for Forest Fire RL Environment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from typing import Dict, Any, Optional

from ..core.environment import ForestFireEnv
from ..agents.base import BaseAgent


class SimpleVisualizer:
    """Simple matplotlib-based visualizer for forest fire environment"""
    
    def __init__(self, figsize: tuple = (8, 8)):
        """
        Initialize the visualizer
        
        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
        
    def setup_plot(self, grid_size: int):
        """Setup the matplotlib plot"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            plt.ion()  # Interactive mode
        
        # Enhanced color scheme
        colors = ['#654321', '#228B22', '#FF4500', '#0064FF', '#808080']  # Brown, Green, Red, Blue, Gray
        self.cmap = ListedColormap(colors)
        
        self.ax.set_xlim(-0.5, grid_size - 0.5)
        self.ax.set_ylim(-0.5, grid_size - 0.5)
        self.ax.set_aspect('equal')
    
    def render_state(self, observation: Dict[str, Any], step: int, reward: float):
        """Render current environment state"""
        if self.ax is None:
            grid_size = observation['grid'].shape[0]
            self.setup_plot(grid_size)
        
        self.ax.clear()
        
        # Display grid
        grid = observation['grid']
        self.ax.imshow(grid, cmap=self.cmap, vmin=0, vmax=4, alpha=0.8)
        
        # Add firefighter position marker
        ff_pos = observation['firefighter_pos'] 
        self.ax.scatter(ff_pos[1], ff_pos[0], c='white', s=200, marker='*', 
                       edgecolors='black', linewidth=2)
        
        # Add title with stats
        fire_count = np.sum(grid == 2)
        forest_count = np.sum(grid == 1)
        burned_count = np.sum(grid == 0)
        
        title = f'Step {step} - Fire: {fire_count}, Forest: {forest_count}, Burned: {burned_count}\n'
        title += f'Reward: {reward:.2f}'
        self.ax.set_title(title, fontsize=12)
        
        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add grid lines
        grid_size = grid.shape[0]
        self.ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        self.ax.grid(which="minor", color="white", linestyle='-', linewidth=0.3, alpha=0.5)
        
        plt.draw()
        plt.pause(0.1)
    
    def run_episode(self, env: ForestFireEnv, agent: BaseAgent, 
                   max_steps: int = 100, delay: float = 0.5, 
                   verbose: bool = True):
        """
        Run and visualize a complete episode
        
        Args:
            env: Forest fire environment
            agent: Agent to run
            max_steps: Maximum steps per episode
            delay: Delay between steps (seconds)
            verbose: Print step information
        """
        observation, info = env.reset()
        total_reward = 0
        
        if verbose:
            print(f"🎬 Starting episode visualization...")
            print(f"📋 Grid size: {observation['grid'].shape[0]}x{observation['grid'].shape[1]}")
        
        try:
            for step in range(max_steps):
                # Render current state
                self.render_state(observation, step, total_reward)
                
                # Get action and step
                action = agent.act(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if verbose and step % 10 == 0:
                    print(f"Step {step:3d}: Action={action}, Reward={reward:6.2f}, "
                         f"Fire={info.get('fire_count', 0):3d}")
                
                # Add delay
                time.sleep(delay)
                
                if terminated or truncated:
                    # Render final state
                    self.render_state(observation, step + 1, total_reward)
                    if verbose:
                        print(f"🏁 Episode ended at step {step + 1}")
                    break
            
            if verbose:
                print(f"🏆 Final reward: {total_reward:.2f}")
                print(f"🔥 Final fire count: {info.get('fire_count', 0)}")
                print(f"🌲 Forest preservation: {info.get('forest_preservation_ratio', 0):.2%}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Visualization interrupted by user")
        
        return total_reward
    
    def close(self):
        """Close the visualization"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
