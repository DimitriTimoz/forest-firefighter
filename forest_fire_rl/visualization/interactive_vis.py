"""
Interactive visualizer for Forest Fire RL Environment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button
import time
from typing import Dict, Any

from ..core.environment import ForestFireEnv
from ..agents.base import BaseAgent


class InteractiveVisualizer:
    """Interactive matplotlib-based visualizer with control buttons"""
    
    def __init__(self, figsize: tuple = (10, 8)):
        """
        Initialize the interactive visualizer
        
        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.button_ax = None
        self.buttons = {}
        self.is_paused = False
        self.step_requested = False
        
    def setup_plot(self, grid_size: int):
        """Setup the matplotlib plot with controls"""
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
            
            # Main plot
            self.ax = plt.subplot(111)
            
            # Control buttons
            self._setup_controls()
            
            plt.ion()  # Interactive mode
        
        # Enhanced color scheme
        colors = ['#654321', '#228B22', '#FF4500', '#0064FF', '#808080']
        self.cmap = ListedColormap(colors)
        
        self.ax.set_xlim(-0.5, grid_size - 0.5)
        self.ax.set_ylim(-0.5, grid_size - 0.5)
        self.ax.set_aspect('equal')
    
    def _setup_controls(self):
        """Setup interactive control buttons"""
        # Pause/Resume button
        pause_ax = plt.axes([0.02, 0.95, 0.08, 0.04])
        self.buttons['pause'] = Button(pause_ax, 'Pause')
        self.buttons['pause'].on_clicked(self._toggle_pause)
        
        # Step button
        step_ax = plt.axes([0.12, 0.95, 0.08, 0.04])  
        self.buttons['step'] = Button(step_ax, 'Step')
        self.buttons['step'].on_clicked(self._request_step)
        
        # Reset button
        reset_ax = plt.axes([0.22, 0.95, 0.08, 0.04])
        self.buttons['reset'] = Button(reset_ax, 'Reset')
        self.buttons['reset'].on_clicked(self._request_reset)
    
    def _toggle_pause(self, event):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        label = 'Resume' if self.is_paused else 'Pause'
        self.buttons['pause'].label.set_text(label)
        plt.draw()
    
    def _request_step(self, event):
        """Request single step"""
        if self.is_paused:
            self.step_requested = True
    
    def _request_reset(self, event):
        """Request episode reset"""
        # This would need to be handled by the calling code
        pass
    
    def render_state(self, observation: Dict[str, Any], step: int, reward: float, 
                    additional_info: Dict[str, Any] = None):
        """Render current environment state with enhanced info"""
        if self.ax is None:
            grid_size = observation['grid'].shape[0]
            self.setup_plot(grid_size)
        
        self.ax.clear()
        
        # Display grid
        grid = observation['grid']
        im = self.ax.imshow(grid, cmap=self.cmap, vmin=0, vmax=4, alpha=0.9)
        
        # Add firefighter position marker
        ff_pos = observation['firefighter_pos']
        self.ax.scatter(ff_pos[1], ff_pos[0], c='white', s=300, marker='*', 
                       edgecolors='black', linewidth=3, alpha=0.9)
        
        # Enhanced statistics
        fire_count = np.sum(grid == 2)
        forest_count = np.sum(grid == 1)
        burned_count = np.sum(grid == 0)
        
        # Main title
        title = f'Interactive Forest Fire Simulation - Step {step}\n'
        title += f'🔥 Fire: {fire_count} | 🌲 Forest: {forest_count} | 🔥 Burned: {burned_count}\n'
        title += f'💰 Reward: {reward:.2f}'
        
        if additional_info:
            if 'action_name' in additional_info:
                title += f' | 🎯 Action: {additional_info["action_name"]}'
        
        self.ax.set_title(title, fontsize=11, pad=20)
        
        # Style improvements
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Grid lines
        grid_size = grid.shape[0]
        self.ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        self.ax.grid(which="minor", color="white", linestyle='-', linewidth=0.3, alpha=0.5)
        
        # Add legend
        if not hasattr(self, '_legend_added'):
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='#654321', label='Burned'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#228B22', label='Forest'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#FF4500', label='Fire'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#0064FF', label='Firefighter'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#808080', label='Suppressed')
            ]
            self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            self._legend_added = True
        
        plt.tight_layout()
        plt.draw()
    
    def run_episode(self, env: ForestFireEnv, agent: BaseAgent, 
                   max_steps: int = 200, auto_delay: float = 0.5,
                   verbose: bool = True):
        """
        Run and visualize an interactive episode
        
        Args:
            env: Forest fire environment
            agent: Agent to run
            max_steps: Maximum steps per episode
            auto_delay: Delay when not paused (seconds)
            verbose: Print step information
        """
        observation, info = env.reset()
        total_reward = 0
        
        # Action names for display
        action_names = ['Up', 'Down', 'Left', 'Right', 'Suppress', 'Nothing']
        
        if verbose:
            print(f"🎮 Starting interactive episode...")
            print(f"📋 Grid size: {observation['grid'].shape[0]}x{observation['grid'].shape[1]}")
            print(f"🎛️  Controls: Pause/Resume, Step (when paused), Reset")
        
        try:
            for step in range(max_steps):
                # Get action
                action = agent.act(observation)
                action_name = action_names[action] if action < len(action_names) else f"Action {action}"
                
                # Render current state
                additional_info = {'action_name': action_name}
                self.render_state(observation, step, total_reward, additional_info)
                
                # Handle pause/step logic
                while self.is_paused and not self.step_requested:
                    plt.pause(0.1)  # Small delay while paused
                
                if self.step_requested:
                    self.step_requested = False
                
                # Take environment step
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if verbose and step % 20 == 0:
                    print(f"Step {step:3d}: {action_name:>8} | Reward={reward:6.2f} | "
                         f"Fire={info.get('fire_count', 0):3d}")
                
                # Auto delay when not paused
                if not self.is_paused:
                    time.sleep(auto_delay)
                
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
            print("\n⏹️  Interactive visualization interrupted by user")
        
        return total_reward
    
    def close(self):
        """Close the interactive visualizer"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.buttons = {}
