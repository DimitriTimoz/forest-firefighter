"""
Advanced visualizer for Forest Fire RL Environment with analytics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import time
from typing import Dict, Any, List, Optional
from collections import deque

from ..core.environment import ForestFireEnv
from ..agents.base import BaseAgent


class AdvancedVisualizer:
    """Advanced visualizer with analytics, plots, and recording capabilities"""
    
    def __init__(self, figsize: tuple = (15, 10), track_metrics: bool = True):
        """
        Initialize the advanced visualizer
        
        Args:
            figsize: Figure size for matplotlib plots
            track_metrics: Whether to track and display metrics over time
        """
        self.figsize = figsize
        self.track_metrics = track_metrics
        self.fig = None
        self.axes = {}
        
        # Metrics tracking
        if self.track_metrics:
            self.metrics_history = {
                'rewards': deque(maxlen=1000),
                'fire_counts': deque(maxlen=1000),
                'forest_counts': deque(maxlen=1000),
                'actions': deque(maxlen=1000),
                'steps': deque(maxlen=1000)
            }
        
        # Recording
        self.frames = []
        self.recording = False
    
    def setup_plot(self, grid_size: int):
        """Setup the advanced matplotlib layout"""
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
            
            # Create grid layout
            if self.track_metrics:
                # 2x3 layout: main plot (2x2), metrics (2x1)
                gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                
                # Main environment plot (larger)
                self.axes['main'] = self.fig.add_subplot(gs[:2, :2])
                
                # Metrics plots
                self.axes['rewards'] = self.fig.add_subplot(gs[0, 2])
                self.axes['counts'] = self.fig.add_subplot(gs[1, 2])  
                self.axes['actions'] = self.fig.add_subplot(gs[2, :])
                
            else:
                # Single main plot
                self.axes['main'] = self.fig.add_subplot(111)
            
            plt.ion()  # Interactive mode
        
        # Enhanced color scheme
        colors = ['#654321', '#228B22', '#FF4500', '#0064FF', '#808080']
        self.cmap = ListedColormap(colors)
        
        # Setup main plot
        main_ax = self.axes['main']
        main_ax.set_xlim(-0.5, grid_size - 0.5)
        main_ax.set_ylim(-0.5, grid_size - 0.5)
        main_ax.set_aspect('equal')
    
    def render_state(self, observation: Dict[str, Any], step: int, reward: float,
                    action: Optional[int] = None, additional_info: Dict[str, Any] = None):
        """Render current state with advanced visualization"""
        if 'main' not in self.axes:
            grid_size = observation['grid'].shape[0]
            self.setup_plot(grid_size)
        
        # Update metrics tracking
        if self.track_metrics:
            self.metrics_history['rewards'].append(reward)
            self.metrics_history['fire_counts'].append(np.sum(observation['grid'] == 2))
            self.metrics_history['forest_counts'].append(np.sum(observation['grid'] == 1))
            if action is not None:
                self.metrics_history['actions'].append(action)
            self.metrics_history['steps'].append(step)
        
        # Clear axes
        for ax in self.axes.values():
            ax.clear()
        
        # Main environment plot
        self._render_main_plot(observation, step, reward, action, additional_info)
        
        # Metrics plots
        if self.track_metrics and len(self.metrics_history['steps']) > 1:
            self._render_metrics_plots()
        
        plt.draw()
        
        # Record frame if recording
        if self.recording:
            self.frames.append(self._capture_frame())
    
    def _render_main_plot(self, observation: Dict[str, Any], step: int, reward: float,
                         action: Optional[int] = None, additional_info: Dict[str, Any] = None):
        """Render the main environment visualization"""
        ax = self.axes['main']
        
        # Display grid with enhanced visualization
        grid = observation['grid']
        im = ax.imshow(grid, cmap=self.cmap, vmin=0, vmax=4, alpha=0.9)
        
        # Add firefighter with trail effect
        firefighter_positions = observation.get('firefighter_positions', 
                                                observation.get('firefighter_pos', np.array([[0, 0]])))
        
        # Handle both single and multiple firefighter formats
        if len(firefighter_positions.shape) == 1:
            firefighter_positions = firefighter_positions.reshape(1, -1)
        
        for i, ff_pos in enumerate(firefighter_positions):
            # Use different markers for multiple firefighters
            marker = '*' if i == 0 else ['o', 's', '^', 'v', 'D'][i % 5] if i < 5 else 'X'
            size = 400 if i == 0 else 300
            
            ax.scatter(ff_pos[1], ff_pos[0], c='white', s=size, marker=marker, 
                      edgecolors='black', linewidth=3, alpha=1.0, zorder=10)
        
        # Add action indicators
        action_names = ['↑', '↓', '←', '→', '💧', '⏸️']
        if action is not None and action < len(action_names):
            ax.text(ff_pos[1] + 0.3, ff_pos[0] + 0.3, action_names[action], 
                   fontsize=16, fontweight='bold', color='yellow',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))
        
        # Enhanced title with comprehensive stats
        fire_count = np.sum(grid == 2)
        forest_count = np.sum(grid == 1)
        burned_count = np.sum(grid == 0)
        suppressed_count = np.sum(grid == 4)
        
        title = f'Advanced Forest Fire Simulation - Step {step}\n'
        title += f'🔥 Fire: {fire_count:3d} | 🌲 Forest: {forest_count:3d} | 🔥 Burned: {burned_count:3d} | 💧 Suppressed: {suppressed_count:3d}\n'
        title += f'💰 Reward: {reward:+6.2f}'
        
        if additional_info:
            if 'total_reward' in additional_info:
                title += f' | 🏆 Total: {additional_info["total_reward"]:+6.2f}'
            if 'forest_preservation' in additional_info:
                title += f' | 🌲 Preserved: {additional_info["forest_preservation"]:.1%}'
        
        ax.set_title(title, fontsize=11, pad=15)
        
        # Style improvements
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Enhanced grid
        grid_size = grid.shape[0]
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.2, alpha=0.3)
        
        # Add colorbar if not exists
        if not hasattr(self, '_colorbar_added'):
            cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(['Burned', 'Forest', 'Fire', 'Firefighter', 'Suppressed'])
            cbar.ax.tick_params(labelsize=9)
            self._colorbar_added = True
    
    def _render_metrics_plots(self):
        """Render metrics tracking plots"""
        steps = list(self.metrics_history['steps'])
        
        # Rewards plot
        if 'rewards' in self.axes:
            ax = self.axes['rewards']
            rewards = list(self.metrics_history['rewards'])
            ax.plot(steps, rewards, 'b-', linewidth=2, alpha=0.7)
            ax.fill_between(steps, rewards, alpha=0.3)
            ax.set_title('Rewards', fontsize=10)
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
        
        # Fire/Forest counts
        if 'counts' in self.axes:
            ax = self.axes['counts']
            fire_counts = list(self.metrics_history['fire_counts'])
            forest_counts = list(self.metrics_history['forest_counts'])
            
            ax.plot(steps, fire_counts, 'r-', label='Fire', linewidth=2)
            ax.plot(steps, forest_counts, 'g-', label='Forest', linewidth=2)
            ax.set_title('Fire vs Forest', fontsize=10)
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Action distribution
        if 'actions' in self.axes and len(self.metrics_history['actions']) > 0:
            ax = self.axes['actions']
            actions = list(self.metrics_history['actions'])
            action_names = ['Up', 'Down', 'Left', 'Right', 'Suppress', 'Nothing']
            
            # Recent action distribution (last 50 actions)
            recent_actions = actions[-50:]
            action_counts = [recent_actions.count(i) for i in range(6)]
            
            bars = ax.bar(range(6), action_counts, color=['skyblue', 'lightgreen', 'orange', 'pink', 'red', 'gray'])
            ax.set_title('Action Distribution (Last 50)', fontsize=10)
            ax.set_xticks(range(6))
            ax.set_xticklabels([name[:4] for name in action_names], rotation=45, fontsize=8)
            ax.set_ylabel('Count')
    
    def _capture_frame(self):
        """Capture current frame for recording"""
        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return frame
    
    def start_recording(self):
        """Start recording frames"""
        self.recording = True
        self.frames = []
    
    def stop_recording(self):
        """Stop recording frames"""
        self.recording = False
        return self.frames.copy()
    
    def save_recording(self, filename: str, fps: int = 5):
        """Save recorded frames as video"""
        if not self.frames:
            print("No frames recorded")
            return
        
        try:
            import cv2
            
            height, width, layers = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)
            
            video.release()
            print(f"Saved recording to {filename}")
            
        except ImportError:
            print("OpenCV not available for video saving")
    
    def run_episode(self, env: ForestFireEnv, agent: BaseAgent, 
                   max_steps: int = 200, delay: float = 0.3,
                   record: bool = False, verbose: bool = True):
        """
        Run and visualize an advanced episode
        
        Args:
            env: Forest fire environment
            agent: Agent to run  
            max_steps: Maximum steps per episode
            delay: Delay between steps (seconds)
            record: Whether to record the episode
            verbose: Print step information
        """
        if record:
            self.start_recording()
        
        observation, info = env.reset()
        total_reward = 0
        
        if verbose:
            print(f"🎥 Starting advanced episode visualization...")
            print(f"📊 Tracking metrics: {self.track_metrics}")
            print(f"🎬 Recording: {record}")
        
        try:
            for step in range(max_steps):
                # Get action
                action = agent.act(observation)
                
                # Additional info for display
                additional_info = {
                    'total_reward': total_reward,
                    'forest_preservation': info.get('forest_preservation_ratio', 0)
                }
                
                # Render current state
                self.render_state(observation, step, total_reward, action, additional_info)
                
                # Take environment step
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if verbose and step % 25 == 0:
                    action_name = ['Up', 'Down', 'Left', 'Right', 'Suppress', 'Nothing'][action]
                    print(f"Step {step:3d}: {action_name:>8} | Reward={reward:+6.2f} | "
                         f"Total={total_reward:+6.2f} | Fire={info.get('fire_count', 0):3d}")
                
                # Delay
                time.sleep(delay)
                
                if terminated or truncated:
                    # Final render
                    self.render_state(observation, step + 1, total_reward, None, additional_info)
                    if verbose:
                        print(f"🏁 Episode ended at step {step + 1}")
                    break
            
            if record:
                frames = self.stop_recording()
                if verbose:
                    print(f"📹 Recorded {len(frames)} frames")
            
            if verbose:
                print(f"🏆 Final reward: {total_reward:.2f}")
                print(f"📊 Final statistics:")
                print(f"  🔥 Fire count: {info.get('fire_count', 0)}")
                print(f"  🌲 Forest count: {info.get('forest_count', 0)}")
                print(f"  🌲 Forest preservation: {info.get('forest_preservation_ratio', 0):.2%}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Advanced visualization interrupted by user")
            if record:
                self.stop_recording()
        
        return total_reward
    
    def close(self):
        """Close the advanced visualizer"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = {}
            if hasattr(self, '_colorbar_added'):
                delattr(self, '_colorbar_added')
