"""Minimalist Forest Fire Gymnasium Environment"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import convolve


class ForestFireEnv(gym.Env):
    """Minimalist forest fire environment"""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, grid_size=15, fire_spread_prob=0.1, max_steps=100):
        super().__init__()
        
        self.grid_size = grid_size
        self.fire_spread_prob = fire_spread_prob
        self.max_steps = max_steps
        
        # Spaces
        self.action_space = spaces.Discrete(6)  # up, down, left, right, suppress, wait
        self.observation_space = spaces.Box(0, 3, (grid_size, grid_size), dtype=np.int32)
        
        # State
        self.grid = None
        self.firefighter_pos = None
        self.step_count = 0
        
        # Fire spreading kernel
        self.fire_kernel = np.array([
            [0.1, 0.3, 0.1],
            [0.3, 0.0, 0.3], 
            [0.1, 0.3, 0.1]
        ])
        
        # Rendering
        self.fig, self.ax = None, None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize grid: 0=empty, 1=forest, 2=fire, 3=firefighter
        self.grid = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Add initial fire in center
        center = self.grid_size // 2
        self.grid[center, center] = 2
        
        # Place firefighter randomly
        self.firefighter_pos = self.np_random.integers(0, self.grid_size, size=2)
        self.step_count = 0
        
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        reward = 0.0
        
        # Move firefighter
        old_pos = self.firefighter_pos.copy()
        if action == 0 and self.firefighter_pos[0] > 0:  # up
            self.firefighter_pos[0] -= 1
        elif action == 1 and self.firefighter_pos[0] < self.grid_size - 1:  # down
            self.firefighter_pos[0] += 1
        elif action == 2 and self.firefighter_pos[1] > 0:  # left
            self.firefighter_pos[1] -= 1
        elif action == 3 and self.firefighter_pos[1] < self.grid_size - 1:  # right
            self.firefighter_pos[1] += 1
        elif action == 4:  # suppress fire
            reward += self._suppress_fire()
        # action 5 = wait (do nothing)
        
        # Spread fire
        self._spread_fire()
        
        # Calculate reward
        fire_count = np.sum(self.grid == 2)
        forest_count = np.sum(self.grid == 1)
        
        reward += forest_count * 0.1  # reward for preserving forest
        reward -= fire_count * 0.2    # penalty for active fires
        reward -= 0.01                # time penalty
        
        # Check termination
        terminated = fire_count == 0  # all fires out
        truncated = self.step_count >= self.max_steps
        
        if terminated:
            reward += 50.0  # bonus for success
        
        obs = self._get_obs()
        info = {'fire_count': fire_count, 'forest_count': forest_count}
        
        return obs, reward, terminated, truncated, info
    
    def _suppress_fire(self):
        """Suppress fires around firefighter position"""
        reward = 0.0
        x, y = self.firefighter_pos
        
        # Check 3x3 area around firefighter
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[nx, ny] == 2:  # fire
                        self.grid[nx, ny] = 0  # extinguish to empty
                        reward += 10.0
        
        return reward
    
    def _spread_fire(self):
        """Kernel-based fire spreading using convolution"""
        # Create fire mask for spreading calculation
        fire_mask = (self.grid == 2).astype(np.float32)
        
        # Convolve with fire spreading kernel
        spread_potential = convolve(fire_mask, self.fire_kernel, mode='constant', cval=0.0)
        
        # Apply spreading probability to forest cells
        forest_mask = (self.grid == 1)
        spread_prob = spread_potential * self.fire_spread_prob
        
        # Random spreading based on kernel-weighted probabilities
        random_vals = self.np_random.random(self.grid.shape)
        new_fires = forest_mask & (random_vals < spread_prob)
        
        # Update grid with new fires
        self.grid[new_fires] = 2
    
    def _get_obs(self):
        """Get observation with firefighter marked"""
        obs = self.grid.copy()
        obs[self.firefighter_pos[0], self.firefighter_pos[1]] = 3  # mark firefighter
        return obs
    
    def render(self, mode="human"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            plt.ion()
        
        self.ax.clear()
        
        # Color map: empty=brown, forest=green, fire=red, firefighter=blue
        colors = ['#8B4513', '#228B22', '#FF4500', '#0000FF']
        cmap = ListedColormap(colors)
        
        obs = self._get_obs()
        im = self.ax.imshow(obs, cmap=cmap, vmin=0, vmax=3)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#8B4513', label='Empty/Burned'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#228B22', label='Forest'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#FF4500', label='Fire'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#0000FF', label='Firefighter')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        fire_count = np.sum(self.grid == 2)
        forest_count = np.sum(self.grid == 1)
        self.ax.set_title(f'Forest Fire Simulation\nStep {self.step_count} | Fire: {fire_count} | Forest: {forest_count}')
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None
