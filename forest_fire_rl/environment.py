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
        self.action_space = spaces.Discrete(5)  # up, down, left, right, wait
        # One-hot encoded observation: (4 channels, height, width) for [empty, forest, fire, firefighter]
        self.observation_space = spaces.Box(0, 1, (4, grid_size, grid_size), dtype=np.float32)
        
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
        # Reduce initial fire spread to make task more manageable
        self._spread_fire()
        self._spread_fire()
        self._spread_fire()
        self._spread_fire()

        # Place firefighter randomly, but not too close to the fire
        while True:
            self.firefighter_pos = self.np_random.integers(0, self.grid_size, size=2)
            # Ensure firefighter is at least 3 cells away from the center fire
            dist_to_fire = np.abs(self.firefighter_pos - center).max()
            if dist_to_fire >= 3:  # Manhattan distance of at least 3
                break
        
        self.step_count = 0
        
        # Reset fire count tracking for reward calculation
        self.prev_fire_count = np.sum(self.grid == 2)
        
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

        # action 4 = wait (do nothing)
        
        # Suppress fire first, then spread (gives firefighter advantage)
        fires_suppressed = self._suppress_fire()
        
        # Only spread fire every few steps to balance the game
        if self.step_count % 2 == 0:  # Spread fire every 2 steps instead of every step
            self._spread_fire()
        
        # Calculate reward
        fire_count = np.sum(self.grid == 2)
        forest_count = np.sum(self.grid == 1)
        
        # Store previous fire count for reward calculation
        if not hasattr(self, 'prev_fire_count'):
            self.prev_fire_count = fire_count
            
        # Reward based on change in fire count (encourage fire reduction)
        fire_change = self.prev_fire_count - fire_count
        reward = fire_change * 100.0  # reward for reducing fires
        
        # Small penalty for fires remaining (much smaller than before)
        reward -= fire_count * 0.1
        
        # Small reward for staying alive
        reward += 1.0
        
        # Big bonus for extinguishing all fires
        terminated = fire_count == 0
        if terminated:
            reward += 1000.0
            
        # Update previous fire count
        self.prev_fire_count = fire_count
        
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_obs()
        info = {'fire_count': fire_count, 'forest_count': forest_count}
        
        return obs, reward, terminated, truncated, info
    
    def _suppress_fire(self):
        """Suppress fires around firefighter position with larger radius"""
        x, y = self.firefighter_pos
        fires_suppressed = 0
        
        # Check 5x5 area around firefighter (increased from 3x3)
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[nx, ny] == 2:  # fire
                        self.grid[nx, ny] = 0  # extinguish to empty (burned area)
                        fires_suppressed += 1
        
        return fires_suppressed
    
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
        """Get one-hot encoded observation with firefighter marked"""
        # Create one-hot encoded observation (4 channels: empty, forest, fire, firefighter)
        obs = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Channel 0: empty/burned areas
        obs[0] = (self.grid == 0).astype(np.float32)
        
        # Channel 1: forest areas
        obs[1] = (self.grid == 1).astype(np.float32)
        
        # Channel 2: fire areas
        obs[2] = (self.grid == 2).astype(np.float32)
        
        # Channel 3: firefighter position
        obs[3, self.firefighter_pos[0], self.firefighter_pos[1]] = 1.0
        
        return obs
    
    def render(self, mode="human"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            plt.ion()
        
        self.ax.clear()
        
        # Color map: empty=brown, forest=green, fire=red, firefighter=blue
        colors = ['#8B4513', '#228B22', '#FF4500', '#0000FF']
        cmap = ListedColormap(colors)
        
        # Create visualization grid from current state
        vis_grid = self.grid.copy()
        vis_grid[self.firefighter_pos[0], self.firefighter_pos[1]] = 3  # mark firefighter
        
        im = self.ax.imshow(vis_grid, cmap=cmap, vmin=0, vmax=3)
        
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
