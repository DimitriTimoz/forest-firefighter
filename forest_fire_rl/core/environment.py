"""Forest Fire Environment - High-performance Gymnasium environment for forest fire simulation"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm


class ForestFireEnv(gym.Env):
    """Forest Fire Environment with multi-firefighter support and efficient fire dynamics"""
    
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}
    
    # State and action constants
    EMPTY, FOREST, FIRE, FIREFIGHTER, SUPPRESSED = 0, 1, 2, 3, 4
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, SUPPRESS, WAIT = 0, 1, 2, 3, 4, 5
    
    def __init__(self, grid_size: int = 20, initial_fire_positions: Optional[list] = None,
                 fire_spread_prob: float = 0.1, wind_direction: Optional[Tuple[int, int]] = None,
                 wind_strength: float = 0.0, max_steps: int = 200, render_mode: Optional[str] = None,
                 reward_structure: Optional[Dict[str, float]] = None, num_firefighters: int = 1):
        super().__init__()
        
        self.grid_size = grid_size
        self.initial_fire_positions = initial_fire_positions or [(grid_size//2, grid_size//2)]
        self.fire_spread_prob = fire_spread_prob
        self.wind_direction = wind_direction or (0, 0)
        self.wind_strength = wind_strength
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.num_firefighters = max(1, num_firefighters)
        
        # Default rewards
        default_rewards = {'fire_suppressed': 10.0, 'forest_saved': 1.0, 'forest_burned': -2.0,
                          'invalid_action': -0.5, 'time_penalty': -0.1, 'fire_contained': 50.0}
        self.rewards = reward_structure or default_rewards
        
        # Action and observation spaces
        if self.num_firefighters == 1:
            self.action_space = spaces.Discrete(6)
        else:
            self.action_space = spaces.MultiDiscrete([6] * self.num_firefighters)
        
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=4, shape=(grid_size, grid_size), dtype=np.int32),
            'firefighter_positions': spaces.Box(low=0, high=grid_size-1, 
                                              shape=(self.num_firefighters, 2), dtype=np.int32),
            'steps_remaining': spaces.Box(low=0, high=max_steps, shape=(), dtype=np.int32)
        })
        
        # Internal state
        self.grid = None
        self.firefighter_positions = None
        self.step_count = 0
        self.initial_forest_count = 0
        self.original_cells = None  # Track original states under firefighters
        self._fire_spread_kernel = self._create_fire_spread_kernel()
        self.fig = None
        self.ax = None
        
    def _create_fire_spread_kernel(self) -> np.ndarray:
        """Create convolution kernel for efficient fire spread computation"""
        kernel = np.array([
            [0, 1, 0],
            [1, 0, 1], 
            [0, 1, 0]
        ], dtype=np.float32)
        
        # Apply wind influence
        if self.wind_strength > 0:
            wind_x, wind_y = self.wind_direction
            # Increase spread probability in wind direction
            if wind_x > 0:  # Wind blowing right
                kernel[1, 2] *= (1 + self.wind_strength)
            elif wind_x < 0:  # Wind blowing left
                kernel[1, 0] *= (1 + self.wind_strength)
            
            if wind_y > 0:  # Wind blowing down
                kernel[2, 1] *= (1 + self.wind_strength)
            elif wind_y < 0:  # Wind blowing up
                kernel[0, 1] *= (1 + self.wind_strength)
        
        return kernel
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # Initialize grid with forest
        self.grid = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Set initial fires
        for fire_pos in self.initial_fire_positions:
            if 0 <= fire_pos[0] < self.grid_size and 0 <= fire_pos[1] < self.grid_size:
                self.grid[fire_pos[0], fire_pos[1]] = 2
        
        # Place firefighters randomly (not on fire)
        self.firefighter_positions = np.zeros((self.num_firefighters, 2), dtype=np.int32)
        self.original_cells = np.zeros((self.num_firefighters,), dtype=np.int32)  # Track original cell states
        
        valid_positions = np.where(self.grid == 1)
        if len(valid_positions[0]) >= self.num_firefighters:
            # Sample unique positions for all firefighters
            indices = self.np_random.choice(len(valid_positions[0]), size=self.num_firefighters, replace=False)
            for i in range(self.num_firefighters):
                idx = indices[i]
                pos = np.array([valid_positions[0][idx], valid_positions[1][idx]], dtype=np.int32)
                self.firefighter_positions[i] = pos
                self.original_cells[i] = self.grid[pos[0], pos[1]]  # Store original cell (forest)
                self.grid[pos[0], pos[1]] = self.FIREFIGHTER
        else:
            # Fallback: place firefighters even if positions overlap
            for i in range(self.num_firefighters):
                if len(valid_positions[0]) > 0:
                    idx = self.np_random.integers(len(valid_positions[0]))
                    pos = np.array([valid_positions[0][idx], valid_positions[1][idx]], dtype=np.int32)
                else:
                    pos = np.array([i, 0], dtype=np.int32)  # Emergency fallback
                
                self.firefighter_positions[i] = pos
                self.original_cells[i] = self.grid[pos[0], pos[1]] if len(valid_positions[0]) > 0 else self.FOREST
                self.grid[pos[0], pos[1]] = self.FIREFIGHTER
        
        self.step_count = 0
        self.initial_forest_count = np.sum(self.grid == 1) + np.sum(self.grid == 3)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        self.step_count += 1
        reward = self.rewards['time_penalty']  # Base time penalty
        
        # Store current state for reward calculation
        prev_fire_count = np.sum(self.grid == 2)
        prev_forest_count = np.sum(self.grid == 1)
        
        # Handle both single and multi-firefighter actions
        if self.num_firefighters == 1:
            # Single firefighter: action is an integer
            actions = [action] if isinstance(action, int) else action
        else:
            # Multiple firefighters: action should be array/list
            actions = action if hasattr(action, '__len__') else [action] * self.num_firefighters
            
        # Ensure we have the right number of actions
        while len(actions) < self.num_firefighters:
            actions.append(self.WAIT)  # Default to wait if not enough actions
        
        # Execute actions for all firefighters
        for i, ff_action in enumerate(actions[:self.num_firefighters]):
            valid_action = self._execute_action(ff_action, firefighter_id=i)
            if not valid_action:
                reward += self.rewards['invalid_action'] / self.num_firefighters  # Distribute penalty
        
        # Spread fire (efficient vectorized operation)
        self._spread_fire()
        
        # Calculate rewards
        current_fire_count = np.sum(self.grid == 2)
        current_forest_count = np.sum(self.grid == 1)
        
        # Reward for suppressing fires
        fires_suppressed = prev_fire_count - current_fire_count
        if fires_suppressed > 0:
            reward += fires_suppressed * self.rewards['fire_suppressed']
        
        # Penalty for forest burning
        forest_burned = prev_forest_count - current_forest_count
        if forest_burned > 0:
            reward += forest_burned * self.rewards['forest_burned']
        
        # Check termination conditions
        terminated = False
        
        # Fire contained (no more fires)
        if current_fire_count == 0:
            reward += self.rewards['fire_contained']
            terminated = True
        
        # All forest burned or max steps reached
        if current_forest_count == 0 or self.step_count >= self.max_steps:
            terminated = True
        
        # Final reward based on forest preservation
        if terminated:
            forest_preservation_ratio = current_forest_count / max(self.initial_forest_count, 1)
            reward += forest_preservation_ratio * self.rewards['forest_saved'] * 10
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _execute_action(self, action: int, firefighter_id: int = 0) -> bool:
        """Execute action for a specific firefighter and return whether it was valid"""
        if firefighter_id >= self.num_firefighters:
            return False
            
        old_pos = self.firefighter_positions[firefighter_id].copy()
        
        if action == self.MOVE_UP:  # Move up
            new_pos = old_pos + np.array([-1, 0])
        elif action == self.MOVE_DOWN:  # Move down  
            new_pos = old_pos + np.array([1, 0])
        elif action == self.MOVE_LEFT:  # Move left
            new_pos = old_pos + np.array([0, -1])
        elif action == self.MOVE_RIGHT:  # Move right
            new_pos = old_pos + np.array([0, 1])
        elif action == self.SUPPRESS:  # Suppress fire
            return self._suppress_fire(firefighter_id)
        elif action == self.WAIT:  # Do nothing
            return True
        else:
            return False
        
        # Check bounds for movement actions
        if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            # Check if another firefighter is already at the target position
            for other_id in range(self.num_firefighters):
                if other_id != firefighter_id and np.array_equal(self.firefighter_positions[other_id], new_pos):
                    return False  # Can't move to occupied position
            
            # Restore old position to its original state
            self.grid[old_pos[0], old_pos[1]] = self.original_cells[firefighter_id]
            
            # Move firefighter to new position
            self.firefighter_positions[firefighter_id] = new_pos
            
            # Store the original state of the new cell before placing firefighter
            original_new_cell = self.grid[new_pos[0], new_pos[1]]
            
            # If moving to a fire, suppress it (becomes burned/empty)
            if original_new_cell == self.FIRE:
                self.original_cells[firefighter_id] = self.EMPTY  # Fire becomes burned
            else:
                self.original_cells[firefighter_id] = original_new_cell  # Keep original state
            
            # Place firefighter at new position
            self.grid[new_pos[0], new_pos[1]] = self.FIREFIGHTER
            return True
        
        return False
    
    def _suppress_fire(self, firefighter_id: int = 0) -> bool:
        """Suppress fire around specified firefighter position"""
        if firefighter_id >= self.num_firefighters:
            return False
            
        suppressed = False
        firefighter_pos = self.firefighter_positions[firefighter_id]
        
        # Suppress fire in 3x3 area around firefighter
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = firefighter_pos[0] + dx, firefighter_pos[1] + dy
                if (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                    if self.grid[x, y] == self.FIRE:  # On fire
                        self.grid[x, y] = self.SUPPRESSED  # Suppress to suppressed state
                        suppressed = True
        
        return suppressed
    
    def _spread_fire(self):
        """Efficiently spread fire using vectorized operations"""
        # Get current fire positions
        fire_mask = (self.grid == 2)
        
        if not np.any(fire_mask):
            return
        
        # Create padded grid for boundary handling
        padded_grid = np.pad(self.grid, 1, mode='constant', constant_values=0)
        padded_fire_mask = np.pad(fire_mask, 1, mode='constant', constant_values=False)
        
        # Compute fire influence using convolution-like operation
        fire_influence = np.zeros_like(padded_grid, dtype=np.float32)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                # Get kernel weight
                kernel_weight = self._fire_spread_kernel[dx+1, dy+1]
                
                # Shift fire mask and add influence
                shifted_fire = np.roll(np.roll(padded_fire_mask, dx, axis=0), dy, axis=1)
                fire_influence += shifted_fire * kernel_weight
        
        # Apply fire spread probability
        spread_prob = self.np_random.random(padded_grid.shape)
        new_fires = (spread_prob < fire_influence * self.fire_spread_prob)
        
        # Only spread to healthy forest (value 1)
        forest_mask = (padded_grid == 1)
        new_fires = new_fires & forest_mask
        
        # Update grid (remove padding)
        new_fires_unpadded = new_fires[1:-1, 1:-1]
        self.grid[new_fires_unpadded] = 2
    
    def _get_observation(self) -> Dict:
        return {
            'grid': self.grid.copy(),
            'firefighter_positions': self.firefighter_positions.copy(),
            'steps_remaining': np.array(self.max_steps - self.step_count, dtype=np.int32)
        }
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            'fire_count': int(np.sum(self.grid == self.FIRE)),
            'forest_count': int(np.sum(self.grid == self.FOREST)),
            'burned_count': int(np.sum(self.grid == self.EMPTY)),
            'suppressed_count': int(np.sum(self.grid == self.SUPPRESSED)),
            'firefighter_count': self.num_firefighters,
            'step_count': self.step_count,
            'forest_preservation_ratio': float(np.sum(self.grid == self.FOREST) / max(self.initial_forest_count, 1))
        }
    
    def render(self):
        """Render the environment state"""
        if self.render_mode is None:
            return None
        
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            return self._render_human()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array"""
        # Create RGB image
        rgb_array = np.zeros((*self.grid.shape, 3), dtype=np.uint8)
        
        # Enhanced color scheme
        rgb_array[self.grid == self.EMPTY] = [101, 67, 33]       # Dark brown (burned)
        rgb_array[self.grid == self.FOREST] = [34, 139, 34]     # Forest green
        rgb_array[self.grid == self.FIRE] = [255, 69, 0]        # Red-orange (fire)
        rgb_array[self.grid == self.FIREFIGHTER] = [0, 100, 255] # Blue (firefighter)
        rgb_array[self.grid == self.SUPPRESSED] = [128, 128, 128] # Gray (suppressed)
        
        return rgb_array
    
    def _render_human(self):
        """Render for human viewing with matplotlib"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            plt.ion()  # Interactive mode
        
        self.ax.clear()
        
        # Create display grid (separate from game logic)
        display_grid = self.grid.copy()
        
        # Enhanced color mapping with better colors
        colors = [
            '#654321',  # Dark brown for empty/burned
            '#228B22',  # Forest green
            '#FF4500',  # Red-orange for fire
            '#0064FF',  # Blue for firefighter
            '#808080'   # Gray for suppressed
        ]
        
        cmap = ListedColormap(colors)
        bounds = [0, 1, 2, 3, 4, 5]
        norm = BoundaryNorm(bounds, cmap.N)
        
        # Show the grid
        im = self.ax.imshow(display_grid, cmap=cmap, norm=norm, alpha=0.9)
        
        # Add wind indicators if wind is significant
        if hasattr(self, 'wind_strength') and self.wind_strength > 0.1:
            self._add_wind_indicators()
        
        # Add firefighter position markers
        for i, ff_pos in enumerate(self.firefighter_positions):
            ff_y, ff_x = ff_pos
            # Use different markers for multiple firefighters  
            marker = '*' if i == 0 else ['o', 's', '^', 'v', 'D'][i % 5] if i < 5 else 'X'
            size = 200 if i == 0 else 150
            
            self.ax.scatter(ff_x, ff_y, c='white', s=size, marker=marker, 
                           edgecolors='black', linewidth=2, alpha=0.9,
                           label=f'FF{i+1}' if self.num_firefighters > 1 else None)
        
        # Add legend for multiple firefighters
        if self.num_firefighters > 1:
            self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        # Enhanced title with statistics
        fire_count = np.sum(self.grid == self.FIRE)
        forest_count = np.sum(self.grid == self.FOREST)
        burned_count = np.sum(self.grid == self.EMPTY)
        suppressed_count = np.sum(self.grid == self.SUPPRESSED)
        
        title = f'Forest Fire Simulation - Step {self.step_count}\n'
        title += f'Fire: {fire_count} | Forest: {forest_count} | Burned: {burned_count}'
        if suppressed_count > 0:
            title += f' | Suppressed: {suppressed_count}'
        if self.num_firefighters > 1:
            title += f' | Firefighters: {self.num_firefighters}'
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Remove ticks but add subtle grid
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add grid lines
        self.ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.grid(which="minor", color="white", linestyle='-', linewidth=0.3, alpha=0.5)
        
        # Add colorbar legend
        if not hasattr(self, '_colorbar_added'):
            cbar = plt.colorbar(im, ax=self.ax, shrink=0.8, pad=0.02)
            cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(['Burned', 'Forest', 'Fire', 'Firefighter', 'Suppressed'])
            cbar.ax.tick_params(labelsize=10)
            self._colorbar_added = True
        
        plt.tight_layout()
        
        if self.render_mode == "human":
            plt.pause(0.1)
            plt.draw()
        
        return self.fig
    
    def _add_wind_indicators(self):
        """Add wind direction indicators to the visualization"""
        if not hasattr(self, 'wind_direction'):
            return
        
        # Sample points for wind indicators (every 4th cell)
        step = max(2, self.grid_size // 8)
        for i in range(step, self.grid_size - step, step):
            for j in range(step, self.grid_size - step, step):
                # Calculate wind arrow
                dx = np.cos(self.wind_direction) * self.wind_strength * 0.5
                dy = np.sin(self.wind_direction) * self.wind_strength * 0.5
                
                self.ax.arrow(j, i, dx, dy, 
                            head_width=0.2, head_length=0.2, 
                            fc='yellow', ec='orange', alpha=0.7, linewidth=1)
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Utility functions for creating different scenarios
def create_wildfire_scenario(grid_size: int = 20, num_fires: int = 3) -> list:
    """Create random fire positions for wildfire scenario"""
    np.random.seed(42)  # For reproducibility
    positions = []
    for _ in range(num_fires):
        x = np.random.randint(0, grid_size)
        y = np.random.randint(0, grid_size)
        positions.append((x, y))
    return positions

def create_edge_fire_scenario(grid_size: int = 20) -> list:
    """Create fire starting from one edge"""
    return [(0, grid_size//2), (0, grid_size//2-1), (0, grid_size//2+1)]

def create_central_fire_scenario(grid_size: int = 20) -> list:
    """Create fire starting from center"""
    center = grid_size // 2
    return [(center, center)]
