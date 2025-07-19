"""
Configuration module for Forest Fire Environment
Provides different environment configurations for various training scenarios
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np


class EnvConfig:
    """Base configuration class for environment settings"""
    
    def __init__(self):
        self.grid_size = 20
        self.initial_fire_positions = None
        self.fire_spread_prob = 0.1
        self.wind_direction = (0, 0)
        self.wind_strength = 0.0
        self.max_steps = 200
        self.reward_structure = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for environment initialization"""
        return {
            'grid_size': self.grid_size,
            'initial_fire_positions': self.initial_fire_positions,
            'fire_spread_prob': self.fire_spread_prob,
            'wind_direction': self.wind_direction,
            'wind_strength': self.wind_strength,
            'max_steps': self.max_steps,
            'reward_structure': self.reward_structure
        }


class EasyConfig(EnvConfig):
    """Easy configuration - small fires, slow spread"""
    
    def __init__(self):
        super().__init__()
        self.grid_size = 15
        self.initial_fire_positions = [(7, 7)]
        self.fire_spread_prob = 0.05
        self.max_steps = 150
        self.reward_structure = {
            'fire_suppressed': 15.0,
            'forest_saved': 2.0,
            'forest_burned': -1.0,
            'invalid_action': -0.3,
            'time_penalty': -0.05,
            'fire_contained': 75.0
        }


class MediumConfig(EnvConfig):
    """Medium configuration - moderate fires and spread"""
    
    def __init__(self):
        super().__init__()
        self.grid_size = 20
        self.initial_fire_positions = [(10, 10), (5, 15)]
        self.fire_spread_prob = 0.1
        self.wind_direction = (1, 0)  # Wind blowing right
        self.wind_strength = 0.3
        self.max_steps = 200


class HardConfig(EnvConfig):
    """Hard configuration - multiple fires, fast spread, wind"""
    
    def __init__(self):
        super().__init__()
        self.grid_size = 25
        self.initial_fire_positions = [(5, 5), (20, 5), (12, 20), (8, 15)]
        self.fire_spread_prob = 0.15
        self.wind_direction = (1, 1)  # Diagonal wind
        self.wind_strength = 0.5
        self.max_steps = 300
        self.reward_structure = {
            'fire_suppressed': 8.0,
            'forest_saved': 1.5,
            'forest_burned': -3.0,
            'invalid_action': -0.8,
            'time_penalty': -0.15,
            'fire_contained': 100.0
        }


class TrainingConfig(EnvConfig):
    """Randomized configuration for robust training"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        np.random.seed(seed)
        
        # Randomize grid size
        self.grid_size = np.random.choice([15, 20, 25])
        
        # Randomize number and position of fires
        num_fires = np.random.randint(1, 5)
        self.initial_fire_positions = []
        for _ in range(num_fires):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            self.initial_fire_positions.append((x, y))
        
        # Randomize fire spread probability
        self.fire_spread_prob = np.random.uniform(0.05, 0.2)
        
        # Randomize wind
        wind_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        angle = np.random.choice(wind_angles)
        self.wind_direction = (
            int(np.cos(np.radians(angle))),
            int(np.sin(np.radians(angle)))
        )
        self.wind_strength = np.random.uniform(0.0, 0.6)
        
        # Scale max steps with grid size
        self.max_steps = int(self.grid_size * 10)


class BenchmarkConfig(EnvConfig):
    """Standardized configuration for benchmarking and comparison"""
    
    def __init__(self):
        super().__init__()
        self.grid_size = 20
        self.initial_fire_positions = [(10, 10), (5, 5), (15, 15)]
        self.fire_spread_prob = 0.1
        self.wind_direction = (1, 0)
        self.wind_strength = 0.25
        self.max_steps = 200
        self.reward_structure = {
            'fire_suppressed': 10.0,
            'forest_saved': 1.0,
            'forest_burned': -2.0,
            'invalid_action': -0.5,
            'time_penalty': -0.1,
            'fire_contained': 50.0
        }


# Predefined scenarios
SCENARIOS = {
    'easy': EasyConfig,
    'medium': MediumConfig,
    'hard': HardConfig,
    'training': TrainingConfig,
    'benchmark': BenchmarkConfig
}


def get_config(scenario_name: str = 'medium', **kwargs) -> EnvConfig:
    """
    Get environment configuration by name
    
    Args:
        scenario_name: Name of the scenario ('easy', 'medium', 'hard', 'training', 'benchmark')
        **kwargs: Additional parameters to override in the configuration
    
    Returns:
        EnvConfig: Configuration object
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")
    
    config = SCENARIOS[scenario_name]()
    
    # Override any parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter '{key}' ignored")
    
    return config


def create_custom_config(
    grid_size: int = 20,
    num_fires: int = 1,
    fire_positions: Optional[List[Tuple[int, int]]] = None,
    fire_spread_prob: float = 0.1,
    wind_direction: Tuple[int, int] = (0, 0),
    wind_strength: float = 0.0,
    max_steps: Optional[int] = None,
    reward_multiplier: float = 1.0
) -> EnvConfig:
    """
    Create a custom environment configuration
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        num_fires: Number of initial fires (ignored if fire_positions is provided)
        fire_positions: Specific fire positions, if None will generate random positions
        fire_spread_prob: Probability of fire spreading to adjacent cells
        wind_direction: Wind direction as (dx, dy)
        wind_strength: Wind strength modifier (0.0 = no wind effect)
        max_steps: Maximum steps per episode
        reward_multiplier: Multiplier for all rewards
    
    Returns:
        EnvConfig: Custom configuration
    """
    config = EnvConfig()
    config.grid_size = grid_size
    config.fire_spread_prob = fire_spread_prob
    config.wind_direction = wind_direction
    config.wind_strength = wind_strength
    config.max_steps = max_steps or (grid_size * 10)
    
    # Set fire positions
    if fire_positions:
        config.initial_fire_positions = fire_positions
    else:
        # Generate random fire positions
        np.random.seed(42)  # For reproducibility
        config.initial_fire_positions = []
        for _ in range(num_fires):
            x = np.random.randint(0, grid_size)
            y = np.random.randint(0, grid_size)
            config.initial_fire_positions.append((x, y))
    
    # Scale rewards
    if reward_multiplier != 1.0:
        config.reward_structure = {
            'fire_suppressed': 10.0 * reward_multiplier,
            'forest_saved': 1.0 * reward_multiplier,
            'forest_burned': -2.0 * reward_multiplier,
            'invalid_action': -0.5 * reward_multiplier,
            'time_penalty': -0.1 * reward_multiplier,
            'fire_contained': 50.0 * reward_multiplier
        }
    
    return config
