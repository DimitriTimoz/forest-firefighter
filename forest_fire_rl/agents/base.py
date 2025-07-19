"""
Base agent classes for Forest Fire RL Environment
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Union
import gymnasium as gym


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, action_space: gym.spaces.Space, **kwargs):
        """
        Initialize base agent
        
        Args:
            action_space: The action space of the environment
        """
        self.action_space = action_space
        
    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> int:
        """
        Choose an action given an observation
        
        Args:
            observation: Current observation from environment
            
        Returns:
            Selected action
        """
        pass
    
    def reset(self):
        """Reset agent state (optional override)"""
        pass


class RandomAgent(BaseAgent):
    """Simple random agent that takes random actions"""
    
    def __init__(self, action_space: gym.spaces.Space, seed: int = None):
        """
        Initialize random agent
        
        Args:
            action_space: The action space of the environment
            seed: Random seed for reproducibility
        """
        super().__init__(action_space)
        self.np_random = np.random.RandomState(seed)
        
    def act(self, observation: Dict[str, Any]) -> int:
        """Choose a random action"""
        return self.action_space.sample()
    
    def set_seed(self, seed: int):
        """Set random seed"""
        self.np_random = np.random.RandomState(seed)
