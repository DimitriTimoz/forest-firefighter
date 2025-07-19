"""
PyTorch-based agents for Forest Fire RL Environment
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Union, List
import gymnasium as gym

from .base import BaseAgent


class TorchRandomAgent(BaseAgent):
    """Random agent implemented with PyTorch for consistency with RL training"""
    
    def __init__(self, action_space: gym.spaces.Space, device: str = "cpu", seed: int = None):
        """
        Initialize PyTorch random agent
        
        Args:
            action_space: The action space of the environment
            device: Device to run computations on ('cpu' or 'cuda')
            seed: Random seed for reproducibility
        """
        super().__init__(action_space)
        self.device = torch.device(device)
        
        # Handle both Discrete and MultiDiscrete action spaces
        if hasattr(action_space, 'n'):
            # Single discrete action space
            self.n_actions = action_space.n
            self.multi_discrete = False
        else:
            # Multi-discrete action space
            self.n_actions = action_space.nvec[0]  # Assume all agents have same action space
            self.multi_discrete = True
            self.num_agents = len(action_space.nvec)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def act(self, observation: Dict[str, Any]) -> Union[int, List[int]]:
        """Choose a random action using PyTorch"""
        # Convert to tensor for consistency with RL pipelines
        if self.multi_discrete:
            # Multi-agent: return list of actions
            actions = []
            for _ in range(self.num_agents):
                action_probs = torch.ones(self.n_actions, device=self.device) / self.n_actions
                action = torch.multinomial(action_probs, 1).item()
                actions.append(action)
            return actions
        else:
            # Single agent: return single action
            action_probs = torch.ones(self.n_actions, device=self.device) / self.n_actions
            action = torch.multinomial(action_probs, 1).item()
            return action
    
    def get_action_distribution(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Get uniform action distribution"""
        return torch.ones(self.n_actions, device=self.device) / self.n_actions


class TorchHeuristicAgent(BaseAgent):
    """Heuristic agent that tries to move towards fires"""
    
    def __init__(self, action_space: gym.spaces.Space, device: str = "cpu"):
        """
        Initialize heuristic agent
        
        Args:
            action_space: The action space of the environment  
            device: Device to run computations on
        """
        super().__init__(action_space)
        self.device = torch.device(device)
        
        # Handle both Discrete and MultiDiscrete action spaces
        if hasattr(action_space, 'n'):
            self.multi_discrete = False
        else:
            self.multi_discrete = True
            self.num_agents = len(action_space.nvec)
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right, 4=suppress, 5=nothing
        self.action_map = {
            'up': 0,
            'down': 1, 
            'left': 2,
            'right': 3,
            'suppress': 4,
            'nothing': 5
        }
    
    def act(self, observation: Dict[str, Any]) -> Union[int, List[int]]:
        """Choose action(s) based on simple heuristic"""
        grid = observation['grid']
        
        # Handle both old and new observation formats
        if 'firefighter_pos' in observation:
            # Legacy format: single firefighter
            firefighter_positions = observation['firefighter_pos'].reshape(1, -1)
        else:
            # New format: multiple firefighters
            firefighter_positions = observation['firefighter_positions']
        
        # Convert to tensors
        grid_tensor = torch.from_numpy(grid).to(self.device)
        positions_tensor = torch.from_numpy(firefighter_positions).to(self.device)
        
        actions = []
        
        # Decide action for each firefighter
        for i in range(len(firefighter_positions)):
            ff_pos = positions_tensor[i]
            ff_row, ff_col = ff_pos[0].item(), ff_pos[1].item()
            
            # Check if there's fire adjacent to this firefighter
            adjacent_positions = [
                (ff_row - 1, ff_col),  # up
                (ff_row + 1, ff_col),  # down  
                (ff_row, ff_col - 1),  # left
                (ff_row, ff_col + 1),  # right
            ]
            
            # Suppress if fire is adjacent
            adjacent_fire = False
            for pos in adjacent_positions:
                if (0 <= pos[0] < grid.shape[0] and 
                    0 <= pos[1] < grid.shape[1] and
                    grid_tensor[pos[0], pos[1]] == 2):  # Fire state
                    actions.append(self.action_map['suppress'])
                    adjacent_fire = True
                    break
            
            if adjacent_fire:
                continue
            
            # Find nearest fire
            fire_positions = torch.where(grid_tensor == 2)
            if len(fire_positions[0]) > 0:
                # Calculate distances to all fires
                fire_coords = torch.stack([fire_positions[0], fire_positions[1]], dim=1).float()
                ff_coords = ff_pos.float().unsqueeze(0)
                
                distances = torch.norm(fire_coords - ff_coords, dim=1)
                nearest_fire_idx = torch.argmin(distances)
                nearest_fire = fire_coords[nearest_fire_idx]
                
                # Move towards nearest fire
                diff_row = nearest_fire[0] - ff_row
                diff_col = nearest_fire[1] - ff_col
                
                # Prioritize larger movement direction
                if abs(diff_row) > abs(diff_col):
                    if diff_row > 0:
                        actions.append(self.action_map['down'])
                    else:
                        actions.append(self.action_map['up'])
                else:
                    if diff_col > 0:
                        actions.append(self.action_map['right']) 
                    else:
                        actions.append(self.action_map['left'])
            else:
                # No fires found, do nothing
                actions.append(self.action_map['nothing'])
        
        # Return single action for single firefighter, list for multiple
        return actions[0] if len(actions) == 1 else actions
