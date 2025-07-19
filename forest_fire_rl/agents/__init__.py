"""Agents for the Forest Fire RL Environment"""

from .base import BaseAgent, RandomAgent
from .torch_agents import TorchRandomAgent, TorchHeuristicAgent

__all__ = [
    "BaseAgent",
    "RandomAgent", 
    "TorchRandomAgent",
    "TorchHeuristicAgent",
]
