"""
Forest Fire RL Environment Package

A high-performance Gymnasium environment for reinforcement learning
with forest fire simulation and firefighting agents.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__license__ = "MIT"

# Core components
from .core import (
    ForestFireEnv,
    get_config, 
    create_custom_config,
    EasyConfig,
    MediumConfig, 
    HardConfig,
    TrainingConfig,
    BenchmarkConfig
)

# Training utilities
from .training import (
    VectorizedForestFireEnv,
    EnvironmentBenchmark,
    EpisodeLogger,
    random_policy,
    heuristic_policy,
    run_episode,
    evaluate_policy
)

# Visualization
from .visualization import (
    SimpleVisualizer,
    InteractiveVisualizer,
    AdvancedVisualizer,
)

# Agents
from .agents import (
    BaseAgent,
    RandomAgent,
    TorchRandomAgent,
    TorchHeuristicAgent,
)

__all__ = [
    # Core
    "ForestFireEnv",
    "get_config",
    "create_custom_config",
    "EasyConfig", 
    "MediumConfig",
    "HardConfig",
    "TrainingConfig",
    "BenchmarkConfig",
    # Training
    "VectorizedForestFireEnv",
    "EnvironmentBenchmark", 
    "EpisodeLogger",
    "random_policy",
    "heuristic_policy",
    "run_episode",
    "evaluate_policy",
    # Visualization
    "SimpleVisualizer",
    "InteractiveVisualizer", 
    "AdvancedVisualizer",
    # Agents
    "BaseAgent",
    "RandomAgent", 
    "TorchRandomAgent",
    "TorchHeuristicAgent",
]
