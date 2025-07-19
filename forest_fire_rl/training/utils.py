"""Training utilities for Forest Fire Environment"""

import numpy as np
import gymnasium as gym
from typing import List, Dict, Any, Tuple, Optional
import time

from ..core.environment import ForestFireEnv
from ..core.configs import EnvConfig


class VectorizedForestFireEnv:
    """
    Vectorized environment for parallel simulation of multiple forest fire environments
    Optimized for efficient batch training and evaluation
    """
    
    def __init__(self, num_envs: int, config: EnvConfig, seed: Optional[int] = None):
        self.num_envs = num_envs
        self.config = config
        self.seed = seed
        
        # Create environments
        self.envs = []
        for i in range(num_envs):
            env_seed = seed + i if seed is not None else None
            env = ForestFireEnv(**config.to_dict())
            env.reset(seed=env_seed)
            self.envs.append(env)
        
        # Cached properties
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        
        # Performance tracking
        self.step_times = []
        self.total_episodes = 0
    
    def reset(self, env_indices: Optional[List[int]] = None) -> Tuple[List[Dict], List[Dict]]:
        """Reset environments and return observations and info"""
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        
        observations = []
        infos = []
        
        for i in env_indices:
            obs, info = self.envs[i].reset()
            observations.append(obs)
            infos.append(info)
        
        return observations, infos
    
    def step(self, actions: List[int]) -> Tuple[List[Dict], List[float], List[bool], List[bool], List[Dict]]:
        """Step all environments with given actions"""
        start_time = time.time()
        
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = self.envs[i].step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
            
            # Auto-reset terminated environments
            if terminated or truncated:
                self.envs[i].reset()
                self.total_episodes += 1
        
        self.step_times.append(time.time() - start_time)
        return observations, rewards, terminateds, truncateds, infos
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.step_times:
            return {}
        
        return {
            'avg_step_time': np.mean(self.step_times[-1000:]),  # Last 1000 steps
            'steps_per_second': 1.0 / np.mean(self.step_times[-1000:]),
            'total_episodes': self.total_episodes,
            'environments': self.num_envs
        }


class EnvironmentBenchmark:
    """Benchmark tool for measuring environment performance"""
    
    def __init__(self, config: EnvConfig):
        self.config = config
    
    def benchmark_single_env(self, num_steps: int = 1000, num_runs: int = 3) -> Dict[str, float]:
        """Benchmark single environment performance"""
        times = []
        
        for run in range(num_runs):
            env = ForestFireEnv(**self.config.to_dict())
            obs, info = env.reset()
            
            start_time = time.time()
            
            for step in range(num_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    env.reset()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        
        return {
            'total_time': avg_time,
            'steps_per_second': num_steps / avg_time,
            'time_per_step': avg_time / num_steps,
            'runs': num_runs,
            'steps': num_steps
        }
    
    def benchmark_vectorized_env(self, num_envs: int, num_steps: int = 1000) -> Dict[str, float]:
        """Benchmark vectorized environment performance"""
        vec_env = VectorizedForestFireEnv(num_envs, self.config)
        observations, infos = vec_env.reset()
        
        start_time = time.time()
        
        for step in range(num_steps):
            actions = [env.action_space.sample() for env in vec_env.envs]
            observations, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        stats = vec_env.get_performance_stats()
        stats.update({
            'total_time': total_time,
            'effective_steps_per_second': (num_steps * num_envs) / total_time,
            'speedup_vs_single': ((num_steps * num_envs) / total_time) / (num_steps / total_time * num_envs)
        })
        
        return stats


class EpisodeLogger:
    """Logger for tracking episode statistics during training"""
    
    def __init__(self, log_frequency: int = 100):
        self.log_frequency = log_frequency
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_info = []
        self.episode_count = 0
    
    def log_episode(self, total_reward: float, episode_length: int, info: Dict[str, Any]):
        """Log episode data"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.episode_info.append(info)
        self.episode_count += 1
        
        if self.episode_count % self.log_frequency == 0:
            self._print_stats()
    
    def _print_stats(self):
        """Print episode statistics"""
        recent_rewards = self.episode_rewards[-self.log_frequency:]
        recent_lengths = self.episode_lengths[-self.log_frequency:]
        recent_info = self.episode_info[-self.log_frequency:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        avg_forest_saved = np.mean([info.get('forest_preservation_ratio', 0) for info in recent_info])
        
        print(f"Episodes {self.episode_count-self.log_frequency+1}-{self.episode_count}:")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Length: {avg_length:.1f}")
        print(f"  Avg Forest Saved: {avg_forest_saved:.2%}")
        print(f"  Reward Range: [{min(recent_rewards):.2f}, {max(recent_rewards):.2f}]")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': self.episode_count,
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'avg_forest_saved': np.mean([info.get('forest_preservation_ratio', 0) for info in self.episode_info]),
            'best_reward': max(self.episode_rewards),
            'worst_reward': min(self.episode_rewards)
        }


def run_episode(env: ForestFireEnv, policy, max_steps: Optional[int] = None) -> Dict[str, Any]:
    """Run a single episode with given policy"""
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated or (max_steps and steps >= max_steps):
            break
    
    return {'total_reward': total_reward, 'steps': steps, 'final_info': info}


def evaluate_policy(env_config: EnvConfig, policy, num_episodes: int = 100) -> Dict[str, Any]:
    """Evaluate policy performance across multiple episodes"""
    env = ForestFireEnv(**env_config.to_dict())
    results = []
    
    for episode in range(num_episodes):
        result = run_episode(env, policy)
        results.append(result)
        
        if (episode + 1) % 10 == 0:
            print(f"Evaluated {episode + 1}/{num_episodes} episodes")
    
    # Aggregate results
    rewards = [r['total_reward'] for r in results]
    lengths = [r['steps'] for r in results]
    forest_ratios = [r['final_info'].get('forest_preservation_ratio', 0) for r in results]
    
    return {
        'num_episodes': len(results),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'mean_forest_saved': np.mean(forest_ratios),
        'success_rate': np.mean([r > 0 for r in rewards])
    }


# Simple policies for testing
def random_policy(obs: Dict[str, Any]) -> int:
    """Random policy for baseline testing"""
    return np.random.randint(0, 6)


def heuristic_policy(obs: Dict[str, Any]) -> int:
    """Heuristic policy that moves toward fires and suppresses them"""
    grid = obs['grid']
    firefighter_pos = obs.get('firefighter_pos', obs.get('firefighter_positions', [[0, 0]])[0])
    
    # Find nearest fire
    fire_positions = np.where(grid == 2)
    if len(fire_positions[0]) == 0:
        return 5  # Do nothing if no fires
    
    fire_coords = list(zip(fire_positions[0], fire_positions[1]))
    distances = [abs(pos[0] - firefighter_pos[0]) + abs(pos[1] - firefighter_pos[1]) 
                for pos in fire_coords]
    nearest_fire = fire_coords[np.argmin(distances)]
    
    # Check if adjacent to fire - suppress
    if min(distances) <= 1:
        return 4
    
    # Move toward nearest fire
    dx = nearest_fire[0] - firefighter_pos[0]
    dy = nearest_fire[1] - firefighter_pos[1]
    
    return 1 if abs(dx) > abs(dy) and dx > 0 else 0 if abs(dx) > abs(dy) else 3 if dy > 0 else 2
