#!/usr/bin/env python3
"""
Example training script for Forest Fire RL Environment
Shows how to train a simple Q-learning agent
"""

import numpy as np
import pickle
import time
from collections import defaultdict, deque

from forest_fire_env import ForestFireEnv
from env_configs import get_config
from training_utils import EpisodeLogger, evaluate_policy


class SimpleQLearningAgent:
    """
    Simple Q-Learning agent for the forest fire environment
    Uses discretized state space for tractability
    """
    
    def __init__(self, action_space_size=6, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: defaultdict for automatic initialization
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        
        # Training statistics
        self.training_episodes = 0
    
    def get_state_key(self, observation):
        """
        Convert observation to a hashable state key
        Discretizes the state to make Q-table tractable
        """
        grid = observation['grid']
        firefighter_pos = tuple(observation['firefighter_pos'])
        
        # Count different cell types
        fire_count = np.sum(grid == 2)
        forest_count = np.sum(grid == 1)
        
        # Find nearest fire distance (simplified)
        fire_positions = np.where(grid == 2)
        if len(fire_positions[0]) > 0:
            distances = []
            for i in range(len(fire_positions[0])):
                fire_pos = (fire_positions[0][i], fire_positions[1][i])
                dist = abs(fire_pos[0] - firefighter_pos[0]) + abs(fire_pos[1] - firefighter_pos[1])
                distances.append(dist)
            nearest_fire_dist = min(distances)
        else:
            nearest_fire_dist = -1  # No fires
        
        # Discretize fire count and forest count for manageable state space
        fire_count_discrete = min(fire_count // 3, 10)  # Group fires in buckets of 3
        forest_count_discrete = min(forest_count // 10, 20)  # Group forest in buckets of 10
        
        # Create state key
        state_key = (
            firefighter_pos,
            fire_count_discrete,
            forest_count_discrete,
            min(nearest_fire_dist, 15)  # Cap distance at 15
        )
        
        return state_key
    
    def choose_action(self, observation, training=True):
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(observation)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning update rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state) if not done else None
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * next_max_q
        
        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save the Q-table and agent parameters"""
        agent_data = {
            'q_table': dict(self.q_table),
            'action_space_size': self.action_space_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'training_episodes': self.training_episodes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load the Q-table and agent parameters"""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size), agent_data['q_table'])
        self.action_space_size = agent_data['action_space_size']
        self.learning_rate = agent_data['learning_rate']
        self.discount_factor = agent_data['discount_factor']
        self.epsilon = agent_data['epsilon']
        self.training_episodes = agent_data['training_episodes']
        
        print(f"Agent loaded from {filepath}")
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'training_episodes': self.training_episodes
        }


def train_agent(config, num_episodes=2000, eval_frequency=200, save_frequency=500):
    """Train the Q-learning agent"""
    
    # Create environment and agent
    env = ForestFireEnv(**config.to_dict())
    agent = SimpleQLearningAgent()
    logger = EpisodeLogger(log_frequency=100)
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Environment: {config.grid_size}x{config.grid_size} grid")
    
    # Training loop
    start_time = time.time()
    recent_rewards = deque(maxlen=100)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Choose and take action
            action = agent.choose_action(obs, training=True)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            # Update agent
            agent.update(obs, action, reward, next_obs, terminated or truncated)
            
            total_reward += reward
            steps += 1
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Update training statistics
        agent.training_episodes += 1
        agent.decay_epsilon()
        recent_rewards.append(total_reward)
        logger.log_episode(total_reward, steps, next_info)
        
        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Recent average reward: {np.mean(recent_rewards):.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Q-table size: {len(agent.q_table)}")
            
            # Evaluate current policy
            eval_policy = lambda obs: agent.choose_action(obs, training=False)
            eval_stats = evaluate_policy(config, eval_policy, num_episodes=20)
            print(f"Evaluation - Mean reward: {eval_stats['mean_reward']:.2f}, "
                  f"Success rate: {eval_stats['success_rate']:.2%}")
        
        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            agent.save(f'agent_checkpoint_{episode+1}.pkl')
    
    # Final training statistics
    training_time = time.time() - start_time
    final_stats = logger.get_stats()
    agent_stats = agent.get_stats()
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final average reward: {final_stats['avg_reward']:.2f}")
    print(f"Q-table size: {agent_stats['q_table_size']}")
    
    # Save final agent
    agent.save('trained_agent.pkl')
    
    env.close()
    return agent, final_stats


def test_agent(config, agent_filepath='trained_agent.pkl', num_episodes=50, render=False):
    """Test a trained agent"""
    
    # Load agent
    agent = SimpleQLearningAgent()
    agent.load(agent_filepath)
    
    # Create environment
    render_mode = "human" if render else None
    env = ForestFireEnv(**config.to_dict(), render_mode=render_mode)
    
    print(f"Testing agent for {num_episodes} episodes...")
    
    # Test policy
    test_policy = lambda obs: agent.choose_action(obs, training=False)
    
    if render:
        # Run a few episodes with visualization
        for episode in range(min(3, num_episodes)):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode+1}:")
            
            while True:
                action = test_policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if render:
                    env.render()
                    time.sleep(0.1)
                
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            print(f"Reward: {total_reward:.2f}, Steps: {steps}, "
                  f"Forest saved: {info['forest_preservation_ratio']:.1%}")
    
    # Full evaluation without rendering
    env.close()
    env = ForestFireEnv(**config.to_dict())
    eval_stats = evaluate_policy(config, test_policy, num_episodes=num_episodes)
    
    print(f"\nTest Results over {num_episodes} episodes:")
    print(f"Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"Success rate: {eval_stats['success_rate']:.2%}")
    print(f"Mean forest saved: {eval_stats['mean_forest_saved']:.1%}")
    print(f"Best episode reward: {eval_stats['max_reward']:.2f}")
    
    env.close()
    return eval_stats


def main():
    """Main training and testing pipeline"""
    
    print("Forest Fire RL Training Example")
    print("===============================")
    
    # Configuration
    config = get_config('easy')  # Start with easy configuration
    
    # Training phase
    print("Phase 1: Training Agent")
    agent, train_stats = train_agent(config, num_episodes=1000, eval_frequency=200)
    
    # Testing phase
    print("\nPhase 2: Testing Trained Agent")
    test_stats = test_agent(config, num_episodes=100, render=False)
    
    # Compare with baseline
    print("\nPhase 3: Baseline Comparison")
    from training_utils import random_policy, heuristic_policy
    
    random_stats = evaluate_policy(config, random_policy, num_episodes=100)
    heuristic_stats = evaluate_policy(config, heuristic_policy, num_episodes=100)
    
    print("Performance Comparison:")
    print(f"Random Policy:     {random_stats['mean_reward']:.2f} (success: {random_stats['success_rate']:.1%})")
    print(f"Heuristic Policy:  {heuristic_stats['mean_reward']:.2f} (success: {heuristic_stats['success_rate']:.1%})")
    print(f"Trained Q-Agent:   {test_stats['mean_reward']:.2f} (success: {test_stats['success_rate']:.1%})")
    
    # Calculate improvement
    improvement_vs_random = (test_stats['mean_reward'] - random_stats['mean_reward']) / abs(random_stats['mean_reward']) * 100
    improvement_vs_heuristic = (test_stats['mean_reward'] - heuristic_stats['mean_reward']) / abs(heuristic_stats['mean_reward']) * 100
    
    print(f"\nImprovement:")
    print(f"vs Random: {improvement_vs_random:+.1f}%")
    print(f"vs Heuristic: {improvement_vs_heuristic:+.1f}%")
    
    print(f"\nTraining completed! Agent saved as 'trained_agent.pkl'")
    print("You can test the agent anytime by running:")
    print("python -c \"from train_example import test_agent; from env_configs import get_config; test_agent(get_config('easy'), render=True)\"")


if __name__ == "__main__":
    main()
