from collections import deque, namedtuple
import math
import random
import time
from matplotlib import pyplot as plt
import matplotlib
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from forest_fire_rl import ForestFireEnv
from models import DeeQModel, DuelingDQN
from itertools import count

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Enable optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# MPS specific workarounds
if device.type == "mps":
    print("Warning: Using MPS backend with compatibility mode for macOS")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Optimized hyperparameters
BATCH_SIZE = 256  # Increased for better GPU utilization
GAMMA = 0.99  # Increased for better long-term planning
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000  # Adjusted for better exploration-exploitation balance
TAU = 0.005  # Soft update parameter
LR = 1e-3  # Optimized learning rate
UPDATE_FREQ = 4  # Update frequency for better stability
TARGET_UPDATE_FREQ = 100  # Less frequent target updates 


class PrioritizedReplayMemory:
    """Prioritized Experience Replay for better sample efficiency"""
    
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, *args):
        """Save a transition"""
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """Sample with prioritization"""
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        return samples, indices, torch.FloatTensor(weights).to(device)
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.memory)


class ReplayMemory(object):
    """Standard Experience Replay Buffer"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Environment setup with optimized parameters
env = ForestFireEnv(grid_size=25, fire_spread_prob=0.12, max_steps=200)

obs, info = env.reset()
print(f"Observation space: {env.observation_space.shape}")
print(f"Using device: {device}")

# Use Dueling DQN for better performance
policy_net = DuelingDQN(grid_size=25, n_state=4, ac=5).to(device)
target_net = DuelingDQN(grid_size=25, n_state=4, ac=5).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Optimized optimizer with better parameters
optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

# Use prioritized replay for better sample efficiency
memory = PrioritizedReplayMemory(50000, alpha=0.6)

steps_done = 0
episode_durations = []
rewards = []
losses = []

def select_action(state):
    """Improved action selection with better exploration"""
    global steps_done
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model_prioritized():
    """Optimized training with prioritized replay"""
    if len(memory) < BATCH_SIZE:
        return None
    
    # Sample from prioritized replay
    transitions, indices, weights = memory.sample(BATCH_SIZE, beta=0.4)
    batch = Transition(*zip(*transitions))

    # Compute masks and batches
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Current Q values
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Next state values from target network
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute TD errors for prioritized replay
    td_errors = (state_action_values.squeeze() - expected_state_action_values).abs().detach().cpu().numpy()
    
    # Weighted loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
    loss = (loss.squeeze() * weights).mean()

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    # Update priorities
    memory.update_priorities(indices, td_errors + 1e-6)
    
    return loss.item()


def optimize_model():
    """Standard optimization for regular replay buffer"""
    if len(memory) < BATCH_SIZE:
        return None
        
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Use Huber loss for stability
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def plot_training_metrics(show_result=False):
    """Plot training metrics including loss"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode durations
    if episode_durations:
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        ax1.plot(durations_t.numpy(), label='Episode Duration')
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax1.plot(means.numpy(), label='100-Episode Average')
        ax1.set_title('Episode Duration')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Duration')
        ax1.legend()
    
    # Rewards
    if rewards:
        ax2.plot(rewards, label='Episode Reward')
        if len(rewards) >= 100:
            reward_means = np.convolve(rewards, np.ones(100)/100, mode='valid')
            ax2.plot(range(99, len(rewards)), reward_means, label='100-Episode Average')
        ax2.set_title('Episode Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.legend()
    
    # Loss
    if losses:
        ax3.plot(losses, alpha=0.7)
        if len(losses) >= 100:
            loss_means = np.convolve(losses, np.ones(100)/100, mode='valid')
            ax3.plot(range(99, len(losses)), loss_means, label='100-Step Average')
        ax3.set_title('Training Loss')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.set_yscale('log')
    
    # Learning rate
    current_lr = optimizer.param_groups[0]['lr']
    ax4.axhline(y=current_lr, color='r', linestyle='--', label=f'Current LR: {current_lr:.6f}')
    ax4.set_title('Learning Rate')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Learning Rate')
    ax4.legend()
    
    plt.tight_layout()
    plt.pause(0.001)
    
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
# Optimized training parameters
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 2000  # Increased for better convergence
else:
    num_episodes = 100

def demo_run(env, model):
    """Run a demo episode with the given model"""
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    
    for step in range(200):  # Increased step limit
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = model(state).max(1).indices.item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:  # Render less frequently for speed
            env.render()
        
        if terminated or truncated:
            print(f"Demo episode ended at step {step}")
            break
    
    print(f"Demo total reward: {total_reward:.2f}")
    env.close()


# Main training loop with optimizations
print(f"Starting training for {num_episodes} episodes...")

for i_episode in range(num_episodes):
    # Initialize episode
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_episode_reward = 0.0
    update_count = 0
    
    # Demo run less frequently for performance
    if i_episode % 500 == 0 and i_episode > 0:
        print(f"\nRunning demo at episode {i_episode}")
        demo_run(env, policy_net)
    
    for t in count():
        # Select and perform action
        action = select_action(state)
        observation, reward_step, terminated, truncated, _ = env.step(action.item())
        
        total_episode_reward += reward_step
        
        reward_tensor = torch.tensor([reward_step], device=device, dtype=torch.float32)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store transition
        memory.push(state, action, next_state, reward_tensor)
        state = next_state

        # Perform optimization every UPDATE_FREQ steps
        if steps_done % UPDATE_FREQ == 0:
            if isinstance(memory, PrioritizedReplayMemory):
                loss = optimize_model_prioritized()
            else:
                loss = optimize_model()
            
            if loss is not None:
                losses.append(loss)
            
            update_count += 1

        # Update target network less frequently
        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            episode_durations.append(t + 1)
            rewards.append(total_episode_reward)
            
            # Update learning rate
            scheduler.step()
            
            # Print progress
            if i_episode % 50 == 0:
                avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                print(f"Episode {i_episode:4d}: Reward={total_episode_reward:7.2f}, "
                      f"Avg Reward={avg_reward:7.2f}, Duration={t+1:3d}, "
                      f"Epsilon={EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY):.3f}, "
                      f"Updates={update_count}")
                
                # Plot metrics less frequently for performance
                if i_episode % 100 == 0:
                    plot_training_metrics()
            break

print('\nTraining Complete!')
plot_training_metrics(show_result=True)
plt.ioff()
plt.show()

# Save trained model
torch.save({
    'policy_net_state_dict': policy_net.state_dict(),
    'target_net_state_dict': target_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'episode_durations': episode_durations,
    'rewards': rewards,
    'losses': losses
}, 'forest_fire_dqn_model.pth')
print("Model saved as 'forest_fire_dqn_model.pth'")
