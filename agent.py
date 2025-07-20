from collections import deque, namedtuple
import math
import random
import time
from matplotlib import pyplot as plt
import matplotlib
import torch
from torch import nn
from torch import optim
import numpy as np
from forest_fire_rl import ForestFireEnv
from models import DeeQModel
from itertools import count

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.95  # Reduced from 0.98 to focus more on immediate rewards
EPS_START = 0.9
EPS_END = 0.05  # Increased from 0.01 to maintain more exploration
EPS_DECAY = 5000  # Reduced from 25500 for faster epsilon decay
TAU = 0.01  # Increased from 0.005 for faster target network updates
LR = 5e-4  # Increased learning rate from 1e-4 


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def random_policy(obs):
    """Random policy for demonstration"""
    return np.random.randint(0, 5)


env = ForestFireEnv(grid_size=25, fire_spread_prob=0.15)

obs, info = env.reset()
print(env.observation_space.shape)
policy_net = DeeQModel(grid_size=25, n_state=4, ac=5).to(device)
target_net = DeeQModel(grid_size=25, n_state=4, ac=5).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample < 0.01:
        print(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
def plot_rewards(rewards):
    plt.figure(2)
    plt.clf()
    plt.title('Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(rewards)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)


episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 1500
else:
    num_episodes = 50

def demo_run(env, model):
    """Run a demo episode with the given model"""
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    
    for step in range(100):
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = model(state).max(1).indices.item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()

rewards = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # Track the total reward for the episode
    total_episode_reward = 0.0
    
    if i_episode % 100 == 0:
        demo_run(env, policy_net)
            
    for t in count():
        action = select_action(state)
        observation, reward_step, terminated, truncated, _ = env.step(action.item())
        
        total_episode_reward += reward_step
        
        reward_tensor = torch.tensor([reward_step], device=device, dtype=torch.float32)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward_tensor)

        state = next_state

        optimize_model()

        # θ′ ← τ θ + (1 − τ)θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            rewards.append(total_episode_reward)
            if i_episode % 10 == 0:
                print(f"Episode {i_episode}: Total Reward = {total_episode_reward:.2f}, Duration = {t+1}")
                plot_durations()
                plot_rewards(rewards)
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()