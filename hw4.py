
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import os

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
EPISODES        = 700          # Train for at least 600
BATCH_SIZE      = 128
GAMMA           = 0.99         # Discount factor
LR              = 5e-4         # Learning rate
MEMORY_SIZE     = 100_000      # Replay buffer capacity
TAU             = 1e-3         # Soft-update coefficient for target network
EPS_START       = 1.0          # Initial epsilon (exploration)
EPS_END         = 0.01         # Minimum epsilon
EPS_DECAY       = 0.995        # Epsilon decay per episode
UPDATE_EVERY    = 4            # Steps between network updates
HIDDEN_SIZE     = 256          # Hidden layer width
VIDEO_FOLDER    = "videos"
EVAL_EPISODES   = 100          # Episodes for final evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ─────────────────────────────────────────────
# Q-Network (Dueling DQN architecture)
# ─────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_SIZE):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        # Combine: Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# ─────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPS_START
        self.step_count = 0

        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.policy_net(state_t).argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        self.step_count += 1
        if self.step_count % UPDATE_EVERY != 0:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states      = torch.FloatTensor(np.array([t.state      for t in batch])).to(device)
        actions     = torch.LongTensor( np.array([t.action     for t in batch])).unsqueeze(1).to(device)
        rewards     = torch.FloatTensor(np.array([t.reward     for t in batch])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(device)
        dones       = torch.FloatTensor(np.array([t.done       for t in batch])).unsqueeze(1).to(device)

        # Double DQN: action selected by policy, evaluated by target
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            target_q = rewards + GAMMA * (1 - dones) * self.target_net(next_states).gather(1, best_actions)

        current_q = self.policy_net(states).gather(1, actions)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def make_env(record=False, episode_trigger=None):
    env = gym.make("LunarLander-v3", render_mode="rgb_array" if record else None)
    if record:
        os.makedirs(VIDEO_FOLDER, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=VIDEO_FOLDER,
            episode_trigger=episode_trigger,
            name_prefix="lunar_lander",
        )
    return env

def train():
    # Record every 25 episodes, starting within the first 50
    def episode_trigger(ep_id):
        return ep_id % 25 == 0

    env = make_env(record=True, episode_trigger=episode_trigger)
    state_dim  = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.n               # 4

    agent = DQNAgent(state_dim, action_dim)

    episode_rewards   = []
    episode_durations = []
    rolling_avg       = []

    print(f"\n{'─'*55}")
    print(f"  Training DQN on LunarLander-v3 for {EPISODES} episodes")
    print(f"{'─'*55}\n")

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        duration = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, float(done))
            agent.learn()

            state = next_state
            total_reward += reward
            duration += 1

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_durations.append(duration)

        avg100 = np.mean(episode_rewards[-100:])
        rolling_avg.append(avg100)

        if episode % 25 == 0 or episode == 1:
            print(f"Ep {episode:>4}/{EPISODES}  |  "
                  f"Reward: {total_reward:>8.1f}  |  "
                  f"Avg(100): {avg100:>8.1f}  |  "
                  f"ε: {agent.epsilon:.3f}")

        if episode >= 100 and avg100 >= 200:
            print(f"\n✓ Solved at episode {episode}! Avg(100) = {avg100:.1f}")

    env.close()

    # Save model
    torch.save(agent.policy_net.state_dict(), "dqn_lunar_lander.pth")
    print("\nModel saved to dqn_lunar_lander.pth")

    return agent, episode_rewards, episode_durations, rolling_avg

# ─────────────────────────────────────────────
# Plot Training Graphs
# ─────────────────────────────────────────────
def plot_training(episode_rewards, episode_durations, rolling_avg):
    episodes = range(1, len(episode_rewards) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("DQN LunarLander-v3 – Training Curves", fontsize=14, fontweight="bold")

    # Reward plot
    ax1.plot(episodes, episode_rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax1.plot(episodes, rolling_avg,     color="navy",      linewidth=2, label="Rolling avg (100 ep)")
    ax1.axhline(200, color="green", linestyle="--", linewidth=1.5, label="Passing threshold (200)")
    ax1.set_ylabel("Reward")
    ax1.set_xlabel("Episode")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Duration plot
    ax2.plot(episodes, episode_durations, alpha=0.4, color="coral", label="Steps per episode")
    window = min(100, len(episode_durations))
    dur_avg = np.convolve(episode_durations, np.ones(window)/window, mode="valid")
    ax2.plot(range(window, len(episode_durations)+1), dur_avg, color="darkred", linewidth=2, label=f"Rolling avg ({window} ep)")
    ax2.set_ylabel("Duration (steps)")
    ax2.set_xlabel("Episode")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Training curves saved to training_curves.png")

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(agent, n_episodes=EVAL_EPISODES):
    print(f"\nEvaluating agent for {n_episodes} episodes (greedy policy)…")
    env = gym.make("LunarLander-v3")
    eval_rewards   = []
    eval_durations = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        duration = 0
        while True:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            duration += 1
            if terminated or truncated:
                break
        eval_rewards.append(total_reward)
        eval_durations.append(duration)

    env.close()

    mean_r = np.mean(eval_rewards)
    mean_d = np.mean(eval_durations)
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward   : {mean_r:.2f}  ({'PASS ✓' if mean_r >= 200 else 'FAIL ✗'})")
    print(f"  Mean Duration : {mean_d:.1f} steps")

    # Histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("DQN LunarLander-v3 – Evaluation Histograms", fontsize=13, fontweight="bold")

    ax1.hist(eval_rewards, bins=20, color="steelblue", edgecolor="white")
    ax1.axvline(mean_r, color="navy", linestyle="--", linewidth=2, label=f"Mean = {mean_r:.1f}")
    ax1.axvline(200,    color="green", linestyle="--", linewidth=1.5, label="Threshold (200)")
    ax1.set_xlabel("Reward")
    ax1.set_ylabel("Count")
    ax1.set_title("Reward Distribution")
    ax1.legend()

    ax2.hist(eval_durations, bins=20, color="coral", edgecolor="white")
    ax2.axvline(mean_d, color="darkred", linestyle="--", linewidth=2, label=f"Mean = {mean_d:.1f}")
    ax2.set_xlabel("Duration (steps)")
    ax2.set_ylabel("Count")
    ax2.set_title("Duration Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("eval_histograms.png", dpi=150)
    plt.show()
    print("Evaluation histograms saved to eval_histograms.png")

    return eval_rewards, eval_durations

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent, ep_rewards, ep_durations, rolling_avg = train()
    plot_training(ep_rewards, ep_durations, rolling_avg)
    evaluate(agent)