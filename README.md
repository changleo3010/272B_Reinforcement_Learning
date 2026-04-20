# DQN LunarLander-v3

A Deep Q-Network (DQN) agent trained to solve the `LunarLander-v3` environment from OpenAI Gymnasium, using a **Dueling DQN + Double DQN** architecture with experience replay and soft target-network updates.

---

## Results

| Metric | Value |
|---|---|
| Eval Mean Reward (100 ep) | **242.6** ✓ (threshold: 200) |
| Eval Mean Duration | 371.0 steps |
| Training Episodes | 700 |
| Solved At (rolling avg ≥ 200) | ~Episode 450 |

---

## Architecture

### Dueling DQN

```
Input (8) → Linear(256) → ReLU → Linear(256) → ReLU
                                                    ├─ Value stream:     Linear(128) → ReLU → Linear(1)
                                                    └─ Advantage stream: Linear(128) → ReLU → Linear(4)
Output: Q(s,a) = V(s) + A(s,a) − mean(A(s,·))
```

### Key Techniques

- **Dueling DQN** — separates state value and action advantage estimation
- **Double DQN** — action selected by policy net, evaluated by target net (reduces overestimation bias)
- **Experience Replay** — uniform random sampling from a 100k-transition replay buffer
- **Soft Target Updates** — target network updated via exponential moving average (τ = 1e-3)
- **Gradient Clipping** — max norm of 1.0 to stabilize training
- **ε-greedy Exploration** — epsilon decays multiplicatively from 1.0 → 0.01

---

## Hyperparameters

```python
EPISODES      = 700        # Training episodes
BATCH_SIZE    = 128
GAMMA         = 0.99       # Discount factor
LR            = 5e-4       # Adam learning rate
MEMORY_SIZE   = 100_000    # Replay buffer capacity
TAU           = 1e-3       # Soft-update coefficient
EPS_START     = 1.0        # Initial epsilon
EPS_END       = 0.01       # Minimum epsilon
EPS_DECAY     = 0.995      # Per-episode epsilon decay
UPDATE_EVERY  = 4          # Steps between gradient updates
HIDDEN_SIZE   = 256        # Hidden layer width
EVAL_EPISODES = 100        # Greedy evaluation episodes
```

---

## Training Behavior (4 Eras)

| Era | Episodes | Graph Signal | Agent Behavior |
|---|---|---|---|
| 1 — Exploration | 0–100 | Reward ~−150, short duration (~100 steps) | Random crashes, no learned strategy |
| 2 — Survival | 100–300 | Duration rising to ~850 steps, reward climbing slowly | Discovers hovering to avoid crashes; overcautious |
| 3 — Learning to Land | 300–450 | Duration drops sharply, reward crosses 200 | Commits to descent; abandons hovering |
| 4 — Refinement | 450–700 | Reward stable 220–250, duration declining to ~370 | Efficient, repeatable landings |

The most diagnostic signal is the **Era 2→3 transition**: the duration peak (~850 steps at episode 300) followed by a sharp decline coincides exactly with the reward curve accelerating upward — confirming the agent switched from hovering to active landing.

---

## Evaluation Summary

The greedy policy was evaluated over 100 episodes:

- **Reward distribution**: heavily clustered in 220–300, mean = 242.6. A small left tail (~5 episodes below 100) reflects difficulty with extreme random starting conditions.
- **Duration distribution**: most episodes complete in 150–350 steps. ~11 episodes hit the 1000-step cap, indicating the Era 2 hovering behavior occasionally re-emerges on outlier starting states.
- **Overall**: ~80–85% of episodes exceed the 200 threshold; the policy is largely consistent but not fully robust to edge-case initializations.

---

## File Structure

```
.
├── hw4.py                  # Main training, evaluation, and plotting script
├── dqn_lunar_lander.pth    # Saved model weights (generated after training)
├── training_curves.png     # Reward and duration training curves
├── eval_histograms.png     # Evaluation reward and duration histograms
└── videos/                 # Recorded gameplay videos (every 25 episodes)
```

---

## Requirements

```bash
pip install gymnasium[box2d] torch numpy matplotlib
```

> Box2D is required for the LunarLander environment. On some systems you may also need `swig`: `pip install swig` before installing `gymnasium[box2d]`.

---

## Usage

### Train from scratch

```bash
python hw4.py
```

This will:
1. Train for 700 episodes with video recording every 25 episodes
2. Save the model to `dqn_lunar_lander.pth`
3. Plot and save training curves to `training_curves.png`
4. Run 100-episode greedy evaluation and save histograms to `eval_histograms.png`

### Load and evaluate a saved model

```python
import torch, gymnasium as gym
from hw4 import DuelingDQN, DQNAgent, evaluate

env = gym.make("LunarLander-v3")
state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
env.close()

agent = DQNAgent(state_dim, action_dim)
agent.policy_net.load_state_dict(torch.load("dqn_lunar_lander.pth"))
evaluate(agent)
```

---

## Device

Training automatically uses CUDA if available, otherwise falls back to CPU.

```
Using device: cuda   # or cpu
```
