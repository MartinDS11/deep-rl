# Deep Reinforcement Learning Portfolio

Implementation of classical and deep reinforcement learning algorithms from scratch in PyTorch, progressing from tabular methods to modern policy optimization and LLM fine-tuning with RLHF.

---

## Algorithms

### Classical RL (Tabular)
| Algorithm | Environment | On/Off Policy | Convergence |
|-----------|-------------|---------------|-------------|
| Value Iteration | Gridworld 4×4 | — | Guaranteed (DP) |
| Monte Carlo | Gridworld 4×4 | On-policy | Asymptotic |
| SARSA | Gridworld 4×4 | On-policy | Guaranteed* |
| Q-Learning | Gridworld 4×4 | Off-policy | Guaranteed* |

*Under standard conditions: all state-action pairs visited infinitely often, learning rate satisfying Robbins-Monro conditions.

### Deep RL
| Algorithm | Environment | On/Off Policy | Action Space |
|-----------|-------------|---------------|--------------|
| DQN | CartPole-v1 | Off-policy | Discrete |
| REINFORCE | CartPole-v1 | On-policy | Discrete |
| SAC | Pendulum-v1 | Off-policy | Continuous |
| PPO | LunarLanderContinuous-v3 | On-policy | Continuous |

### LLM Fine-tuning
| Method | Model | Task |
|--------|-------|------|
| RLHF (PPO) | GPT-2 | Positive sentiment generation |

---

## Environments

### Gridworld 4×4
- **States**: 16 discrete states (cells), 1 obstacle (state 5), 1 goal (state 15)
- **Actions**: 4 discrete (UP, DOWN, LEFT, RIGHT)
- **Reward**: +1.0 on reaching goal, −0.04 per step
- **Discount**: γ = 0.99

### CartPole-v1
- **States**: 4 continuous values (cart position, cart velocity, pole angle, pole angular velocity)
- **Actions**: 2 discrete (push left, push right)
- **Reward**: +1 per timestep the pole remains upright
- **Episode termination**: pole angle > ±12°, cart position > ±2.4, or 500 steps reached
- **Solved**: average reward ≥ 475 over 100 consecutive episodes

### Pendulum-v1
- **States**: 3 continuous values (cos θ, sin θ, angular velocity)
- **Actions**: 1 continuous torque in [−2, 2]
- **Reward**: −(θ² + 0.1·θ̇² + 0.001·u²) per step, maximum 0
- **Episode length**: fixed 200 steps

### LunarLanderContinuous-v3
- **States**: 8 continuous values (position x/y, velocity x/y, angle, angular velocity, leg contacts)
- **Actions**: 2 continuous values (main engine throttle, lateral engine throttle)
- **Reward**: +100/−100 for landing/crashing, −0.3 per frame main engine firing
- **Solved**: average reward ≥ 200 over 100 consecutive episodes

---

## Algorithm Details

### Value Iteration
Model-based dynamic programming. Solves the Bellman optimality equation iteratively:

$$V_{k+1}(s) = \max_a \sum_{s'} p(s'|s,a)\left[r(s,a,s') + \gamma V_k(s')\right]$$

Converges to $V^*$ when $\|V_{k+1} - V_k\|_\infty < \theta$. Requires full knowledge of the transition model $p(s'|s,a)$. Complexity $O(|S|^2|A|)$ per iteration.

### Monte Carlo
Model-free, episodic. Estimates $Q^\pi(s,a)$ by averaging observed returns:

$$Q(s,a) \leftarrow \frac{1}{N(s,a)} \sum_{i=1}^{N(s,a)} G_t^{(i)}$$

where $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$. Uses ε-greedy policy improvement. Requires episodic tasks (episodes must terminate). Converges asymptotically as $N(s,a) \to \infty$.

### SARSA (On-policy TD)
Model-free, bootstrapped. Updates Q using the action actually taken:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma Q(s',a') - Q(s,a)\right]$$

where $a' \sim \pi(\cdot|s')$. On-policy: learns the value of the policy being followed. Converges to $Q^\pi$ (the current policy's Q-function) under Robbins-Monro conditions on $\alpha$.

### Q-Learning (Off-policy TD)
Model-free, bootstrapped. Updates Q using the greedy action:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$$

Off-policy: learns $Q^*$ regardless of the exploration policy used. Converges to $Q^*$ under the same conditions as SARSA. Key difference from SARSA: uses $\max_{a'} Q(s',a')$ instead of $Q(s',a')$.

### DQN (Deep Q-Network)
Extends Q-Learning to continuous state spaces using a neural network $Q_\theta(s,a) \approx Q^*(s,a)$. Two key stabilization techniques:

- **Experience Replay**: stores transitions $(s,a,r,s')$ in buffer $\mathcal{D}$, samples random minibatches to break correlation
- **Target Network**: separate frozen network $Q_{\theta^-}$ for computing targets, updated every $C$ steps

Loss function:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right]$$

No theoretical convergence guarantee with function approximation (deadly triad).

### REINFORCE (Policy Gradient)
Directly optimizes the policy $\pi_\theta$ by gradient ascent on $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$. By the Policy Gradient Theorem:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

Monte Carlo estimates $G_t$ from full episodes. On-policy: data must come from the current policy. High variance estimator; returns are normalized to reduce variance.

### SAC (Soft Actor-Critic)
Off-policy Actor-Critic for continuous actions. Maximizes a maximum entropy objective:

$$J(\pi) = \mathbb{E}\left[\sum_t \gamma^t \left(r_t + \alpha H(\pi(\cdot|s_t))\right)\right]$$

The soft Bellman equation:

$$Q^\pi(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}\left[\mathbb{E}_{a' \sim \pi}[Q^\pi(s',a')] + \alpha H(\pi(\cdot|s'))\right]$$

The optimal policy is:

$$\pi^*(a|s) \propto \exp\left(\frac{1}{\alpha} Q^*(s,a)\right)$$

Uses **Clipped Double-Q** to reduce overestimation bias, **reparameterization trick** for differentiable sampling, and **soft target network updates**: $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$.

### PPO (Proximal Policy Optimization)
On-policy Actor-Critic. Constrains policy updates to avoid destructive large steps. The clipped surrogate objective:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio. Uses **Generalized Advantage Estimation (GAE)**:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

KL divergence is monitored as an early stopping criterion. No guaranteed convergence but empirically stable.

### RLHF (Reinforcement Learning from Human Feedback)
Three-stage pipeline for aligning LLMs:

1. **SFT**: supervised fine-tuning on demonstration data
2. **Reward Model**: trained on human preference pairs $(y_w \succ y_l)$ to estimate $r_\phi(x,y)$
3. **PPO**: optimizes the LM policy with a KL penalty:

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}[r_\phi(x,y)] - \beta D_{KL}[\pi(\cdot|x) \| \pi_{ref}(\cdot|x)]$$

The KL term prevents reward hacking by keeping the policy close to the reference model. Implemented using TRL with GPT-2 and a sentiment classifier as proxy reward model.

---

## Results

### DQN - CartPole-v1
![DQN](results/dqn_rewards.png)

### REINFORCE - CartPole-v1
![REINFORCE](results/reinforce_rewards.png)

### SAC - Pendulum-v1
![SAC](results/sac_rewards.png)

### PPO - LunarLanderContinuous-v3
![PPO](results/ppo_rewards.png)

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy >= 1.22.0
- Matplotlib >= 3.5.0

For RLHF only:
- Transformers >= 4.30.0
- TRL >= 0.7.0
- Datasets >= 2.0.0
- Accelerate >= 0.20.0

---

## Installation

```bash
git clone https://github.com/MartinDS11/deep-rl.git
cd deep-rl
pip install -r requirements.txt
```

---

## Usage

### Run experiments

```bash
# Classical RL
python experiments/train_classical.py

# Deep RL
python experiments/train_dqn.py
python experiments/train_reinforce.py
python experiments/train_sac.py
python experiments/train_ppo.py

# RLHF (requires GPU, recommended: Google Colab)
python experiments/train_rlhf.py
```

### Import as library

```python
from algorithms.dqn import DQNAgent
from algorithms.reinforce import REINFORCEAgent
from algorithms.sac import SACAgent
from algorithms.ppo import PPO
from algorithms.classical import montecarlo, value_iteration, sarsa, q_learning

import gymnasium as gym

# DQN example
env = gym.make('CartPole-v1')
agent = DQNAgent(state_dim=4, action_dim=2)

# SAC example
env = gym.make('Pendulum-v1')
agent = SACAgent(state_dim=3, action_dim=1, action_limit=2.0)
```

---

## Project Structure
deep-rl/
├── algorithms/
│   ├── classical/
│   │   ├── gridworld.py        # shared environment
│   │   ├── value_iteration.py
│   │   ├── monte_carlo.py
│   │   ├── sarsa.py
│   │   └── q_learning.py
│   ├── dqn/
│   │   ├── agent.py
│   │   ├── network.py
│   │   └── replay_buffer.py
│   ├── reinforce/
│   │   └── agent.py
│   ├── sac/
│   │   ├── agent.py
│   │   └── networks.py
│   ├── ppo/
│   │   ├── ppo.py
│   │   └── network.py
│   └── rlhf/
│       └── trainer.py
├── experiments/
│   ├── train_classical.py
│   ├── train_dqn.py
│   ├── train_reinforce.py
│   ├── train_sac.py
│   ├── train_ppo.py
│   └── train_rlhf.py
├── results/
├── requirements.txt
├── setup.py
└── README.md
---

## References

- **DQN**: Mnih et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). Nature.
- **REINFORCE**: Williams (1992). [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696). Machine Learning.
- **SAC**: Haarnoja et al. (2018). [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). ICML.
- **PPO**: Schulman et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv.
- **RLHF**: Ouyang et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). NeurIPS.
- **GAE**: Schulman et al. (2015). [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). ICLR.
- **Sutton & Barto**: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html). MIT Press.