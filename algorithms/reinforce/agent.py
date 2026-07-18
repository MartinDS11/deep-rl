import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, training=True):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        if training:
            self.log_probs.append(dist.log_prob(action).squeeze())
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def compute_returns(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        return returns_t

    def update(self):
        returns_t = self.compute_returns()
        log_probs_t = torch.stack(self.log_probs)
        loss = -(log_probs_t * returns_t).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs = []
        self.rewards = []
        return loss.item()