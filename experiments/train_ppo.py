import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from algorithms.ppo import PPO


def train(total_timesteps=800000):
    env = gym.make('LunarLanderContinuous-v3')
    model = PPO(env)
    model.learn(total_timesteps)
    env.close()


if __name__ == '__main__':
    train()