import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from algorithms.reinforce import REINFORCEAgent


def train(n_episodes=1000):
    env = gym.make('CartPole-v1')
    agent = REINFORCEAgent(state_dim=4, action_dim=2)

    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.update()
        rewards_history.append(total_reward)

        if episode % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"Episodio {episode}, reward promedio: {avg:.1f}")

    env.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.4, label='Reward por episodio')
    if len(rewards_history) >= 20:
        moving_avg = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
        plt.plot(range(19, len(rewards_history)), moving_avg,
                 color='red', linewidth=2, label='Media movil (20 ep)')
    plt.xlabel('Episodio')
    plt.ylabel('Reward')
    plt.title('REINFORCE - CartPole-v1')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/reinforce_rewards.png')
    plt.show()

    return agent, rewards_history


if __name__ == '__main__':
    train()