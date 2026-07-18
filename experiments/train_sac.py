import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from algorithms.sac import SACAgent


def train(n_episodes=300):
    env = gym.make('Pendulum-v1')
    agent = SACAgent(state_dim=3, action_dim=1, action_limit=2.0)

    rewards_history = []

    # llenamos el buffer con experiencias aleatorias
    print("Llenando replay buffer...")
    state, _ = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        if done:
            state, _ = env.reset()

    print("Entrenando...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.update()
            state = next_state
            total_reward += reward
            if done:
                break

        rewards_history.append(total_reward)

        if episode % 10 == 0:
            avg = np.mean(rewards_history[-10:])
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
    plt.title('SAC - Pendulum-v1')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/sac_rewards.png')
    plt.show()

    return agent, rewards_history


if __name__ == '__main__':
    train()