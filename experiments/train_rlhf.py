import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from algorithms.rlhf import RLHFTrainer


def train(n_epochs=5):
    trainer = RLHFTrainer()
    rewards_history = trainer.train(n_epochs=n_epochs)

    # grafico
    plt.figure(figsize=(12, 5))
    plt.plot(rewards_history, alpha=0.4, label='Reward por batch')
    if len(rewards_history) >= 20:
        moving_avg = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
        plt.plot(range(19, len(rewards_history)), moving_avg,
                 color='red', linewidth=2, label='Media movil (20 batches)')
    plt.xlabel('Batch')
    plt.ylabel('Reward promedio')
    plt.title('RLHF con PPO - GPT2')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/rlhf_rewards.png')
    plt.show()

    # evaluacion cualitativa
    prompts = [
        "This movie was",
        "The acting in this film",
        "I watched this movie and",
    ]
    trainer.evaluate(prompts)

    return rewards_history


if __name__ == '__main__':
    train()