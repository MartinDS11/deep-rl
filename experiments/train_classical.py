import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.classical import (value_iteration, montecarlo,
                                   sarsa, q_learning,
                                   to_coords, ROWS, COLS, GOAL, OBSTACLE)
from algorithms.classical.gridworld import N_STATES, UP, DOWN, LEFT, RIGHT

ARROWS = {UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→'}


def plot(V, policy, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    V_grid = V.reshape(ROWS, COLS).copy().astype(float)
    V_grid[to_coords(OBSTACLE)] = np.nan
    ax.imshow(V_grid, cmap='RdYlGn', origin='upper')

    for s in range(N_STATES):
        r, c = to_coords(s)
        if s == OBSTACLE:
            ax.text(c, r, 'X', ha='center', va='center', fontsize=18, color='gray')
        elif s == GOAL:
            ax.text(c, r, 'G', ha='center', va='center', fontsize=18,
                    color='darkgreen', fontweight='bold')
        else:
            ax.text(c, r, f"{ARROWS[policy[s]]}\n{V[s]:.3f}",
                    ha='center', va='center', fontsize=11)

    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=2)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"results/{title.lower().replace(' ', '_')}.png")
    plt.show()


if __name__ == '__main__':
    print("=== Value Iteration ===")
    V_vi, policy_vi = value_iteration()
    plot(V_vi, policy_vi, 'Value Iteration')

    print("\n=== Monte Carlo ===")
    V_mc, policy_mc = montecarlo(n_episodes=10000)
    plot(V_mc, policy_mc, 'Montecarlo')

    print("\n=== SARSA ===")
    V_sarsa, policy_sarsa = sarsa(n_episodes=10000)
    plot(V_sarsa, policy_sarsa, 'SARSA')

    print("\n=== Q-Learning ===")
    V_ql, policy_ql = q_learning(n_episodes=10000)
    plot(V_ql, policy_ql, 'Q-Learning')