import numpy as np
from .gridworld import (N_STATES, N_ACTIONS, OBSTACLE, GOAL,
                        GAMMA, step)


def choose_action(Q, s, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)
    else:
        return np.argmax(Q[s])


def montecarlo(n_episodes=10000):
    """
    Aprende la política óptima mediante episodios completos.
    No necesita conocer el modelo del entorno, solo interactuar con él.
    """
    Q = np.zeros((N_STATES, N_ACTIONS))
    Returns = np.zeros((N_STATES, N_ACTIONS))
    Count = np.zeros((N_STATES, N_ACTIONS))
    epsilon = 1.0

    for episode in range(n_episodes):
        secuencia = []
        s = 0

        while True:
            if s == GOAL:
                G = 0
                for s_ep, a, r in reversed(secuencia):
                    G = GAMMA * G + r
                    Returns[s_ep, a] += G
                    Count[s_ep, a] += 1
                    Q[s_ep, a] = Returns[s_ep, a] / Count[s_ep, a]
                epsilon = max(0.01, epsilon * 0.999)
                break

            a = choose_action(Q, s, epsilon)
            s_next, r = step(s, a)
            secuencia.append((s, a, r))
            s = s_next

    policy = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    return V, policy