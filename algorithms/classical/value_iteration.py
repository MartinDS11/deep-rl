import numpy as np
from .gridworld import (N_STATES, N_ACTIONS, OBSTACLE, GOAL,
                        GAMMA, THETA, UP, DOWN, LEFT, RIGHT, step)


def value_iteration():
    """
    Encuentra la política óptima resolviendo exactamente
    la ecuación de Bellman de forma iterativa.
    No necesita interactuar con el entorno: usa el modelo completo.
    """
    V = np.zeros(N_STATES)
    iteration = 0

    while True:
        delta = 0.0
        V_new = V.copy()
        for s in range(N_STATES):
            if s in (OBSTACLE, GOAL):
                continue
            q_values = []
            for a in (UP, DOWN, LEFT, RIGHT):
                s_next, reward = step(s, a)
                q_values.append(reward + GAMMA * V[s_next])
            V_new[s] = max(q_values)
            delta = max(delta, np.abs(V_new[s] - V[s]))
        V = V_new
        iteration += 1

        if delta < THETA:
            print(f"Convergencia en iteracion {iteration}")
            break

    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        if s in (OBSTACLE, GOAL):
            continue
        q_values = []
        for a in (UP, DOWN, LEFT, RIGHT):
            s_next, reward = step(s, a)
            q_values.append(reward + GAMMA * V[s_next])
        policy[s] = np.argmax(q_values)

    return V, policy