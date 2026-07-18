import numpy as np
from .gridworld import N_STATES, N_ACTIONS, GOAL, GAMMA, step


def choose_action(Q, s, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)
    else:
        return np.argmax(Q[s])


def sarsa(n_episodes=10000, alpha=0.1):
    """
    On-policy TD control. Aprende Q(s,a) usando la accion
    que realmente tomara en el siguiente estado (a').
    """
    Q = np.zeros((N_STATES, N_ACTIONS))
    epsilon = 1.0

    for episode in range(n_episodes):
        s = 0
        a = choose_action(Q, s, epsilon)

        while s != GOAL:
            s_next, r = step(s, a)
            if s_next == GOAL:
                Q[s, a] += alpha * (r - Q[s, a])
                break
            else:
                a_next = choose_action(Q, s_next, epsilon)
                Q[s, a] += alpha * (r + GAMMA * Q[s_next, a_next] - Q[s, a])
                s = s_next
                a = a_next

        epsilon = max(0.01, epsilon * 0.999)

    policy = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    return V, policy