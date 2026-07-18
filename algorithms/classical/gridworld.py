import numpy as np

ROWS, COLS = 4, 4
N_STATES = ROWS * COLS
N_ACTIONS = 4

GOAL = 15
OBSTACLE = 5

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

DELTAS = {
    UP: (-1, 0),
    DOWN: (+1, 0),
    LEFT: (0, -1),
    RIGHT: (0, +1)
}

R_GOAL = +1.0
R_STEP = -0.04
GAMMA = 0.99
THETA = 1e-6

def to_coords(s):
    return s // COLS, s % COLS


def to_state(r, c):
    return r * COLS + c


def is_valid(r, c):
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return False
    if to_state(r, c) == OBSTACLE:
        return False
    return True


def step(s, a):
    if s == GOAL:
        return s, 0.0
    r, c = to_coords(s)
    dr, dc = DELTAS[a]
    nr, nc = r + dr, c + dc
    if is_valid(nr, nc):
        s_next = to_state(nr, nc)
    else:
        s_next = s
    reward = R_GOAL if s_next == GOAL else R_STEP
    return s_next, reward