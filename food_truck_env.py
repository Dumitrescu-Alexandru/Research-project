import gym
from gym import spaces
from gym.utils import seeding
import torch
import numpy as np
from itertools import product

WALL = -1
GRID = 1
DONUT_NORTH_FIRST = 2
DONUT_NORTH_SECOND = 22

DONUT_SOUTH_FIRST = 3
DONUT_SOUTH_SECOND = 33

VEGAN_FIRST = 4
VEGAN_SECOND = 44

NOODLE_FIRST = 5
NOODLE_SECOND = 55

rwds = {(8, 1): 10, (3, 3): 10, (1, 5): -10, (6, 6): 0}
delayed_rwds = {(8, 1): -10, (3, 3): -10, (1, 5): 20, (6, 6): 0}
time_cost = -0.01

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
MAPS = {
    "original": torch.tensor([[WALL, WALL, WALL, WALL, VEGAN_FIRST, WALL],
                              [WALL, WALL, WALL, GRID, GRID, GRID],
                              [WALL, WALL, DONUT_NORTH_FIRST, GRID, WALL, GRID],
                              [WALL, WALL, WALL, GRID, WALL, GRID],
                              [WALL, WALL, WALL, GRID, GRID, GRID],
                              [WALL, WALL, WALL, GRID, WALL, NOODLE_FIRST],
                              [GRID, GRID, GRID, GRID, WALL, WALL],
                              [DONUT_SOUTH_FIRST, WALL, WALL, GRID, WALL, WALL]])}

REWARDS = {"original": ((10, -10), (10, -10), (-10, 20), (0, 0)),
           "version_1": ((11, -10), (11, -10), (-10, 20), (0, 0))}


class FoodTruck(gym.Env):
    def __init__(self):
        # original states are in 1...8; 1...6
        self.ft_map = -torch.ones(10, 8)
        self.ft_map[1:9, 1:7] = MAPS["original"]
        self.position = np.array([8, 4])
        self.all_actions = [LEFT, RIGHT, UP, DOWN]

    def step(self, action):
        pos_change = self.get_change(action)
        x, y = pos_change + self.position
        if self.ft_map[x, y] == WALL:
            print("Tried to hit a wall. Position did not change...")
        else:
            self.position = self.position + pos_change
        x, y = self.position
        if self.ft_map[x, y] != GRID:
            # if after stepping I end up at a restaurant, return True - the episode ended; else, return False
            return True
        return False

    def reset(self):
        self.position = (8, 4)

    def possible_actions(self):
        possible_actions = []
        for a in self.all_actions:
            x, y = self.position + self.get_change(a)
            if self.ft_map[x, y] != WALL:
                possible_actions.append(a)

    def get_reward(self, state):
        if rwds[tuple(state)]:
            return time_cost + rwds[tuple(state)]
        else:
            return time_cost

    @staticmethod
    def get_change(action):
        if action == LEFT:
            index_change = np.array([0, -1])
        elif action == DOWN:
            index_change = np.array([1, 0])
        elif action == UP:
            index_change = np.array([-1, 0])
        else:
            index_change = np.array([0, 1])
        return index_change
