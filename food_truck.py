import gym
from gym import spaces
from gym.utils import seeding
import torch
from itertools import product



LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

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

TERMINAL = -2

MAPS = {
    "original": torch.tensor([[WALL, WALL, WALL, WALL, VEGAN_FIRST, WALL],
[WALL, WALL, WALL, GRID, GRID, GRID],
[WALL, WALL, DONUT_NORTH_FIRST,GRID,WALL,GRID],
[WALL,WALL,WALL,GRID,WALL,GRID],
[WALL, WALL, WALL, GRID, GRID, GRID],
[WALL, WALL, WALL, GRID, WALL, NOODLE_FIRST],
[GRID, GRID, GRID, GRID, WALL, WALL],
[DONUT_SOUTH_FIRST, WALL, WALL, GRID, WALL, WALL]])}

REWARDS = {"original": ((10,-10), (10,-10), (-10,20), (0,0)), "version_1": ((11,-10), (11,-10), (-10,20), (0,0))}


class FoodTruck(gym.Env):

    def __init__(self, map_matrix = MAPS["original"], rewards=REWARDS["original"], time_cost = -0.01):
        self.n_row = map_matrix.shape[0]
        self.n_col = map_matrix.shape[1]
        self.grid_map = map_matrix
        self.states = self.grid_map.reshape(-1)
        self.rewards = rewards
        self.time_cost = time_cost
        self.state = 39  # Start at beginning of the chain
        self.seed()

        #States with feature >1 are restaurants.
        self.n_vegan = torch.sum(self.states == 4).item()
        self.n_donut_south = torch.sum(self.states == 3).item()
        self.n_donut_north = torch.sum(self.states == 2).item()
        self.n_noodle = torch.sum(self.states == 5).item()

        #Add the pre-terminal states per restaurant. and a terminal state
        self.extended_states = torch.cat((self.states,torch.tensor([22,33,44,55,-2])))

        self.n_states_extended = self.extended_states.shape[0]
        self.n_states = self.n_row * self.n_col
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states_extended)
        self.R = torch.zeros((self.n_states_extended,self.n_actions,self.n_states_extended)) + self.time_cost

        self.P = torch.zeros((self.n_states_extended, self.n_actions, self.n_states_extended))

        for s_row,s_col, a in product(range(self.n_row),range(self.n_col), range(self.n_actions)):
            s_type = self.grid_map[s_row,s_col]
            p_index = (s_row * self.n_col) + s_col
            #Wall states are inaccessible anyays
            if s_type == WALL:
                continue

            #First entry to restaurants, whatever you do, will go to the second-time state of the same restaurant
            #Trick for handling the "delayed reward" in the example
            elif s_type == DONUT_NORTH_FIRST:
                self.P[p_index,:,-5] = 1.0
                self.R[p_index,:,:] = self.rewards[0][0]
            elif s_type == DONUT_SOUTH_FIRST:
                self.P[p_index,:,-4] = 1.0
                self.R[p_index,:,:] = self.rewards[1][0]

            elif s_type == VEGAN_FIRST:
                self.P[p_index,:,-3] = 1.0
                self.R[p_index,:,:] = self.rewards[2][0]

            elif s_type == NOODLE_FIRST:
                self.P[p_index,:,-2] = 1.0
                self.R[p_index,:,:] = self.rewards[3][0]

            #If it is an accessible grid
            elif s_type == GRID:
                if (s_col - 1) >= 0 and self.grid_map[s_row,s_col-1] != WALL:
                    self.P[p_index,LEFT,p_index-1] = 1.0
                else:
                    self.P[p_index,LEFT,p_index] = 1.0

                if (s_col + 1) < self.n_col and self.grid_map[s_row,s_col+1] != WALL:
                    self.P[p_index,RIGHT,p_index+1] = 1.0
                else:
                    self.P[p_index,RIGHT,p_index] = 1.0

                if (s_row - 1) >= 0 and self.grid_map[s_row-1,s_col] != WALL:
                    self.P[p_index,UP,p_index-self.n_col] = 1.0
                else:
                    self.P[p_index,UP,p_index] = 1.0

                if (s_row + 1) < self.n_row and self.grid_map[s_row+1,s_col] != WALL:
                    self.P[p_index,DOWN,p_index+self.n_col] = 1.0
                else:
                    self.P[p_index,DOWN,p_index] = 1.0

        #Another hack. Terminal states. No escape.

        self.P[-5, :, -1] = 1.0
        self.R[-5,:,:] = self.rewards[0][1]

        self.P[-4, :, -1] = 1.0
        self.R[-4,:,:] = self.rewards[1][1]

        self.P[-3, :, -1] = 1.0
        self.R[-3,:,:] = self.rewards[2][1]

        self.P[-2, :, -1] = 1.0
        self.R[-2,:,:] = self.rewards[3][1]

        self.P[-1, :, -1] = 1.0
        self.R[-1,:,:] = 0



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        reward = self.R[self.state,0,0]
        done = False
        if self.extended_states[self.state] == TERMINAL:
            done = True
            self.state = torch.argmax(self.P[self.state, action, :]).item()

        else:
            self.state = torch.argmax(self.P[self.state, action, :]).item()

        return self.state, reward, done, {}




    def reset(self):
        self.state = 39
        return self.state