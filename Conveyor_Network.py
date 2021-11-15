# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 12:19:42 2021

@author: busach
"""

import gym
from gym import spaces
from gym import error
from gym import utils
from gym.utils import seeding
import numpy as np
import logging
import sys

sys.path.append('C:\source\Research Project\Scripts\Research-Project\snakes-master')
from n_trial.updated_trial_network import TrialConveyorNetwork
from Conveying_Network import ConveyorNetwork
from snakes.utils.simul import StateSpace

logger = logging.getLogger(__name__)

JOBS_TRIAL = [1, 2, 3]
JOBS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
RESOURCES = ['Red', 'Green', 'Blue', 'Violet']
PLACES_TRIAL = ['S', 'S1', 'N1', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'N2', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3',
                'E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'J1', 'J2', 'J3', 'G1', 'G2', 'G3', 'K1', 'K2', 'K3', 'T1',
                'W1', 'Red', 'Green', 'N3', 'N4', 'N0', 'N6', 'N9']
PLACES = ['S', 'S1', 'N1', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'N2', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3',
                'E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'J1', 'J2', 'J3', 'G1', 'G2', 'G3', 'K1', 'K2', 'K3', 'T1',
                'W1', 'Red', 'Green', 'N3', 'N4', 'N0', 'N6', 'N9']
TRANSITION_TRIAL = ['s1', 'SN1', 'AN1', 'BN1', 'P_A1', 'P_A2', 'P_A3', 'N_A1', 'N_A2', 'N_A3', 'P_B1', 'P_B2', 'P_B3',
                    'N_B1', 'N_B2', 'N_B3', 'BN9', 'JN9', 'KN9', 'P_K1', 'P_K2', 'P_K3', 'N_K1', 'N_K2', 'N_K3', 'KN0',
                    't1', 'T', 'P_J1', 'P_J2', 'P_J3', 'N_J1', 'N_J2', 'N_J3', 'JN6', 'GN6', 'P_G1', 'P_G2', 'P_G3',
                    'N_G1', 'N_G2', 'N_G3', 'GN3', 'FN3', 'EN3', 'P_E1', 'P_E2', 'P_E3', 'N_E1', 'N_E2', 'N_E3', 'AN2',
                    'EN2', 'CN2', 'P_C1', 'P_C2', 'P_C3', 'N_C1', 'N_C2', 'N_C3', 'C0W1', 'C1W1', 'C2W1', 'C3W1', 'D0W1',
                    'D1W1', 'D2W1', 'D3W1', 'P_D1', 'P_D2', 'P_D3', 'N_D1', 'N_D2', 'N_D3', 'DN4', 'FN4', 'P_F1', 'P_F2',
                    'P_F3', 'N_F1', 'N_F2', 'N_F3']
TRANSITION = ['s1', 'SN1', 'AN1', 'BN1', 'P_A1', 'P_A2', 'P_A3', 'N_A1', 'N_A2', 'N_A3', 'P_B1', 'P_B2', 'P_B3',
                    'N_B1', 'N_B2', 'N_B3', 'BN9', 'JN9', 'KN9', 'P_K1', 'P_K2', 'P_K3', 'N_K1', 'N_K2', 'N_K3', 'KN0',
                    't1', 'T', 'P_J1', 'P_J2', 'P_J3', 'N_J1', 'N_J2', 'N_J3', 'JN6', 'GN6', 'P_G1', 'P_G2', 'P_G3',
                    'N_G1', 'N_G2', 'N_G3', 'GN3', 'FN3', 'EN3', 'P_E1', 'P_E2', 'P_E3', 'N_E1', 'N_E2', 'N_E3', 'AN2',
                    'EN2', 'CN2', 'P_C1', 'P_C2', 'P_C3', 'N_C1', 'N_C2', 'N_C3', 'C0W1', 'C1W1', 'C2W1', 'C3W1', 'D0W1',
                    'D1W1', 'D2W1', 'D3W1', 'P_D1', 'P_D2', 'P_D3', 'N_D1', 'N_D2', 'N_D3', 'DN4', 'FN4', 'P_F1', 'P_F2',
                    'P_F3', 'N_F1', 'N_F2', 'N_F3']


def generate_random_orders(version, seed):
    """ Generates the random orders for the conveying network"""
    np.random.seed(seed)
    init = 0
    quantity = []
    orders = {}
    if version == 'trial':
        size = np.random.choice(JOBS_TRIAL)
        jobs = np.random.randint(1, 4, size, dtype=np.int16)
        red = np.random.randint(100, 5000, 1, dtype=np.int16)
        green = np.random.randint(100, 5000, 1, dtype=np.int16)
        for i in range(len(jobs)):
            quantity[i] = np.random.randint(10, 500, 1, dtype=np.int16)
            orders[f"job_{jobs[i]}"] = quantity[i]
            init += quantity[i]
        resources = [init, red, green]

        return jobs, resources, quantity, orders
    else:
        size = np.random.choice(JOBS)
        jobs = np.random.randint(1, 16, size, dtype=np.int16)
        red = np.random.randint(100, 5000, 1, dtype=np.int16)
        green = np.random.randint(100, 5000, 1, dtype=np.int16)
        blue = np.random.randint(100, 5000, 1, dtype=np.int16)
        violet = np.random.randint(100, 5000, 1, dtype=np.int16)
        for i in range(len(jobs)):
            quantity[i] = np.random.randint(10, 500, 1, dtype=np.int16)
            orders[f"job_{jobs[i]}"] = quantity[i]
            init += quantity[i]
        resources = [init, red, green, blue, violet]

        return jobs, resources, quantity, orders


def random_resource_generator(seed):
    np.random.seed(seed)
    res_size = np.random.randint(100, 5000, 1, dtype=np.int16)

    return res_size


class ConveyorEnv(gym.Env):
    metadata = {'render.modes': ['Human']}

    def __init__(self, version='trail', final_reward = 10):
        self.current_step = 0
        self.version = version
        self.final_reward = final_reward
        self.seed = seeding.create_seed()
        self.jobs, self.res, self.quantity, self.orders = generate_random_orders(self.version, self.seed)
        if self.version == 'trial':
            self.network = TrialConveyorNetwork(self.jobs, self.res, self.quantity)
            self.net, self.trans = self.network.trial_conveyor_petrinet()
            self.no_places = 40
            self.no_trans = len(self.trans)
        else:
            self.network = ConveyorNetwork(self.jobs, self.res, self.quantity)
            self.net, self.trans = self.network.trial_conveyor_petrinet()
            self.no_places = 67
            self.no_trans = len(self.trans)

        self._stateSpace = StateSpace(self.net)
        self.marking = self._stateSpace.get()
        self.reward_range = [-1, 1, 2]
        # Observation space represents the places, and there are
        if self.version == 'trial':
            self.observation_space = spaces.Dict({i: spaces.Discrete(2, seed = self.seed) for i in PLACES_TRIAL})
        else:
            self.observation_space = spaces.Dict({i: spaces.Discrete(2, seed=self.seed) for i in PLACES})
        # Action space represents the transitions to get fired
        if self.version == 'trial':
            self.action_space = spaces.Dict({i: spaces.Discrete(2, seed = self.seed) for i in TRANSITION_TRIAL})
        else:
            self.action_space = spaces.Dict({i: spaces.Discrete(2, seed=self.seed) for i in TRANSITION})

    def reset(self):
        self.seed = seeding.create_seed()
        np.random.seed(self.seed)
        if self.version == 'trial':
            self.network = TrialConveyorNetwork(self.jobs, self.res, self.quantity)
            self.net, self.trans = self.network.trial_conveyor_petrinet()
        else:
            self.network = ConveyorNetwork(self.jobs, self.res, self.quantity)
            self.net, self.trans = self.network.trial_conveyor_petrinet()
        self._stateSpace = StateSpace(self.net)
        self.current_step = 0

        return self._next_observation()

    def _next_observation(self):
        self.marking_places = list(self._stateSpace.get().keys())
        for i in range(len(PLACES)):
            if PLACES_TRIAL[i] in self.marking_places:
                self.observation_space.__setitem__(PLACES_TRIAL[i], 1)
            else:
                self.observation_space.__setitem__(PLACES_TRIAL[i], 0)

        return list(self.observation_space.values())

    def step(self, action):
        # check for the resources, if not present add resources randomly
        self.marking = self._stateSpace.get()

        if self.version == 'trial':
            if 'Red' not in self.marking:
                r_size = random_resource_generator(seed = self.seed)
                r = [1 for _ in range(r_size)]
                self.net.add_marking(Marking(Red=MultiSet(r)))
                self._stateSpace.current = self._stateSpace.add(self.net.get_marking())
            elif 'Green' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                g = [2 for _ in range(r_size)]
                self.net.add_marking(Marking(Green=MultiSet(g)))
                self._stateSpace.current = self._stateSpace.add(self.net.get_marking())
        else:
            if 'Red' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                r = [1 for _ in range(r_size)]
                self.net.add_marking(Marking(Red=MultiSet(r)))
                self._stateSpace.current = self._stateSpace.add(self.net.get_marking())
            elif 'Green' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                g = [2 for _ in range(r_size)]
                self.net.add_marking(Marking(Green=MultiSet(g)))
                self._stateSpace.current = self._stateSpace.add(self.net.get_marking())
            elif 'Blue' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                b = [4 for _ in range(r_size)]
                self.net.add_marking(Marking(Blue=MultiSet(b)))
                self._stateSpace.current = self._stateSpace.add(self.net.get_marking())
            elif 'Violet' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                v = [8 for _ in range(r_size)]
                self.net.add_marking(Marking(Violet=MultiSet(v)))
                self._stateSpace.current = self._stateSpace.add(self.net.get_marking())

        # Execute 1 time step within the environment
        self._take_action(action)
        self.current_step += 1

        reward = self._calculate_reward()
        done = self._done_status()
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        pass

    def _calculate_reward(self):
        self.reward_marking_keys = list(self._stateSpace.get().keys())
        self.reward_marking = dict(self._stateSpace.get())
        self.available_tokens = 0
        for i in self.reward_marking_keys:
            if i in RESOURCES:
                del self.reward_marking[i]
        for i in range(len(self.reward_marking)):
            self.available_tokens += len(list(self.reward_marking.values())[i])
        if self.available_tokens == self.res[0]:
            return -1
        if self.res[0] > self.available_tokens > 0:
            return self.res[0] - self.available_tokens
        if self.available_tokens == 0:
            return self.final_reward

    def _done_status(self):
        self.status_marking = list(self._stateSpace.get().keys())
        for i in self.status_marking:
            if i in RESOURCES:
                self.status_marking.remove(i)
        if len(self.status_marking) == 0:
            return True
        return False

    def render(self, mode = "Human"):
        pass

