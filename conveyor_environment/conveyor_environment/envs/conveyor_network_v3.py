# -*- coding: utf-8 -*-
"""
@author: busach
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import logging
import sys
import time
from collections import defaultdict

sys.path.append('../../snakes_master')
from conveyor_environment.updated_trial_network import TrialConveyorNetwork
from conveyor_environment.trial_network import ConveyorNetwork

# from Conveying_Network import ConveyorNetwork, Value
from snakes.utils.simul import StateSpace
from snakes import ConstraintError, ModeError

logger = logging.getLogger(__name__)

JOBS_TRIAL = [1, 2, 3]
JOBS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
RESOURCES = ['Red', 'Green', 'Blue', 'Violet']
PLACES_TRIAL = ['S', 'S1', 'N1', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'N2', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3',
                'E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'J1', 'J2', 'J3', 'G1', 'G2', 'G3', 'K1', 'K2', 'K3', 'T1',
                'W1', 'Red', 'Green', 'N3', 'N4', 'N0', 'N6', 'N9']

PLACES = ['S', 'S1', 'N1', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'N2', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3',
          'E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'J1', 'J2', 'J3', 'G1', 'G2', 'G3', 'K1', 'K2', 'K3', 'T1',
          'W1', 'Red', 'Green', 'N3', 'N4', 'N0', 'N6', 'N9', 'I1', 'I2', 'I3', 'H1', 'H2', 'H3', 'L1',
          'L2', 'L3', 'M1', 'M2', 'M3', 'O1', 'O2', 'O3', 'P1', 'P2', 'P3', 'Q1', 'Q2', 'Q3', 'N5', 'N7',
          'N8', 'W2', 'Blue', 'Violet']

NEXT_TRANSITIONS = {'S': ['s1'], 'S1': ['SN1'], 'N1': ['P_A1', 'N_B3'], 'A1': ['AN1', 'P_A2'], 'A2': ['N_A1', 'P_A3'],
                    'A3': ['N_A2', 'AN2'], 'B1': ['BN9', 'P_B2'], 'B2': ['N_B1', 'P_B3'], 'B3': ['N_B2', 'BN1'],
                    'N2': ['N_A3', 'P_C1', 'N_E3'], 'C1': ['CN2', 'P_C2'], 'C2': ['N_C1', 'P_C3'], 'C3':
                    ['N_C2', 'C2W1', 'C3W1', 'C0W1', 'C1W1'], 'D1': ['P_D2', 'DN4'], 'D2': ['N_D1', 'P_D3'],
                    'D3': ['N_D2', 'D1W1', 'D0W1', 'D3W1', 'D2W1'], 'E1': ['EN3', 'P_E2'], 'E2': ['P_E3', 'N_E1'],
                    'E3': ['N_E2', 'EN2'], 'F1': ['FN3', 'P_F2'], 'F2': ['N_F1', 'P_F3'], 'F3': ['N_F2', 'FN4'],
                    'J1': ['P_J2', 'JN9'], 'J2': ['P_J3', 'N_J1'], 'J3': ['N_J2', 'JN6'], 'G1': ['P_G2', 'GN6'],
                    'G2': ['P_G3', 'N_G1'], 'G3': ['N_G2', 'GN3'], 'K1': ['P_K2', 'KN0'], 'K2': ['P_K3', 'N_K1'],
                    'K3': ['N_K2', 'KN9'], 'T1': ['T'], 'W1': ['N_D3', 'N_C3'], 'Red': ['D1W1', 'D3W1', 'C1W1', 'C3W1'],
                    'Green': ['C2W1', 'C3W1', 'D2W1', 'D3W1'], 'N3': ['P_E1', 'N_G3', 'P_F1'], 'N4':
                    ['P_D1', 'N_F3', 'N_H3'], 'N0': ['P_K1', 't1', 'N_O3'], 'N6': ['P_G1', 'N_J3', 'N_I3'],
                    'N9': ['P_B1', 'N_K3', 'P_J1'], 'H1': ['P_H2', 'HN5'], 'H2': ['N_H1', 'P_H3'],
                    'H3': ['N_H2', 'HN4'], 'I1': ['IN7', 'P_I2'], 'I2': ['N_I1', 'P_I3'], 'I3': ['N_I2', 'IN6'],
                    'L1': ['P_L2', 'L0W2', 'L4W2', 'L8W2', 'L12W2'], 'L2': ['P_L3', 'N_L1'], 'L3': ['N_L2', 'LN5'],
                    'M1': ['MN8', 'P_M2'], 'M2': ['N_M1', 'P_M3'], 'M3': ['N_M2', 'M0W2', 'M4W2', 'M8W2', 'M12W2'],
                    'O1': ['ON8', 'P_O2'], 'O2': ['N_O1', 'P_O3'], 'O3': ['N_O2', 'ON0'], 'P1': ['PN7', 'P_P2'],
                    'P2': ['N_P1', 'P_P3'], 'P3': ['N_P2', 'PN5'], 'Q1': ['QN8', 'P_Q2'], 'Q2': ['N_Q1', 'P_Q3'],
                    'Q3': ['N_Q2', 'QN7'], 'N5': ['N_L3', 'P_H1', 'N_P3'], 'N7': ['P_P1', 'N_Q3', 'P_I1'], 'N8':
                    ['P_M1', 'P_Q1', 'P_O1'], 'W2': ['P_L1', 'N_M3'], 'Blue': ['M4W2', 'M12W2', 'L4W2', 'L12W2'],
                    'Violet': ['M8W2', 'M12W2', 'L8W2', 'L12W2']}

NEXT_TRANSITIONS_TRIAL = {'S': ['s1'], 'S1': ['SN1'], 'N1': ['P_A1', 'N_B3'], 'A1': ['AN1', 'P_A2'],
                          'A2': ['N_A1', 'P_A3'], 'A3': ['N_A2', 'AN2'], 'B1': ['BN9', 'P_B2'], 'B2': ['N_B1', 'P_B3'],
                          'B3': ['N_B2', 'BN1'], 'N2': ['N_A3', 'P_C1', 'N_E3'], 'C1': ['CN2', 'P_C2'],
                          'C2': ['N_C1', 'P_C3'], 'C3': ['N_C2', 'C2W1', 'C3W1', 'C0W1', 'C1W1'], 'D1': ['P_D2', 'DN4'],
                          'D2': ['N_D1', 'P_D3'], 'D3': ['N_D2', 'D1W1', 'D0W1', 'D3W1', 'D2W1'], 'E1': ['EN3', 'P_E2'],
                          'E2': ['P_E3', 'N_E1'], 'E3': ['N_E2', 'EN2'], 'F1': ['FN3', 'P_F2'], 'F2': ['N_F1', 'P_F3'],
                          'F3': ['N_F2', 'FN4'], 'J1': ['P_J2', 'JN9'], 'J2': ['P_J3', 'N_J1'], 'J3': ['N_J2', 'JN6'],
                          'G1': ['P_G2', 'GN6'], 'G2': ['P_G3', 'N_G1'], 'G3': ['N_G2', 'GN3'], 'K1': ['P_K2', 'KN0'],
                          'K2': ['P_K3', 'N_K1'], 'K3': ['N_K2', 'KN9'], 'T1': ['T'], 'W1': ['N_D3', 'N_C3'],
                          'Red': ['D1W1', 'D3W1', 'C1W1', 'C3W1'], 'Green': ['C2W1', 'C3W1', 'D2W1', 'D3W1'],
                          'N3': ['P_E1', 'N_G3', 'P_F1'], 'N4': ['P_D1', 'N_F3'], 'N0': ['P_K1', 't1'],
                          'N6': ['P_G1', 'N_J3'], 'N9': ['P_B1', 'N_K3', 'P_J1'], }

for v in NEXT_TRANSITIONS.values():
    if len(v) < 5:
        for i in range(5 - len(v)):
            v.append('Nan')

for v in NEXT_TRANSITIONS_TRIAL.values():
    if len(v) < 5:
        for i in range(5 - len(v)):
            v.append('Nan')

TRANSITION_TRIAL = ['s1', 'SN1', 'AN1', 'BN1', 'P_A1', 'P_A2', 'P_A3', 'N_A1', 'N_A2', 'N_A3', 'P_B1', 'P_B2', 'P_B3',
                    'N_B1', 'N_B2', 'N_B3', 'BN9', 'JN9', 'KN9', 'P_K1', 'P_K2', 'P_K3', 'N_K1', 'N_K2', 'N_K3', 'KN0',
                    't1', 'T', 'P_J1', 'P_J2', 'P_J3', 'N_J1', 'N_J2', 'N_J3', 'JN6', 'GN6', 'P_G1', 'P_G2', 'P_G3',
                    'N_G1', 'N_G2', 'N_G3', 'GN3', 'FN3', 'EN3', 'P_E1', 'P_E2', 'P_E3', 'N_E1', 'N_E2', 'N_E3', 'AN2',
                    'EN2', 'CN2', 'P_C1', 'P_C2', 'P_C3', 'N_C1', 'N_C2', 'N_C3', 'C0W1', 'C1W1', 'C2W1', 'C3W1',
                    'D0W1', 'D1W1', 'D2W1', 'D3W1', 'P_D1', 'P_D2', 'P_D3', 'N_D1', 'N_D2', 'N_D3', 'DN4', 'FN4',
                    'P_F1', 'P_F2', 'P_F3', 'N_F1', 'N_F2', 'N_F3']

TRANSITION = ['s1', 'SN1', 'AN1', 'BN1', 'P_A1', 'P_A2', 'P_A3', 'N_A1', 'N_A2', 'N_A3', 'P_B1', 'P_B2', 'P_B3',
              'N_B1', 'N_B2', 'N_B3', 'BN9', 'JN9', 'KN9', 'P_K1', 'P_K2', 'P_K3', 'N_K1', 'N_K2', 'N_K3', 'KN0',
              't1', 'T', 'P_J1', 'P_J2', 'P_J3', 'N_J1', 'N_J2', 'N_J3', 'JN6', 'GN6', 'P_G1', 'P_G2', 'P_G3',
              'N_G1', 'N_G2', 'N_G3', 'GN3', 'FN3', 'EN3', 'P_E1', 'P_E2', 'P_E3', 'N_E1', 'N_E2', 'N_E3', 'AN2',
              'EN2', 'CN2', 'P_C1', 'P_C2', 'P_C3', 'N_C1', 'N_C2', 'N_C3', 'C0W1', 'C1W1', 'C2W1', 'C3W1', 'D0W1',
              'D1W1', 'D2W1', 'D3W1', 'P_D1', 'P_D2', 'P_D3', 'N_D1', 'N_D2', 'N_D3', 'DN4', 'FN4', 'P_F1', 'P_F2',
              'P_F3', 'N_F1', 'N_F2', 'N_F3', 'P_H1', 'P_H2', 'P_H3', 'N_H1', 'N_H2', 'N_H3', 'P_I1', 'P_I2', 'P_I3',
              'N_I1', 'N_I2', 'N_I3', 'P_L1', 'P_L2', 'P_L3', 'N_L1', 'N_L2', 'N_L3', 'P_M1', 'P_M2', 'P_M3', 'N_M1',
              'N_M2', 'N_M3', 'P_O1', 'P_O2', 'P_O3', 'N_O1', 'N_O2', 'N_O3', 'P_P1', 'P_P2', 'P_P3', 'N_P1', 'N_P2',
              'N_P3', 'P_Q1', 'P_Q2', 'P_Q3', 'N_Q1', 'N_Q2', 'N_Q3', 'HN4', 'HN5', 'IN6', 'IN7', 'LN5', 'L0W2', 'L4W2',
              'L8W2', 'L12W2', 'M0W2', 'M4W2', 'M8W2', 'M12W2', 'MN8', 'ON8', 'ON0', 'PN5', 'PN7', 'QN7', 'QN8']


def generate_random_orders(version, seed):
    """ Generates the random orders for the conveying network"""
    np.random.seed(seed)
    init = 0
    quantity = []
    orders = {}
    if version == 'trial':
        size = np.random.choice(JOBS_TRIAL)
        jobs = np.random.randint(1, 2, size, dtype=np.int16)
        quantity = np.zeros(len(jobs), dtype=np.int16)
        red = np.random.randint(100, 5000, 1, dtype=np.int16)[0]
        green = np.random.randint(100, 5000, 1, dtype=np.int16)[0]
        orders = defaultdict(list)
        for i in range(len(jobs)):
            quantity[i] = int(np.random.randint(1, 2, 1, dtype=np.int16)[0])
            orders[f"job_{jobs[i]}"].append(quantity[i])
            init += quantity[i]
        resources = [init, red, green]

        print(f'jobs {jobs}, resources {resources}, quantity {quantity}, orders {orders}')

        return jobs, resources, quantity, orders
    else:
        size = np.random.choice(JOBS)
        jobs = np.random.randint(1, 16, size, dtype=np.int16)
        quantity = np.zeros(len(jobs), dtype=np.int16)
        red = np.random.randint(100, 5000, 1, dtype=np.int16)[0]
        green = np.random.randint(100, 5000, 1, dtype=np.int16)[0]
        blue = np.random.randint(100, 5000, 1, dtype=np.int16)[0]
        violet = np.random.randint(100, 5000, 1, dtype=np.int16)[0]
        orders = defaultdict(list)
        for i in range(len(jobs)):
            quantity[i] = int(np.random.randint(1, 4, 1, dtype=np.int16)[0])
            orders[f"job_{jobs[i]}"].append(quantity[i])
            init += quantity[i]
        resources = [init, red, green, blue, violet]

        return jobs, resources, quantity, orders


def random_resource_generator(seed):
    np.random.seed(seed)
    res_size = np.random.randint(100, 5000, 1, dtype=np.int16)

    return res_size


class ConveyorEnv_v3(gym.Env):
    metadata = {'render.modes': ['Human']}

    def __init__(self, env_config, version="trial1", final_reward=10, mask=True):
        self.version = version
        self.env_config = env_config
        self.final_reward = final_reward
        self.mask = mask
        self.done = False
        # self.seed = seeding.create_seed()
        self.seed = 42
        self.jobs, self.res, self.quantity, self.orders = generate_random_orders(self.version, self.seed)
        self.throughput = []
        self.avg_throughput = 0
        self.o_c_time = []
        self.count = 0
        self.pass_this = False
        self.error = False
        self.termination = False
        self.object_no = 0
        self.completed_orders = np.zeros(len(self.jobs))
        self.o_c_time = np.zeros(len(self.jobs))
        self.order_throughput = np.zeros(len(self.jobs))
        self.order_complete = False
        if self.version == 'trial':
            self.network = TrialConveyorNetwork(self.jobs, self.res, self.quantity)
            self.net, self.trans = self.network.trial_conveyor_petrinet()
            self.no_places = len(PLACES_TRIAL)
            self.no_trans = len(TRANSITION_TRIAL)
        else:
            self.network = ConveyorNetwork(self.jobs, self.res, self.quantity)
            self.net, self.trans = self.network.conveyor_petrinet()
            self.no_places = len(PLACES)
            self.no_trans = len(TRANSITION)

        self.reward_range = [-1, 1, self.final_reward]
        # Observation space represents the places
        place = self.no_places + 3
        obs_space = spaces.Box(0, 1, shape=(place,))
        # Action space represents the transitions to get fired
        self.action_space = spaces.Discrete(5)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(5,)),
                "avail_actions": spaces.Box(0, 1, shape=(5,)),
                "state": obs_space
            })
        else:
            self.observation_space = obs_space

        self.reset()

    def _RESET(self):
        self.seed = self.seed + 1
        np.random.seed(self.seed)
        if self.version == 'trial':
            self.network = TrialConveyorNetwork(self.jobs, self.res, self.quantity)
            self.net, self.trans = self.network.trial_conveyor_petrinet()
        else:
            self.network = ConveyorNetwork(self.jobs, self.res, self.quantity)
            self.net, self.trans = self.network.conveyor_petrinet()
        self.reward = 0
        self.step_count = 0
        self.total_time_units = 0
        self.throughput = []
        self.avg_throughput = 0
        self.o_c_time = np.zeros(len(self.jobs))
        self.order_throughput = np.zeros(len(self.jobs))
        self.order_time = 0
        self.count = 0
        self.done = False
        self.pass_this = False
        self.error = False
        self.termination = False
        self._next_observation('s1')
        self.object_no = 0
        self.order_complete = False

        return self.state

    def _next_observation(self, current_transition):
        self.marking = self.net.get_marking()
        state = None
        if current_transition != 'Nan' and self.error is False:
            self.current_token = self.net.transition(current_transition).modes()[0]
            self.next_place = self.net.post(current_transition)

        if self.pass_this:
            self.current_token = self.net.transition(current_transition).modes()[0]
            self.next_place = str(self.net.post(current_transition))

        if self.version == 'trial':
            for i in PLACES_TRIAL:
                if state is None:
                    state = np.array([1 if i in list(self.marking.keys()) else 0], dtype=np.int8)
                    state = np.concatenate((state, 0, 0, 0), axis=None)
                    start = None
                else:
                    state = np.concatenate((state, np.array([1 if i in list(self.marking.keys()) else 0],
                                                            dtype=np.int8)), axis=None)
            if start:
                state = np.concatenate((state, 0, 0, 0), axis = None)
            else:
                state = np.concatenate((state, self.current_token['dir'], self.current_token['c'],
                                        self.current_token['f']), axis = None)
        else:
            for i in PLACES:
                if state is None:
                    state = np.array([1 if i in list(self.marking.keys()) else 0], dtype=np.int8)
                    start = True
                else:
                    state = np.concatenate((state, np.array([1 if i in list(self.marking.keys()) else 0],
                                                            dtype=np.int8)), axis=None)
            if start:
                state = np.concatenate((state, 0, 0, 0), axis = None)
            else:
                state = np.concatenate((state, self.current_token['dir'], self.current_token['c'],
                                        self.current_token['f']), axis = None)
        if self.mask:
            if not start:
                if self.version == 'trial':
                    transition = np.array(NEXT_TRANSITIONS_TRIAL[self.next_place])
                else:
                    transition = np.array(NEXT_TRANSITIONS[self.next_place])
                for idx, i in enumerate(transition):
                    if i != 'Nan':
                        if str(self.net.post(i)) in list(self.marking.keys()):
                            transition[idx] = 'Nan'
                mask = np.where(transition == 'Nan', 0, 1)
                self.state = {
                    "action_mask": mask,
                    "avail_actions": np.ones(5),
                    "state": state
                }
            else:
                self.state = {
                    "action_mask": np.ones(5),
                    "avail_actions": np.ones(5),
                    "state": state
                }
        else:
            self.state = state
        return self.state

    def _STEP(self, action):
        # check for the resources, if not present add resources randomly
        self.marking = self.net.get_marking()
        if self.version == 'trial':
            if 'Red' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                r = [1 for _ in range(r_size)]
                self.net.add_marking(Marking(Red=MultiSet(r)))
            elif 'Green' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                g = [2 for _ in range(r_size)]
                self.net.add_marking(Marking(Green=MultiSet(g)))
        else:
            if 'Red' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                r = [1 for _ in range(r_size)]
                self.net.add_marking(Marking(Red=MultiSet(r)))
            elif 'Green' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                g = [2 for _ in range(r_size)]
                self.net.add_marking(Marking(Green=MultiSet(g)))
            elif 'Blue' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                b = [4 for _ in range(r_size)]
                self.net.add_marking(Marking(Blue=MultiSet(b)))
            elif 'Violet' not in self.marking:
                r_size = random_resource_generator(seed=self.seed)
                v = [8 for _ in range(r_size)]
                self.net.add_marking(Marking(Violet=MultiSet(v)))

        # Calculating the epsilon time instances
        if self.step_count == 0:
            self.marking_places = list(self.marking.keys())
            if self.version == 'trial':
                self.marking_places.remove("Red")
                self.marking_places.remove("Green")
            else:
                self.marking_places.remove("Red")
                self.marking_places.remove("Green")
                self.marking_places.remove("Blue")
                self.marking_places.remove("Violet")
            self.eps_times = len(self.marking_places)
            self.current_marking = self.marking_places

        # Execute 1 time step within the environment
        current_place = self.current_marking[-1 - self.step_count]
        print(self.net.get_marking().keys())
        print(self.step_count)
        print('eps', self.eps_times)
        current_transition = self._take_action(action, current_place)
        if not self.error or self.pass_this:
            self.step_count += 1
            self.count = 0

        if self.step_count == self.eps_times:
            self.total_time_units += 1
            self.step_count = 0

        reward = self._calculate_reward()
        print(f'Reward: {self.reward}.... total time units : {self.total_time_units}')
        done = self._done_status()
        state = self._next_observation(current_transition)

        return state, reward, done, {}

    def _get_obs(self):
        return self.state

    def _take_action(self, action, place):
        if self.version == 'trial':
            trans_fire = NEXT_TRANSITIONS_TRIAL[place][action]
        else:
            trans_fire = NEXT_TRANSITIONS[place][action]
        print(trans_fire)
        self.error = False
        if trans_fire is not 'Nan':
            self.error = False
            self.termination = False
            modes = self.net.transition(trans_fire).modes()
            if len(modes) != 0:
                token = [(modes[0]['dir'], modes[0]['sq_no'], modes[0]['c'], modes[0]['f'], modes[0]['count'])]
                # print(f'modes: {modes}')
                if trans_fire == 't1':
                    self.termination = True
                    print(f'\n Termination of token ',
                          f'\n token : {modes[0]["sq_no"]}, c: {modes[0]["c"]}, f: {modes[0]["f"]}')
                try:
                    self.net.transition(trans_fire).fire(modes[0])
                    # print(f'\n token : {modes[0]["sq_no"]}, c: {modes[0]["c"]}, f: {modes[0]["f"]}')
                except ConstraintError as e1:
                    print(f'{e1}')
                    self.count += 1
                    self.error = True
                    self.net.place(place).add(token)
                    if self.count >= 5:
                        self.pass_this = True
                except ValueError as e2:
                    self.count += 1
                    print(f'{e2}: {trans_fire} is not provided with valid substitution.')
                    self.error = True
                    self.net.place(place).add(token)
                    if self.count >= 5:
                        self.pass_this = True
                except:
                    self.count += 1
                    print(f'{place} and {trans_fire}, something went wrong!!!')
                    self.error = True
                    self.net.place(place).add(token)
                    if self.count >= 5:
                        self.pass_this = True
            else:
                self.error = True
                if self.count >= 5:
                    self.pass_this = True
        else:
            self.error = True

        return trans_fire

    def _calculate_reward(self):
        self.reward_marking_keys = list(self.net.get_marking().keys())
        self.reward_marking = dict(self.net.get_marking())
        self.available_tokens = 0
        for i in self.reward_marking_keys:
            if i in RESOURCES:
                del self.reward_marking[i]
        for i in range(len(self.reward_marking)):
            self.available_tokens += len(list(self.reward_marking.values())[i])
        if not self.error:
            if self.termination:
                if self.order_complete:
                    self.reward = 1
                    return self.reward
                else:
                    self.reward = 0.5
                    return self.reward
            elif self.available_tokens == 0:
                self.reward = self.final_reward
                return self.reward
            else:
                self.reward = -0.1
                return self.reward
        else:
            self.reward = -1
            return self.reward

    def _done_status(self):
        self.status_marking = list(self.net.get_marking().keys())
        for i in self.status_marking:
            if i in RESOURCES:
                self.status_marking.remove(i)
        if len(self.status_marking) == 0:
            # print(f'Returning done as True')
            return True
        else:
            # print(f'Returning done as False')
            return False

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()

    def render(self, mode="Human"):
        self.marking = self.net.get_marking()
        s = "object no.: {:2d}, reward: {:2d}, avg_Throughput: {:2f}"
        s1 = "Order no.: {:2d}, Order Throughput: {:2f}, Avg. System Throughput: {:2f}"
        if 'T1' in list(self.marking.keys()):
            self.throughput.append(list(self.marking['T1'])[0][-1])
            self.avg_throughput = np.sum(self.completed_orders) / np.mean(self.throughput)
            order_no = list(self.marking['T1'])[0][2]
            for idx, num in enumerate(self.jobs):
                if num == order_no:
                    if self.completed_orders[idx] < self.quantity[idx]:
                        self.completed_orders[idx] += 1
                        self.o_c_time[idx] += list(self.marking['T1'])[0][-1]
                        self.order_throughput = self.completed_orders[idx] / self.o_c_time[idx]
                        if self.completed_orders[idx] == self.quantity[idx]:
                            self.order_complete = True
                            print(s1.format(order_no, self.order_throughput[idx], self.avg_throughput))
                        break
            print(s.format(list(self.marking['T1'])[0][1], self.reward, self.avg_throughput))

