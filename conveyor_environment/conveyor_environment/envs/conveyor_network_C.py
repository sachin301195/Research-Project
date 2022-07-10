"""
    Same as conveyor_network_B.py
    N tokens introduced at the stating of an environment then after completion first,
    the others are introduced Randomly after every termination.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import logging
import sys
import time

sys.path.append('../../snakes_master')
from conveyor_environment.updated_trial_network import TrialConveyorNetwork
from conveyor_environment.trial_network import ConveyorNetwork

from snakes import ConstraintError

logger = logging.getLogger(__name__)

ACTION_MAPPING_TRIAL = {'S': {0: 's1', 1: 'Nan', 2: 'Nan', 3: 'Nan'},
                        'S1': {0: 'SN1', 1: 'Nan', 2: 'Nan', 3: 'Nan'},
                        'N1': {0: 'Nan', 1: 'P_A1', 2: 'Nan', 3: 'N_B3'},
                        'A1': {0: 'Nan', 1: 'P_A2', 2: 'Nan', 3: 'AN1'},
                        'A2': {0: 'Nan', 1: 'P_A3', 2: 'Nan', 3: 'N_A1'},
                        'A3': {0: 'AN2', 1: 'Nan', 2: 'Nan', 3: 'N_A2'},
                        'B1': {0: 'Nan', 1: 'P_B2', 2: 'Nan', 3: 'BN9'},
                        'B2': {0: 'Nan', 1: 'P_B3', 2: 'Nan', 3: 'N_B1'},
                        'B3': {0: 'Nan', 1: 'BN1', 2: 'Nan', 3: 'N_B2'},
                        'N2': {0: 'P_C1', 1: 'Nan', 2: 'N_A3', 3: 'N_E3'},
                        'C1': {0: 'P_C2', 1: 'Nan', 2: 'CN2', 3: 'Nan'},
                        'C2': {0: 'P_C3', 1: 'Nan', 2: 'N_C1', 3: 'Nan'},
                        'C3': {0: 'w1', 1: 'Nan', 2: 'N_C2', 3: 'Nan'},
                        'D1': {0: 'Nan', 1: 'P_D2', 2: 'Nan', 3: 'DN4'},
                        'D2': {0: 'Nan', 1: 'P_D3', 2: 'Nan', 3: 'N_D1'},
                        'D3': {0: 'Nan', 1: 'w1', 2: 'Nan', 3: 'N_D2'},
                        'E1': {0: 'Nan', 1: 'P_E2', 2: 'Nan', 3: 'EN3'},
                        'E2': {0: 'Nan', 1: 'P_E3', 2: 'Nan', 3: 'N_E1'},
                        'E3': {0: 'Nan', 1: 'EN2', 2: 'Nan', 3: 'N_E2'},
                        'F1': {0: 'P_F2', 1: 'Nan', 2: 'FN3', 3: 'Nan'},
                        'F2': {0: 'P_F3', 1: 'Nan', 2: 'N_F1', 3: 'Nan'},
                        'F3': {0: 'FN4', 1: 'Nan', 2: 'N_F2', 3: 'Nan'},
                        'J1': {0: 'P_J2', 1: 'Nan', 2: 'JN9', 3: 'Nan'},
                        'J2': {0: 'P_J3', 1: 'Nan', 2: 'N_J1', 3: 'Nan'},
                        'J3': {0: 'JN6', 1: 'Nan', 2: 'N_J2', 3: 'Nan'},
                        'G1': {0: 'Nan', 1: 'P_G2', 2: 'Nan', 3: 'GN6'},
                        'G2': {0: 'Nan', 1: 'P_G3', 2: 'Nan', 3: 'N_G1'},
                        'G3': {0: 'Nan', 1: 'GN3', 2: 'Nan', 3: 'N_G2'},
                        'K1': {0: 'Nan', 1: 'P_K2', 2: 'Nan', 3: 'KN0'},
                        'K2': {0: 'Nan', 1: 'P_K3', 2: 'Nan', 3: 'N_K1'},
                        'K3': {0: 'Nan', 1: 'KN9', 2: 'Nan', 3: 'N_K2'},
                        'T1': {0: 'Nan', 1: 'Nan', 2: 'T', 3: 'Nan'},
                        'W1': {0: 'Nan', 1: 'Nan', 2: 'N_C3', 3: 'N_D3'},
                        'N3': {0: 'P_F1', 1: 'P_E1', 2: 'Nan', 3: 'N_G3'},
                        'N4': {0: 'Nan', 1: 'P_D1', 2: 'N_F3', 3: 'Nan'},
                        'N0': {0: 'Nan', 1: 'P_K1', 2: 't1', 3: 'Nan'},
                        'N6': {0: 'Nan', 1: 'P_G1', 2: 'N_J3', 3: 'Nan'},
                        'N9': {0: 'P_J1', 1: 'P_B1', 2: 'Nan', 3: 'N_K3'}}

ACTION_MAPPING_TRIAL_COMPACT = {'S': {0: 's1', 1: 'Nan', 2: 'Nan', 3: 'Nan'},
                                'N1': {0: 'Nan', 1: 'P_A1', 2: 'Nan', 3: 'N_B3'},
                                'N2': {0: 'P_C1', 1: 'Nan', 2: 'N_A3', 3: 'N_E3'},
                                'W1': {0: 'Nan', 1: 'Nan', 2: 'N_C3', 3: 'N_D3'},
                                'N3': {0: 'P_F1', 1: 'P_E1', 2: 'Nan', 3: 'N_G3'},
                                'N4': {0: 'Nan', 1: 'P_D1', 2: 'N_F3', 3: 'Nan'},
                                'N0': {0: 'Nan', 1: 'P_K1', 2: 't1', 3: 'Nan'},
                                'N6': {0: 'Nan', 1: 'P_G1', 2: 'N_J3', 3: 'Nan'},
                                'N9': {0: 'P_J1', 1: 'P_B1', 2: 'Nan', 3: 'N_K3'}}

ACTION_MAPPING_COMPACT = {'S': {0: 's1', 1: 'Nan', 2: 'Nan', 3: 'Nan'},
                          'N1': {0: 'Nan', 1: 'P_A1', 2: 'Nan', 3: 'N_B3'},
                          'N2': {0: 'P_C1', 1: 'Nan', 2: 'N_A3', 3: 'N_E3'},
                          'W1': {0: 'Nan', 1: 'Nan', 2: 'N_C3', 3: 'N_D3'},
                          'N3': {0: 'P_F1', 1: 'P_E1', 2: 'Nan', 3: 'N_G3'},
                          'N4': {0: 'Nan', 1: 'P_D1', 2: 'N_F3', 3: 'Nan'},
                          'N0': {0: 'Nan', 1: 'P_K1', 2: 't1', 3: 'Nan'},
                          'N6': {0: 'Nan', 1: 'P_G1', 2: 'N_J3', 3: 'Nan'},
                          'N9': {0: 'P_J1', 1: 'P_B1', 2: 'Nan', 3: 'N_K3'},
                          'N5': {0: 'Nan', 1: 'P_H1', 2: 'N_P3', 3: 'N_L3'},
                          'N7': {0: 'P_P1', 1: 'P_I1', 2: 'Nan', 3: 'N_Q3'},
                          'N8': {0: 'P_M1', 1: 'P_Q1', 2: 'P_O1', 3: 'Nan'},
                          'W2': {0: 'Nan', 1: 'P_L1', 2: 'N_M3', 3: 'Nan'}
                          }

ACTION_MAPPING = {'S': {0: 's1', 1: 'Nan', 2: 'Nan', 3: 'Nan'}, 'S1': {0: 'SN1', 1: 'Nan', 2: 'Nan', 3: 'Nan'},
                  'N1': {0: 'Nan', 1: 'P_A1', 2: 'Nan', 3: 'N_B3'}, 'A1': {0: 'Nan', 1: 'P_A2', 2: 'Nan', 3: 'AN1'},
                  'A2': {0: 'Nan', 1: 'P_A3', 2: 'Nan', 3: 'N_A1'}, 'A3': {0: 'AN2', 1: 'Nan', 2: 'Nan', 3: 'N_A2'},
                  'B1': {0: 'Nan', 1: 'P_B2', 2: 'Nan', 3: 'BN9'}, 'B2': {0: 'Nan', 1: 'P_B3', 2: 'Nan', 3: 'N_B1'},
                  'B3': {0: 'Nan', 1: 'BN1', 2: 'Nan', 3: 'N_B2'}, 'N2': {0: 'P_C1', 1: 'Nan', 2: 'N_A3', 3: 'N_E3'},
                  'C1': {0: 'P_C2', 1: 'Nan', 2: 'CN2', 3: 'Nan'}, 'C2': {0: 'P_C3', 1: 'Nan', 2: 'N_C1', 3: 'Nan'},
                  'C3': {0: 'w1', 1: 'Nan', 2: 'N_C2', 3: 'Nan'}, 'D1': {0: 'Nan', 1: 'P_D2', 2: 'Nan', 3: 'DN4'},
                  'D2': {0: 'Nan', 1: 'P_D3', 2: 'Nan', 3: 'N_D1'}, 'D3': {0: 'Nan', 1: 'w1', 2: 'Nan', 3: 'N_D2'},
                  'E1': {0: 'Nan', 1: 'P_E2', 2: 'Nan', 3: 'EN3'}, 'E2': {0: 'Nan', 1: 'P_E3', 2: 'Nan', 3: 'N_E1'},
                  'E3': {0: 'Nan', 1: 'EN2', 2: 'Nan', 3: 'N_E2'}, 'F1': {0: 'P_F2', 1: 'Nan', 2: 'FN3', 3: 'Nan'},
                  'F2': {0: 'P_F3', 1: 'Nan', 2: 'N_F1', 3: 'Nan'}, 'F3': {0: 'FN4', 1: 'Nan', 2: 'N_F2', 3: 'Nan'},
                  'J1': {0: 'P_J2', 1: 'Nan', 2: 'JN9', 3: 'Nan'}, 'J2': {0: 'P_J3', 1: 'Nan', 2: 'N_J1', 3: 'Nan'},
                  'J3': {0: 'JN6', 1: 'Nan', 2: 'N_J2', 3: 'Nan'}, 'G1': {0: 'Nan', 1: 'P_G2', 2: 'Nan', 3: 'GN6'},
                  'G2': {0: 'Nan', 1: 'P_G3', 2: 'Nan', 3: 'N_G1'}, 'G3': {0: 'Nan', 1: 'GN3', 2: 'Nan', 3: 'N_G2'},
                  'K1': {0: 'Nan', 1: 'P_K2', 2: 'Nan', 3: 'KN0'}, 'K2': {0: 'Nan', 1: 'P_K3', 2: 'Nan', 3: 'N_K1'},
                  'K3': {0: 'Nan', 1: 'KN9', 2: 'Nan', 3: 'N_K2'}, 'T1': {0: 'Nan', 1: 'Nan', 2: 'T', 3: 'Nan'},
                  'W1': {0: 'Nan', 1: 'Nan', 2: 'N_C3', 3: 'N_D3'}, 'N3': {0: 'P_F1', 1: 'P_E1', 2: 'Nan', 3: 'N_G3'},
                  'N4': {0: 'Nan', 1: 'P_D1', 2: 'N_F3', 3: 'N_H3'}, 'N0': {0: 'Nan', 1: 'P_K1', 2: 't1', 3: 'N_O3'},
                  'N6': {0: 'Nan', 1: 'P_G1', 2: 'N_J3', 3: 'N_I3'}, 'N9': {0: 'P_J1', 1: 'P_B1', 2: 'Nan', 3: 'N_K3'},
                  'H1': {0: 'Nan', 1: 'P_H2', 2: 'Nan', 3: 'HN5'}, 'H2': {0: 'Nan', 1: 'P_H3', 2: 'Nan', 3: 'N_H1'},
                  'H3': {0: 'Nan', 1: 'HN4', 2: 'Nan', 3: 'N_H2'}, 'I1': {0: 'Nan', 1: 'P_I2', 2: 'Nan', 3: 'IN7'},
                  'I2': {0: 'Nan', 1: 'P_I3', 2: 'Nan', 3: 'N_I1'}, 'I3': {0: 'Nan', 1: 'IN6', 2: 'Nan', 3: 'N_I2'},
                  'L1': {0: 'Nan', 1: 'P_L2', 2: 'Nan', 3: 'w2'}, 'L2': {0: 'Nan', 1: 'P_L3', 2: 'Nan', 3: 'N_L1'},
                  'L3': {0: 'Nan', 1: 'LN5', 2: 'Nan', 3: 'N_L2'}, 'M1': {0: 'P_M2', 1: 'Nan', 2: 'MN8', 3: 'Nan'},
                  'M2': {0: 'P_M3', 1: 'Nan', 2: 'N_M1', 3: 'Nan'}, 'M3': {0: 'w2', 1: 'Nan', 2: 'N_M2', 3: 'Nan'},
                  'O1': {0: 'ON8', 1: 'P_O2', 2: 'Nan', 3: 'Nan'}, 'O2': {0: 'Nan', 1: 'P_O3', 2: 'Nan', 3: 'N_O1'},
                  'O3': {0: 'Nan', 1: 'ON0', 2: 'Nan', 3: 'N_O2'}, 'P1': {0: 'P_P2', 1: 'Nan', 2: 'PN7', 3: 'Nan'},
                  'P2': {0: 'P_P3', 1: 'Nan', 2: 'N_P1', 3: 'Nan'}, 'P3': {0: 'PN5', 1: 'Nan', 2: 'N_P2', 3: 'Nan'},
                  'Q1': {0: 'Nan', 1: 'P_Q2', 2: 'Nan', 3: 'QN8'}, 'Q2': {0: 'Nan', 1: 'P_Q3', 2: 'Nan', 3: 'N_Q1'},
                  'Q3': {0: 'Nan', 1: 'QN7', 2: 'Nan', 3: 'N_Q2'}, 'N5': {0: 'Nan', 1: 'P_H1', 2: 'N_P3', 3: 'N_L3'},
                  'N7': {0: 'P_P1', 1: 'P_I1', 2: 'Nan', 3: 'N_Q3'}, 'N8': {0: 'P_M1', 1: 'P_Q1', 2: 'P_O1', 3: 'Nan'},
                  'W2': {0: 'Nan', 1: 'P_L1', 2: 'N_M3', 3: 'Nan'}}

REWARD_MAPPING_W1 = {'N1': ['S1', 'S'], 'A1': ['N1'], 'A2': ['A1'], 'A3': ['A2'], 'N2': ['A3'], 'C1': ['N2'],
                     'C2': ['C1'], 'C3': ['C2'], 'W1': ['C3'], 'D3': ['W1'], 'D2': ['D3'], 'D1': ['D2'],
                     'N4': ['D3'],  'J3': ['N6'], 'J2': ['J3'], 'J1': ['J2'], 'N9': ['J3'], 'F3': ['N4'], 'F2': ['F3'],
                     'F1': ['F2'], 'N3': ['F3'], 'H3': ['N4'], 'H2': ['H3'], 'H1': ['H2'], 'N5': ['H3'], 'L3': ['N5'],
                     'L2': ['L3'], 'L1': ['L2'], 'W2': ['L3'], 'M3': ['W2'], 'M2': ['M3'], 'M1': ['M2'], 'N8': ['M3'],
                     'G3': ['N3'], 'G2': ['G3'], 'G1': ['G2'], 'N6': ['G3'], 'K3': ['N9'], 'K2': ['K3'], 'K1': ['K2'],
                     'N0': ['K3', 'O3'], 'O1': ['N8'], 'O2': ['O1'], 'O3': ['O2'], 'T1': ['N0']}

REWARD_MAPPING_W2 = {'N1': ['S1', 'S'], 'A1': ['N1'], 'A2': ['A1'], 'A3': ['A2'], 'N2': ['A3'], 'C1': ['N2'],
                     'C2': ['C1'], 'C3': ['C2'], 'W1': ['C3'], 'D3': ['W1'], 'D2': ['D3'], 'D1': ['D2'],
                     'N4': ['D3'], 'H3': ['N4'], 'H2': ['H3'], 'H1': ['H2'], 'N5': ['H3', 'P3'], 'L3': ['N5'],
                     'L2': ['L3'], 'L1': ['L2'], 'W2': ['L3'], 'M3': ['W2'], 'M2': ['M3'], 'M1': ['M2'], 'N8': ['M3'],
                     'B3': ['N1'], 'B2': ['B3'], 'B1': ['B2'], 'N9': ['B3'], 'I3': ['N6'], 'I2': ['I3'], 'I1': ['I2'],
                     'N7': ['I3'], 'N0': ['O3'], 'O1': ['N8'], 'O2': ['O1'], 'O3': ['O2'], 'T1': ['N0'], 'J1': ['N9'],
                     'J2': ['J1'], 'J3': ['J2'], 'N6': ['J3'], 'P1': ['N7'], 'P2': ['P1'], 'P3': ['P2']}

REWARD_MAPPING_W1_W2 = {'N1': ['S1', 'S'], 'A1': ['N1'], 'A2': ['A1'], 'A3': ['A2'], 'N2': ['A3'], 'C1': ['N2'],
                        'C2': ['C1'], 'C3': ['C2'], 'W1': ['C3'], 'D3': ['W1'], 'D2': ['D3'], 'D1': ['D2'],
                        'N4': ['D3'], 'H3': ['N4'], 'H2': ['H3'], 'H1': ['H2'], 'N5': ['H3', 'P3'], 'L3': ['N5'],
                        'L2': ['L3'], 'L1': ['L2'], 'W2': ['L3'], 'M3': ['W2'], 'M2': ['M3'], 'M1': ['M2'],
                        'N8': ['M3'], 'N0': ['O3'], 'O1': ['N8'], 'O2': ['O1'], 'O3': ['O2'], 'T1': ['N0']}


def generate_random_N_orders(version, no_of_token, seed):
    """ Generates random N orders for the conveying network """
    np.random.seed(seed)
    if version == 'trial' or version == 'trial_compact':
        jobs = np.random.randint(1, 4, size=no_of_token)
        quantity = np.ones(len(jobs), dtype=np.int16)
        red = 1
        green = 1
        resources = [no_of_token, red, green]

        print(f'jobs {jobs}, resources {resources}, quantity {quantity}')
    else:
        jobs = np.random.randint(1, 16, size=no_of_token)
        quantity = np.ones(len(jobs), dtype=np.int16)
        red = 1
        green = 1
        blue = 1
        violet = 1
        resources = [no_of_token, red, green, blue, violet]

        print(f'jobs {jobs}, resources {resources}, quantity {quantity}')

    return jobs, resources, quantity


def current_token(token, trans, action, place, step_count, error):
    if not error:
        token = list(token[0])
        details = {'p_place': place, 'c_place': trans[-2:], 'steps': step_count}
        if action == 0 or action == 1:
            token[0] = 1
        else:
            token[0] = -1
        details['dir'] = token[0]
        if trans in ['C1W1', 'D1W1', 'C2W1', 'D2W1', 'C3W1', 'D3W1', 'L4W2', 'M4W2', 'L8W2', 'M8W2', 'L12W2', 'M12W2']:
            if len(trans) > 4:
                token[2] += 12
            else:
                token[2] += int(trans[1])
        details['c_state'] = token[2]
        token[-1] += 1
        details['count'] = token[-1]
        token = [tuple(token)]
    else:
        details = {'count': token[0][-1], 'p_place': place, 'c_place': place, 'c_state': token[0][2],
                   'steps': step_count}

    return token, details


class ConveyorEnv_C(gym.Env):
    metadata = {"render.modes": ["Human"]}

    def __init__(self, env_config: dict):
        self.env_config = env_config
        self.version = env_config["version"]
        self.final_reward = env_config["final_reward"]
        self.no_of_jobs = env_config["no_of_jobs"]
        self.mask = env_config["mask"]
        self.init_jobs = env_config["init_jobs"]
        self.state_extension = env_config['state_extension']
        self.remaining_jobs = self.no_of_jobs - self.init_jobs
        self.start = True
        self.token_state = None
        self.exit_count = -1
        if self.version == 'trial':
            places = 38
        elif self.version == 'trial_compact':
            places = 9
        elif self.version == 'full':
            places = 63
        else:
            places = 13
        # Observation space represents places:
        if self.state_extension:
            places += 30
        else:
            places += 3
        obs_space = spaces.Box(-1, 1, shape=(places,))
        # Action space represents possible transitions
        self.action_space = spaces.Discrete(4)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(4,)),
                "avail_actions": spaces.Box(0, 1, shape=(4,)),
                "state": obs_space
            })
        else:
            self.observation_space = obs_space

    def reset(self):
        self.seed = seeding.create_seed(max_bytes=4)
        self.done = False
        self.start = True
        self.exit_count = -1
        self.no_of_jobs = self.env_config["no_of_jobs"]
        self.init_jobs = self.env_config["init_jobs"]
        self.remaining_jobs = self.no_of_jobs - self.init_jobs
        np.random.seed(self.seed)
        self.jobs, self.res, self.quantity = generate_random_N_orders(self.version, self.no_of_jobs, self.seed)
        if self.version == "trial" or self.version == "trial_compact":
            self.network = TrialConveyorNetwork(self.jobs[:self.init_jobs], self.res, self.quantity)
            self.net, self.trans = self.network.trial_conveyor_petrinet()
        else:
            self.network = ConveyorNetwork(self.jobs[:self.init_jobs], self.res, self.quantity)
            self.net, self.trans = self.network.conveyor_petrinet()
        self.reward = 0
        self.step_count = 0
        self.current_place = 'S'
        self.next_place = 'S'
        self.total_time_units = 0
        self.time_units = np.zeros(self.res[0])
        self.avg_throughput = 0
        self.avg_time_units = 0
        self.o_c_time = np.zeros(len(self.jobs))
        self.avg_order_complete_time = np.zeros(len(self.jobs))
        self.avg_order_throughput = np.zeros(len(self.jobs))
        self.completed_orders = np.zeros(len(self.jobs))
        self.order_time = 0
        self.termination = False
        self.error = False
        self.marking = self.net.get_marking()
        self.modes = self.net.transition('s1').modes()
        self.current_token = self.modes[0]
        self.object_no = 0
        self.order_complete = False
        self.terminating_in_middle = False
        self.transition_log = []
        self.episode_time_begin = time.time()
        self.max_length = 0
        self.eps_step = 0
        self.unit_step = 1
        self.info = {}
        self.token = {}
        self.binding = {}
        if self.version == 'trial':
            for place in ACTION_MAPPING_TRIAL:
                self.binding[place] = {}
        elif self.version == 'trial_compact':
            for place in ACTION_MAPPING_TRIAL_COMPACT.keys():
                self.binding[place] = {}
        elif self.version == 'full':
            for place in ACTION_MAPPING.keys():
                self.binding[place] = {}
        else:
            for place in ACTION_MAPPING_COMPACT.keys():
                self.binding[place] = {}
        self.binding['T'] = {}
        if self.state_extension:
            self.token_state = [0 for _ in range(30)]
            for idx, job in enumerate(self.jobs):
                self.token[f"token_{idx}"] = {}
                self.token[f"token_{idx}"]["dir"] = 0
                self.token[f"token_{idx}"]["job"] = job
                self.token[f"token_{idx}"]["c_state"] = 0
                self.token[f"token_{idx}"]["c_place"] = 'S'
                self.token[f"token_{idx}"]["p_place"] = None
                self.token[f"token_{idx}"]["count"] = 0
                self.token[f"token_{idx}"]["steps"] = 0
                self.token_state[idx * 3 + 2] = job
        else:
            for idx, job in enumerate(self.jobs):
                self.token[f"token_{idx}"] = {}
                self.token[f"token_{idx}"]["dir"] = 0
                self.token[f"token_{idx}"]["job"] = job
                self.token[f"token_{idx}"]["c_state"] = 0
                self.token[f"token_{idx}"]["c_place"] = 'S'
                self.token[f"token_{idx}"]["p_place"] = None
                self.token[f"token_{idx}"]["count"] = 0
                self.token[f"token_{idx}"]["steps"] = 0
        for key, value in self.token.items():
            self.binding['S'].update({key: value})
        state = self._next_observation()

        return state

    def _next_observation(self):
        if self.state_extension:
            state = []
            details = {}
            if not self.start:
                token = list(self.binding[self.next_place].items())[0][0]
                details = list(self.binding[self.next_place].items())[0][1]
                # print(token, details)
            if self.version == 'trial':
                for place in ACTION_MAPPING_TRIAL.keys():
                    if place in set(self.marking.keys()):
                        state.append(1)
                    else:
                        state.append(0)
            elif self.version == 'trial_compact':
                for place in ACTION_MAPPING_TRIAL_COMPACT.keys():
                    if place in set(self.marking.keys()):
                        state.append(1)
                    else:
                        state.append(0)
            elif self.version == 'full':
                for place in ACTION_MAPPING.keys():
                    if place in set(self.marking.keys()):
                        state.append(1)
                    else:
                        state.append(0)
            else:
                for place in ACTION_MAPPING_COMPACT.keys():
                    if place in set(self.marking.keys()):
                        state.append(1)
                    else:
                        state.append(0)
            if self.start:
                for token, detail in self.token.items():
                    idx = int(token[-1]) * 3 + 2
                    self.token_state[idx] = (self.token_state[idx]) / 15
                state.extend(self.token_state)
                mask = np.array((1, 0, 0, 0))
            else:
                idx = int(token[-1]) * 3
                c_state = details['c_state'] / 15
                f_state = (details['job'] - 1) / 14
                self.token_state[idx] = details['dir']
                self.token_state[idx + 1] = c_state
                self.token_state[idx + 2] = f_state
                state.extend(self.token_state)
                if self.version == 'trial':
                    transition = np.array(list(ACTION_MAPPING_TRIAL[self.next_place].values()))
                elif self.version == 'trial_compact':
                    transition = np.array(list(ACTION_MAPPING_TRIAL_COMPACT[self.next_place].values()))
                elif self.version == 'full':
                    transition = np.array(list(ACTION_MAPPING[self.next_place].values()))
                else:
                    transition = np.array(list(ACTION_MAPPING_COMPACT[self.next_place].values()))
                mask = np.where(transition == 'Nan', 0, 1)
            state = np.array(state, dtype=np.float32)

            if self.mask:
                self.state = {
                    "action_mask": mask,
                    "avail_actions": np.ones(4),
                    "state": state
                }
            else:
                self.state = state
            self.current_place = self.next_place
        else:
            state = []
            details = {}
            if not self.start:
                token = list(self.binding[self.next_place].items())[0][0]
                details = list(self.binding[self.next_place].items())[0][1]
                # print(token, details)
            if self.version == 'trial':
                for place in ACTION_MAPPING_TRIAL.keys():
                    if place in set(self.marking.keys()):
                        state.append(1)
                    else:
                        state.append(0)
            elif self.version == 'trial_compact':
                for place in ACTION_MAPPING_TRIAL_COMPACT.keys():
                    if place in set(self.marking.keys()):
                        state.append(1)
                    else:
                        state.append(0)
            elif self.version == 'full':
                for place in ACTION_MAPPING.keys():
                    if place in set(self.marking.keys()):
                        state.append(1)
                    else:
                        state.append(0)
            else:
                for place in ACTION_MAPPING_COMPACT.keys():
                    if place in set(self.marking.keys()):
                        state.append(1)
                    else:
                        state.append(0)
            if self.start:
                f_state = (int(self.current_token['f']) - 1) / 14
                state.extend([0, 0, f_state])
                mask = np.array((1, 0, 0, 0))
            else:
                c_state = details['c_state'] / 15
                f_state = (details['job'] - 1) / 14
                state.extend([details['dir'], c_state, f_state])
                if self.version == 'trial':
                    transition = np.array(list(ACTION_MAPPING_TRIAL[self.next_place].values()))
                elif self.version == 'trial_compact':
                    transition = np.array(list(ACTION_MAPPING_TRIAL_COMPACT[self.next_place].values()))
                elif self.version == 'full':
                    transition = np.array(list(ACTION_MAPPING[self.next_place].values()))
                else:
                    transition = np.array(list(ACTION_MAPPING_COMPACT[self.next_place].values()))
                mask = np.where(transition == 'Nan', 0, 1)
            state = np.array(state, dtype=np.float32)

            if self.mask:
                self.state = {
                    "action_mask": mask,
                    "avail_actions": np.ones(4),
                    "state": state
                }
            else:
                self.state = state
            self.current_place = self.next_place

        return self.state

    def step(self, action):
        self._take_action(action, self.current_place)
        self.marking = self.net.get_marking()
        self.step_count += 1
        self.eps_step += 1
        self.current_token, token_dir = current_token(self.current_token, self.trans_fire, action, self.current_place,
                                                      self.step_count, self.error)
        self.token[f"token_{self.current_token[0][1]}"].update(token_dir)
        if not self.error:
            self.binding[token_dir["c_place"]].update(
                {f"token_{self.current_token[0][1]}": self.binding[token_dir['p_place']].
                    pop(f"token_{self.current_token[0][1]}")})
        if self.eps_step == self.unit_step:
            self.marking_list = list(self.marking.keys())
            if self.version == 'trial':
                self.marking_list.remove("Red")
                self.marking_list.remove("Green")
            elif self.version == 'full':
                self.marking_list.remove("Red")
                self.marking_list.remove("Green")
                self.marking_list.remove("Blue")
                self.marking_list.remove("Violet")
            else:
                for place in self.marking_list:
                    if place in ACTION_MAPPING_COMPACT:
                        pass
                    else:
                        self.marking_list.remove(place)
            self.unit_step = len(self.marking_list)
            self.eps_step = 0
            self.time_units += 1
            if len(self.marking_list) > 0:
                self.next_place = self.marking_list[-1]
            else:
                self.next_place = None
        else:
            self.next_place = self.marking_list[-1-self.eps_step]
        self.done = self._done_status()
        self.info = self._data()
        self.reward = self._calculate_reward()
        # if self.start:
        #     self.c = 0
        self.start = False
        # if self.error:
        #     self.c += 1
        # if self.termination or self.terminating_in_middle:
        #     print(self.c)
        if not self.done:
            self.state = self._next_observation()

        return self.state, self.reward, self.done, self.info

    def _take_action(self, action, place):
        self.error = False
        if self.version == 'trial':
            self.trans_fire = ACTION_MAPPING_TRIAL[place][action]
        elif self.version == 'trial_compact':
            self.trans_fire = ACTION_MAPPING_TRIAL_COMPACT[place][action]
        elif self.version == 'full_compact':
            self.trans_fire = ACTION_MAPPING_COMPACT[place][action]
        else:
            self.trans_fire = ACTION_MAPPING[place][action]

        if self.trans_fire is not 'Nan':
            self.termination = False
            if self.trans_fire == 'w1' or self.trans_fire == 'w2':
                self.trans_fire = self._resolve_workstations(self.trans_fire, action)
            self.modes = self.net.transition(self.trans_fire).modes()
            if len(self.modes) != 0:
                self.current_token = [(self.modes[0]['dir'], self.modes[0]['sq_no'], self.modes[0]['c'],
                                       self.modes[0]['f'], self.modes[0]['count'])]
                try:
                    self.net.transition(self.trans_fire).fire(self.modes[0])
                    if self.trans_fire == 't1':
                        self.termination = True
                        # self.no_of_jobs -= 1
                        self.trans_fire = 'T'
                        self.modes = self.net.transition('T').modes()
                        self.net.transition('T').fire(self.modes[0])
                        print(f'\n Termination of token ',
                              f'\n token : {self.modes[0]["sq_no"]}, c: {self.modes[0]["c"]}, '
                              f'f: {self.modes[0]["f"]}, count: {self.modes[0]["count"]}')
                        self.exit_count += 1
                        if self.remaining_jobs > 0:
                            # Random token introduced after termination
                            if self.remaining_jobs > 1:
                                init_tokens = np.random.randint(low= 1, high= self.remaining_jobs + 1)
                            else:
                                init_tokens = 1
                            self._token_insertion(init_tokens)
                    if self.trans_fire == 's1':
                        self.trans_fire = 'SN1'
                        self.modes = self.net.transition('SN1').modes()
                        self.net.transition('SN1').fire(self.modes[0])
                    # if self.termination:
                    #     print(self.marking)
                except ConstraintError:
                    # print(f'{e1}')
                    self.error = True
                    self.current_token[0] = list(self.current_token[0])
                    self.current_token[0][-1] += 1
                    self.current_token[0] = tuple(self.current_token[0])
                    self.net.place(place).add(self.current_token)
                except ValueError:
                    # print(f'{e2}: {trans_fire} is not provided with valid substitution.')
                    self.error = True
                    self.current_token[0] = list(self.current_token[0])
                    self.current_token[0][-1] += 1
                    self.current_token[0] = tuple(self.current_token[0])
                    self.net.place(place).add(self.current_token)
                except Exception:
                    # print(f'{place} and {trans_fire}, something went wrong!!!')
                    self.error = True
                    self.current_token[0] = list(self.current_token[0])
                    self.current_token[0][-1] += 1
                    self.current_token[0] = tuple(self.current_token[0])
                    self.net.place(place).add(self.current_token)
            else:
                self.error = True
        else:
            # print(place)
            # print(self.net.get_marking())
            for trans in list(ACTION_MAPPING[place].values()):
                if trans != 'Nan' and trans != 'P_J1' and trans != 'P_J2':
                    # print(trans)
                    if trans == 'w1':
                        if place == 'D3':
                            self.modes = self.net.transition('D0W1').modes()
                        else:
                            self.modes = self.net.transition('C0W1').modes()
                    elif trans == 'w2':
                        if place == 'L1':
                            self.modes = self.net.transition('L0W2').modes()
                        else:
                            self.modes = self.net.transition('M0W2').modes()
                    else:
                        self.modes = self.net.transition(trans).modes()
                    # print(self.modes)
                    self.current_token = [(self.modes[0]['dir'], self.modes[0]['sq_no'], self.modes[0]['c'],
                                           self.modes[0]['f'], self.modes[0]['count'] + 1)]
                    self.current_token[0] = tuple(self.current_token[0])
                    if place == 'S':
                        self.net.transition(self.trans_fire).fire(self.modes[0])
                    else:
                        self.net.place(place).empty()
                    self.net.place(place).add(self.current_token)
                    break
            self.error = True

    def _resolve_workstations(self, trans, action):
        if trans == 'w1':
            if action == 0:
                c = self.net.transition('C0W1').modes()[0]['c']
                f = self.net.transition('C0W1').modes()[0]['f']
            else:
                c = self.net.transition('D0W1').modes()[0]['c']
                f = self.net.transition('D0W1').modes()[0]['f']
            if c < f:
                if (f in [1, 5, 9, 13]) and (c not in [1, 5, 9, 13]):
                    if action == 0:
                        trans = 'C1W1'
                    else:
                        trans = 'D1W1'
                elif (f in [2, 6, 10, 14]) and (c not in [2, 6, 10, 14]):
                    if action == 0:
                        trans = 'C2W1'
                    else:
                        trans = 'D2W1'
                elif (f in [3, 7, 11, 15]) and (c not in [3, 7, 11, 15]):
                    if action == 0:
                        trans = 'C3W1'
                    else:
                        trans = 'D3W1'
                else:
                    if action == 0:
                        trans = 'C0W1'
                    else:
                        trans = 'D0W1'
            else:
                if action == 0:
                    trans = 'C0W1'
                else:
                    trans = 'D0W1'
        elif trans == 'w2':
            if action == 0:
                c = self.net.transition('M0W2').modes()[0]['c']
                f = self.net.transition('M0W2').modes()[0]['f']
            else:
                c = self.net.transition('L0W2').modes()[0]['c']
                f = self.net.transition('L0W2').modes()[0]['f']
            if c < f:
                if (f in [4, 5, 6, 7]) and (c not in [4, 5, 6, 7]):
                    if action == 0:
                        trans = 'M4W2'
                    else:
                        trans = 'L4W2'
                elif (f in [8, 9, 10, 11]) and (c not in [8, 9, 10, 11]):
                    if action == 0:
                        trans = 'M8W2'
                    else:
                        trans = 'L8W2'
                elif (f in [12, 13, 14, 15]) and (c not in [12, 13, 14, 15]):
                    if action == 0:
                        trans = 'M12W2'
                    else:
                        trans = 'L12W2'
                else:
                    if action == 0:
                        trans = 'M0W2'
                    else:
                        trans = 'L0W2'
            else:
                if action == 0:
                    trans = 'M0W2'
                else:
                    trans = 'L0W2'

        return trans

    def _token_insertion(self, tokens):
        for seq in range(tokens):
            idx = self.no_of_jobs - self.remaining_jobs
            new_token = [(0, idx, 0, self.jobs[-self.remaining_jobs], 0)]
            self.net.place('S').add(new_token)
            self.token[f"token_{idx}"] = {}
            self.token[f"token_{idx}"]["dir"] = 0
            self.token[f"token_{idx}"]["job"] = self.jobs[-self.remaining_jobs]
            self.token[f"token_{idx}"]["c_state"] = 0
            self.token[f"token_{idx}"]["c_place"] = 'S'
            self.token[f"token_{idx}"]["p_place"] = None
            self.token[f"token_{idx}"]["count"] = 0
            self.token[f"token_{idx}"]["steps"] = 0
            self.binding['S'].update({f"token_{idx}": self.token[f"token_{idx}"]})
            self.remaining_jobs -= 1

    def _data(self):
        # if self.termination:
        #     order = self.modes[0]['f']
        #     object_no = self.modes[0]['sq_no']
        #     time_units = self.modes[0]['count']
        #     self.token[object_no]["job"] = order
        #     self.token[object_no]["count"] = time_units
        #     self.token[object_no]["steps"] = self.step_count
        #     self.info['Job_details'] = self.token
        #     self.o_c_time[object_no] = time_units
        #     self.avg_order_complete_time = sum(self.o_c_time) / self.no_of_jobs
        #     if self.avg_order_complete_time > 0:
        #         self.avg_throughput = 1 / self.avg_order_complete_time
        #     else:
        #         self.avg_throughput = 0
        #     self.info['time_units_each_object'] = self.o_c_time
        #     self.info['avg_order_complete_time'] = self.avg_time_units
        #     self.info['avg_throughput'] = self.avg_throughput
        # self.order_complete = False
        # for idx, job in enumerate(self.jobs):
        #     if job == order:
        #         if self.completed_orders[idx] < self.quantity[idx]:
        #             self.completed_orders[idx] += 1
        #             self.o_c_time[idx] += time_units
        #             self.time_units[object_no] += time_units
        #             if self.completed_orders[idx] == self.quantity[idx]:
        #                 self.order_complete = True
        #                 self.avg_order_complete_time[idx] += self.o_c_time[idx] / self.quantity[idx]
        #                 self.avg_order_throughput[idx] += 1 / self.avg_order_complete_time[idx]
        #                 if next_place is None:
        #                     self.avg_time_units = sum(self.time_units) / self.res[0]
        #                     self.avg_throughput = 1 / self.avg_time_units
        #                     self.info['time_units_each_object'] = self.time_units
        #                     self.info['total_order_completion_time'] = self.o_c_time
        #                     self.info['avg_order_completion_time'] = self.avg_order_complete_time
        #                     self.info['avg_order_throughput'] = self.avg_order_throughput
        #                     self.info['avg_total_time_units'] = self.avg_time_units
        #                     self.info['avg_throughput'] = self.avg_throughput
        #             break
        if self.done:
            info = {'token': self.current_token, 'all_tokens': self.token}
        elif self.termination:
            info = {'token': self.current_token}
        else:
            info = {}

        return info

    def _get_obs(self):

        return self.state

    def _calculate_reward(self):
        # if not self.error:
        #     if self.terminating_in_middle:
        #         print('termination in the middle')
        #         self.reward = -100
        #
        #         return self.reward
        #     elif self.termination:
        #         if self.done:
        #             self.reward = self.final_reward
        #
        #             return self.reward
        #         else:
        #             self.reward = 100
        #
        #             return self.reward
        #     else:
        #         self.reward = -0.01
        #
        #         return self.reward
        # else:
        #     self.reward = -1
        if self.final_reward == 'A':
            self.reward = -self.current_token[0][-1] * (1 / 100100) * (not self.error) - \
                          0.01 * self.error - \
                          5 * self.terminating_in_middle + (30 / self.no_of_jobs) * self.termination
        elif self.final_reward == 'B':
            diff = 40 / (self.no_of_jobs * (self.no_of_jobs - 1))
            self.reward = - 0.001 * (not self.error) - 0.002 * self.error * (not self.done) \
                          - 5 * self.terminating_in_middle + diff * self.exit_count * self.termination \
                          + 10 * self.done * (not self.terminating_in_middle)
        else:
            if self.current_token[0][-2] in [1, 2, 3]:
                if self.token[f"token_{self.current_token[0][1]}"]['c_place'] in REWARD_MAPPING_W1:
                    if self.token[f"token_{self.current_token[0][1]}"]['p_place'] in \
                            REWARD_MAPPING_W1[self.token[f"token_{self.current_token[0][1]}"]['c_place']]:
                        self.reward = -0.001
                    else:
                        self.reward = -0.01
                else:
                    self.reward = -0.01
            elif self.current_token[0][-2] in [4, 8, 12]:
                if self.token[f"token_{self.current_token[0][1]}"]['c_place'] in REWARD_MAPPING_W2:
                    if self.token[f"token_{self.current_token[0][1]}"]['p_place'] in \
                            REWARD_MAPPING_W2[self.token[f"token_{self.current_token[0][1]}"]['c_place']]:
                        self.reward = -0.001
                    else:
                        self.reward = -0.01
                else:
                    self.reward = -0.01
            else:
                if self.token[f"token_{self.current_token[0][1]}"]['c_place'] in REWARD_MAPPING_W1_W2:
                    if self.token[f"token_{self.current_token[0][1]}"]['p_place'] in \
                            REWARD_MAPPING_W1_W2[self.token[f"token_{self.current_token[0][1]}"]['c_place']]:
                        self.reward = -0.001
                    else:
                        self.reward = -0.01
                else:
                    self.reward = -0.01
            self.reward += (-5 * self.terminating_in_middle + 20 / self.no_of_jobs * self.termination +
                            10 * self.done * (not self.terminating_in_middle))
        # self.reward = np.clip(self.reward, a_min=-30, a_max=30)

        return self.reward

    def _done_status(self):
        if len(list(self.marking)) == (len(self.res) - 1):
            # print(f'Returning done as True')
            self.episode_time_ends = time.time()
            self.episode_time = self.episode_time_ends - self.episode_time_begin

            return True
        else:
            # print(f'Returning done as False')
            if self.current_token[0][-1] > 1000:
                self.terminating_in_middle = True
                print(f"Termination_in_middle with steps: {self.step_count}")

                return True

            return False

    def render(self, mode="Human"):
        """
        : jobs: self.jobs (list),
        : quantity: self.quantity (list),
        : next_place: self.next_place (list),
        : time_units_each_object: self.time_units (list),
        : total_order_completion_time: self.o_c_time (list),
        : avg_order_completion_time: self.avg_order_complete_time (list),
        : avg_order_throughput: self.avg_order_throughput (list),
        : avg_total_time_units: self.avg_time_units (int),
        : avg_throughput: self.avg_throughput (float)
        """
        info = self._data()

        return info.values()
