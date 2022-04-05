from conveyor_environment.conveyor_environment.envs.conveyor_network_v4 import ConveyorEnv_v4
import random
import logging
import sys
from pathlib import Path
import time
import csv
import os
import matplotlib.pyplot as plt


def configure_logger():
    Path(f'./agents_runs/ConveyorEnv_v3/deterministic/').mkdir(parents=True, exist_ok=True)
    agent_save_path = './agents_runs/' + 'ConveyorEnv_v3' + '/' + 'deterministic'
    # best_agent_save_path = './agents_runs/' + 'ConveyorEnv_v3' + '/' + 'random' + '_best_agents'
    # Path(best_agent_save_path).mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    # Path("./logs_new").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(agent_save_path + timestamp + '.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


logger = configure_logger()
agent_save_path = './agents_runs/' + 'ConveyorEnv_v3' + '/' + 'deterministic/'

BASE_PATH = '.'
RESULTS_PATH = './results/'
REWARD_RESULTS_PATH = '/reward-results/'
AVG_OVR_EP_PATH = '/avg_over_ep-results/'
CHECKPOINT_ROOT = './checkpoints'

NEXT_TRANSITIONS = {'S': ['s1'], 'S1': ['SN1'], 'N1': ['P_A1', 'N_B3'], 'A1': ['AN1', 'P_A2'], 'A2': ['N_A1', 'P_A3'],
                    'A3': ['N_A2', 'AN2'], 'B1': ['BN9', 'P_B2'], 'B2': ['N_B1', 'P_B3'], 'B3': ['N_B2', 'BN1'],
                    'N2': ['N_A3', 'P_C1', 'N_E3'], 'C1': ['CN2', 'P_C2'], 'C2': ['N_C1', 'P_C3'], 'C3':
                    ['N_C2', 'C2W1', 'C3W1', 'C0W1', 'C1W1'], 'D1': ['P_D2', 'DN4'], 'D2': ['N_D1', 'P_D3'],
                    'D3': ['N_D2', 'D1W1', 'D0W1', 'D3W1', 'D2W1'], 'E1': ['EN3', 'P_E2'], 'E2': ['P_E3', 'N_E1'],
                    'E3': ['N_E2', 'EN2'], 'F1': ['FN3', 'P_F2'], 'F2': ['N_F1', 'P_F3'], 'F3': ['N_F2', 'FN4'],
                    'J1': ['P_J2', 'JN9'], 'J2': ['P_J3', 'N_J1'], 'J3': ['N_J2', 'JN6'], 'G1': ['P_G2', 'GN6'],
                    'G2': ['P_G3', 'N_G1'], 'G3': ['N_G2', 'GN3'], 'K1': ['P_K2', 'KN0'], 'K2': ['P_K3', 'N_K1'],
                    'K3': ['N_K2', 'KN9'], 'T1': ['T'], 'W1': ['N_D3', 'N_C3'], 'N3': ['P_E1', 'N_G3', 'P_F1'], 'N4':
                    ['P_D1', 'N_F3', 'N_H3'], 'N0': ['P_K1', 't1', 'N_O3'], 'N6': ['P_G1', 'N_J3', 'N_I3'],
                    'N9': ['P_B1', 'N_K3', 'P_J1'], 'H1': ['P_H2', 'HN5'], 'H2': ['N_H1', 'P_H3'],
                    'H3': ['N_H2', 'HN4'], 'I1': ['IN7', 'P_I2'], 'I2': ['N_I1', 'P_I3'], 'I3': ['N_I2', 'IN6'],
                    'L1': ['P_L2', 'L0W2', 'L4W2', 'L8W2', 'L12W2'], 'L2': ['P_L3', 'N_L1'], 'L3': ['N_L2', 'LN5'],
                    'M1': ['MN8', 'P_M2'], 'M2': ['N_M1', 'P_M3'], 'M3': ['N_M2', 'M0W2', 'M4W2', 'M8W2', 'M12W2'],
                    'O1': ['ON8', 'P_O2'], 'O2': ['N_O1', 'P_O3'], 'O3': ['N_O2', 'ON0'], 'P1': ['PN7', 'P_P2'],
                    'P2': ['N_P1', 'P_P3'], 'P3': ['N_P2', 'PN5'], 'Q1': ['QN8', 'P_Q2'], 'Q2': ['N_Q1', 'P_Q3'],
                    'Q3': ['N_Q2', 'QN7'], 'N5': ['N_L3', 'P_H1', 'N_P3'], 'N7': ['P_P1', 'N_Q3', 'P_I1'], 'N8':
                    ['P_M1', 'P_Q1', 'P_O1'], 'W2': ['P_L1', 'N_M3']}

OPTIMUM_NEXT_TRANSITION = {'S': ['s1'], 'S1': ['SN1'], 'N1': ['P_A1'], 'A1': ['P_A2'], 'A2': ['P_A3'], 'A3': ['AN2'],
                           'B1': ['P_B2'], 'B2': ['P_B3'], 'B3': ['BN1'], 'N2': ['P_C1'], 'C1': ['P_C2'],
                           'C2': ['P_C3'], 'C3': ['C2W1', 'C3W1', 'C0W1', 'C1W1'], 'D1': ['DN4'], 'D2': ['N_D1'],
                           'D3': ['N_D2'], 'E1': ['EN3'], 'E2': ['N_E1'], 'E3': ['N_E2'], 'F1': ['P_F2'],
                           'F2': ['P_F3'], 'F3': ['FN4'], 'J1': ['JN9'], 'J2': ['N_J1'], 'J3': ['N_J2'], 'G1': ['GN6'],
                           'G2': ['N_G1'], 'G3': ['N_G2'], 'K1': ['P_K2'], 'K2': ['P_K3'], 'K3': ['KN9'], 'T1': ['T'],
                           'W1': ['N_D3'], 'N3': ['P_F1'], 'N4': ['N_H3'], 'N0': ['t1'], 'N6': ['N_J3'], 'N9': ['P_B1'],
                           'H1': ['HN5'], 'H2': ['N_H1'], 'H3': ['N_H2'], 'I1': ['P_I2'], 'I2': ['P_I3'], 'I3': ['IN6'],
                           'L1': ['L0W2', 'L4W2', 'L8W2', 'L12W2'], 'L2': ['N_L1'], 'L3': ['N_L2'], 'M1': ['MN8'],
                           'M2': ['N_M1'], 'M3': ['N_M2'], 'O1': ['P_O2'], 'O2': ['P_O3'], 'O3': ['ON0'], 'P1': ['PN7'],
                           'P2': ['N_P1'], 'P3': ['N_P2'], 'Q1': ['P_Q2'], 'Q2': ['P_Q3'], 'Q3': ['QN7'],
                           'N5': ['N_L3'], 'N7': ['P_I1'], 'N8': ['P_O1'], 'W2': ['N_M3']}

POSITION = {'S': 1, 'S1': 2, 'N1': 3, 'A1': 4, 'A2': 5, 'A3': 6, 'N2': 7, 'C1': 8, 'C2': 9, 'C3': 10, 'W1': 11, 'D3': 12
            , 'D2': 13, 'D1': 14, 'N4': 15, 'H3': 16, 'H2': 17, 'H1': 18, 'N5': 19, 'L3': 20, 'L2': 21, 'L1': 22, 'W2':
            23, 'M3': 24, 'M2': 25, 'M1': 26, 'N8': 27, 'O1': 28, 'O2': 29, 'O3': 30, 'N0': 31, 'T1': 32}

NUM_EPISODES = 2000
REWARDS = []
AVG_THROUGHPUT = []
ORDER_THROUGHPUT = []

env = ConveyorEnv_v4({'version': 'full', 'final_reward': 10, 'mask': True})

header = ['jobs', 'quantity', 'time_units_each_object', 'total_order_completion_time', 'avg_order_completion_time',
          'avg_order_throughput', 'avg_total_time_units', 'avg_throughput']
with open(agent_save_path + 'episodic_result.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

logger.debug('Start Training.')
time_begin = time.time()
episode_save_counter = 0
best_avg_reward = -1000000

for n in range(NUM_EPISODES):
    done = False
    score = 0
    step_count = 0
    obs = env.reset()
    current_marking = ['S']
    p_transition = 'Nan'
    error = False

    while not done:
        for next_place in reversed(current_marking):
            transition = OPTIMUM_NEXT_TRANSITION[next_place][0]
            while True:
                if not error:
                    action = NEXT_TRANSITIONS[next_place].index(transition)
                    first_action = action
                    count = 0
                else:
                    while True:
                        trans = random.choice(NEXT_TRANSITIONS[next_place])
                        if trans != transition:
                            break
                        action = NEXT_TRANSITIONS[next_place].index(transition)
                obs, reward, done, info = env.step(action)
                score += reward
                # actions = list(obs['action_mask'])
                next_place = info['next_place']
                error = info['error']
                if not error:
                    break
        current_marking = info['current_marking']
    avg_reward_per_episode = score / step_count
    logger.info(f"Episode_no: {n}")
    logger.info(f"Mean Rewards: {avg_reward_per_episode}")
    logger.info(f"Timesteps total: {step_count}")
    del info['next_place']
    with open(agent_save_path + 'episodic_result.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(info)
    if avg_reward_per_episode > best_avg_reward:
        best_n = n
        best_avg_reward = avg_reward_per_episode
        best_step_count = step_count


logger.info('Best Episode')
logger.info(f"Episode_no: {best_n}")
logger.info(f"Mean Rewards: {best_avg_reward}")
logger.info(f"Timesteps total: {best_step_count}")

# Measure Time
time_end = time.time()
time_diff = time_end - time_begin
time_diff_h = int(time_diff / 3600)
time_diff_min = int((time_diff - time_diff_h * 3600) / 60)
time_diff_sec = int(time_diff - time_diff_h * 3600 - time_diff_min * 60)
logger.info(f'Training took {time_diff_h}h, {time_diff_min}m and {time_diff_sec}s.')
logger.debug('Training successful.')