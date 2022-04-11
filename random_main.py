from conveyor_environment.conveyor_environment.envs.conveyor_network_v4 import ConveyorEnv_v4
from conveyor_environment.conveyor_environment.envs.conveyor_network_token_n import ConveyorEnv_token_n
import random
import logging
import sys
from pathlib import Path
import time
import csv
import os
import matplotlib.pyplot as plt


def configure_logger():
    Path(f'./agents_runs/ConveyorEnv_token_n/random/').mkdir(parents=True, exist_ok=True)
    agent_save_path = './agents_runs/' + 'ConveyorEnv_token_n' + '/' + 'random'
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
agent_save_path = './agents_runs/' + 'ConveyorEnv_token_n' + '/' + 'random/'

BASE_PATH = '.'
RESULTS_PATH = './results/'
REWARD_RESULTS_PATH = '/reward-results/'
AVG_OVR_EP_PATH = '/avg_over_ep-results/'
CHECKPOINT_ROOT = './checkpoints'

NUM_EPISODES = 10
REWARDS = []
AVG_THROUGHPUT = []
ORDER_THROUGHPUT = []

env = ConveyorEnv_v4({'version': 'full', 'final_reward': 1000, 'mask': True, 'no_of_jobs': 2})

results = []
episode_data = []
header = ['jobs', 'quantity', 'time_units_each_object', 'total_order_completion_time', 'avg_order_completion_time',
          'avg_order_throughput', 'avg_total_time_units', 'avg_throughput']
with open(agent_save_path + 'episodic_result.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

logger.debug('Start Training.')
time_begin = time.time()
episode_save_counter = 0
best_avg_reward = -1000000
episode_completion_time = []

for n in range(NUM_EPISODES):
    done = False
    score = 0
    step_count = 0
    obs = env.reset()
    actions = obs['action_mask']
    # print(actions)

    while not done:
        final_actions = []
        for idx, a in enumerate(actions):
            if a != 0:
                final_actions.append(idx)
        # print(final_actions)
        action = random.choice(final_actions)
        # print(action)
        obs, reward, done, info = env.step(action)
        score += reward
        step_count += 1
        # actions = list(obs['action_mask'])
        actions = obs['action_mask']
        # info = env.render()
        # AVG_THROUGHPUT.append(avg_throughput)
        # ORDER_THROUGHPUT.append(order_throughput)

    avg_reward_per_episode = score/step_count
    logger.info(f"Episode_no: {n}")
    logger.info(f"Mean Rewards: {avg_reward_per_episode}")
    logger.info(f"Timesteps total: {step_count}")
    logger.info(f"Episode Completion Time: {info}")
    episode_completion_time.append(info)
    # del info['next_place']
    # with open(agent_save_path + 'episodic_result.csv', 'a', encoding='UTF8', newline='') as f:
    #     writer = csv.DictWriter(f, fieldnames=header)
    #     writer.writerow(info)
    if avg_reward_per_episode > best_avg_reward:
        best_n = n
        best_avg_reward = avg_reward_per_episode
        best_step_count = step_count
        best_episode_completion_time = info


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
