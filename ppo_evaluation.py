import argparse
import platform
import logging
import getpass
import sys

sys.path.append('./conveyor_environment/snakes_master')

from ray.rllib import agents
from util import TorchParametricActionModel, TorchParametricActionsModelv1, TorchParametricActionsModelv2, \
    TorchParametricActionsModelv3
from conveyor_environment.conveyor_environment.envs.conveyor_network_v1 import ConveyorEnv_v1
from conveyor_environment.conveyor_environment.envs.conveyor_network_v0 import ConveyorEnv_v0
from conveyor_environment.conveyor_environment.envs.conveyor_network_v2 import ConveyorEnv_v2
from conveyor_environment.conveyor_environment.envs.conveyor_network_v3 import ConveyorEnv_v3
from conveyor_environment.conveyor_environment.envs.conveyor_network_v4 import ConveyorEnv_v4
from conveyor_environment.conveyor_environment.envs.conveyor_network_A import ConveyorEnv_A
from conveyor_environment.conveyor_environment.envs.conveyor_network_B import ConveyorEnv_B
from conveyor_environment.conveyor_environment.envs.conveyor_network_C import ConveyorEnv_C
from conveyor_environment.conveyor_environment.envs.conveyor_network_D import ConveyorEnv_D
from conveyor_environment.conveyor_environment.envs.conveyor_network_token_n import ConveyorEnv_token_n

# import numpy as np
# import pandas as pd
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import time
import os
# import random

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents import a3c
from ray.rllib.agents import dqn
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path(f"./logs/{args.algo}").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(f'./logs/application-{args.algo}-{str(args.no_of_jobs)}-Evaluation_ENV_AA_and_BA'
                                       + timestamp + '.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger, timestamp


BASE_PATH = '.'
RESULTS_PATH = './results/'
REWARD_RESULTS_PATH = '/reward-results/'
AVG_OVR_EP_PATH = '/avg_over_ep-results/'
CHECKPOINT_ROOT = './checkpoints'

# VALUE = {
#     0: {'lr': 0.01,
#         'vf_loss_coeff': 0.001},
#     1: {'lr': 0.001,
#         'vf_loss_coeff': 0.001},
#     2: {'lr': 0.0001,
#         'vf_loss_coeff': 0.001},
#     3: {'lr': 0.01,
#         'vf_loss_coeff': 0.0009},
#     4: {'lr': 0.001,
#         'vf_loss_coeff': 0.0009},
#     5: {'lr': 0.0001,
#         'vf_loss_coeff': 0.0009},
#     6: {'lr': 0.01,
#         'vf_loss_coeff': 0.0005},
#     7: {'lr': 0.001,
#         'vf_loss_coeff': 0.0005},
#     8: {'lr': 0.0001,
#         'vf_loss_coeff': 0.0005},
#     9: {'lr': 0.01,
#         'vf_loss_coeff': 0.0001},
#     10: {'lr': 0.001,
#          'vf_loss_coeff': 0.0001},
#     11: {'lr': 0.0001,
#          'vf_loss_coeff': 0.0001},
#     12: {'lr': 0.01,
#          'vf_loss_coeff': 0.00009},
#     13: {'lr': 0.001,
#          'vf_loss_coeff': 0.00009},
#     14: {'lr': 0.0001,
#          'vf_loss_coeff': 0.00009},
#     15: {'lr': 0.001,
#          'vf_loss_coeff': 0.001},
#     16: {'lr': 0.0001,
#          'vf_loss_coeff': 0.001},
#     17: {'lr': 0.001,
#          'vf_loss_coeff': 0.0009},
#     18: {'lr': 0.0001,
#          'vf_loss_coeff': 0.0009},
#     19: {'lr': 0.001,
#          'vf_loss_coeff': 0.0005},
#     20: {'lr': 0.0001,
#          'vf_loss_coeff': 0.0005},
#     21: {'lr': 0.001,
#          'vf_loss_coeff': 0.0001},
#     22: {'lr': 0.0001,
#          'vf_loss_coeff': 0.0001},
#     23: {'lr': 0.001,
#          'vf_loss_coeff': 0.00009},
#     24: {'lr': 0.0001,
#          'vf_loss_coeff': 0.00009},
#
# }

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="ConveyorEnv_A",
    choices=["ConveyorEnv_v1", "ConveyorEnv_v2", "ConveyorEnv_v3", "ConveyorEnv_token_n", "ConveyorEnv_A",
             "ConveyorEnv_B", "ConveyorEnv_C", "ConveyorEnv_D"],
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--algo",
    type=str,
    default="PPO",
    choices=["PPO", "SAC", "A2C", "A3C", "DQN", "DDPG", "APEX", "APEX_DDPG", "PG"],
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier."
)
parser.add_argument(
    "--eval-tune",
    default=True,
    type=bool,
    help="Evaluation while training TRUE/FALSE"
)
parser.add_argument(
    "--eval-interval",
    default=100,
    type=int,
    help="Evaluation after n training iterations"
)
parser.add_argument(
    "--final-reward",
    default='B',
    type=str,
    help="Reward Version from A, B and C"
)
parser.add_argument(
    "--no-tune",
    default=True,
    type=bool,
    help="Run without Tune using a manual train loop instead. In this case,"
         "use ALGO without TensorBoard.")
parser.add_argument(
    "--state-extension",
    default=False,
    type=bool,
    help="Use the extended form of the state vector or not")
parser.add_argument(
    "--local-mode",
    help="Init Ray in local mode for easier debugging.",
    action="store_true"
)
parser.add_argument(
    "--no_of_jobs",
    default=10,
    type=int,
    help="Number of tokens to run in an environment."
)
parser.add_argument(
    "--init_jobs",
    default=4,
    type=int,
    help="Number of tokens to initialize in an environment. This should be greater than or equal to self.no_of_jobs."
)


def evaluate(algo, algo_config: dir, plots_save_path):
    # f = []
    # cnt = 0
    # for root, dirs, files in os.walk(best_agent_save_path):
    #     for idx, name in enumerate(files):
    #         if cnt % 3 == 0 and cnt > 0:
    #             if idx == 1:
    #                 f.append(os.path.join(root, name))
    #     cnt += 1
    # f = [r'./agents_runs/ConveyorEnv_A/PPO/4/checkpoint_000400/checkpoint-400']
    f = [r'PPO_CHECKPOINTS/'
         'PPO_env_cfms_joint_75712_00001_1_'
         'final_reward=B,vf_loss_coeff='
         '0.0005_2022-07-16_22-13-02/'
         'checkpoint_000450/checkpoint-450']
    for no, path in enumerate(f):
        # try:
        #     # ppo_config['lr'] = VALUE[no]['lr']
        #     # ppo_config['vf_loss_coeff'] = VALUE[no]['vf_loss_coeff']
        #     if algo == 'PPO':
        #         agent = ppo.PPOTrainer(config=algo_config, env=ConveyorEnv_A)
        #     elif algo == 'A3c':
        #         agent = a3c.A3CTrainer(config=algo_config, env=ConveyorEnv_A)
        #     else:
        #         agent = dqn.DQNTrainer(config=algo_config, env=ConveyorEnv_B)
        #     # agent.restore(f'{checkpoint_path}/checkpoint_{no}/checkpoint-{no}')
        #     agent.restore(path)
        #     # agent.restore(f'agents_runs/ConveyorEnv_v4/DQN_best_agents/{checkpoint}/checkpoint-{checkpoint_nr}')
        #     # logger.info(f"Evaluating algo: PPO, checkpoint_nr: checkpoint_{checkpoint_nr}")
        #     logger.info(f"Evaluating algo: {algo} with no_of_jobs: {args.no_of_jobs} and init_jobs: {args.init_jobs}")
        #     curr_episode = 1
        #     max_episode = 10
        #     # if no < 15:
        #     #     env = ConveyorEnv_A(
        #     #         {'version': 'full', 'final_reward': 1000, 'mask': True, 'no_of_jobs': args.no_of_jobs,
        #     #          'init_jobs': args.init_jobs})
        #     # else:
        #     #     env = ConveyorEnv_B(
        #     #         {'version': 'full', 'final_reward': 1000, 'mask': True, 'no_of_jobs': args.no_of_jobs,
        #     #          'init_jobs': args.init_jobs})
        #     env = ConveyorEnv_B({'version': 'full', 'final_reward': 'B', 'mask': True, 'no_of_jobs': args.no_of_jobs,
        #                          'init_jobs': args.init_jobs, 'state_extension': args.state_extension})
        #     time.sleep(10)
        #     SCORE_OVERALL = []
        #     AVG_SCORE_EPISODE = []
        #     JOBS = []
        #     TIME_UNITS_EACH_OBJECT = []
        #     AVG_TOTAL_TIME_UNITS = []
        #     AVG_THROUGHPUT = []
        #     jobs = []
        #     time_units_each_object = []
        #     avg_total_time_units = 0
        #     avg_throughput = 0
        #     score_episode = []
        #     trans_logs = []
        #     time_begin = time.time()
        #     while curr_episode <= max_episode:
        #         logger.info(f"Evaluating episode: {curr_episode}")
        #         obs = env.reset()
        #         done = False
        #         score = 0
        #         step = 1
        #         while not done:
        #             # print(f'step: {step}')
        #             score_episode.append(score)
        #             action = agent.compute_action(obs)
        #             obs, reward, done, info = env.step(action)
        #             score += reward
        #             step += 1
        #             if len(info) > 0:
        #                 jobs.append(info['token'][0][-2])
        #                 time_units_each_object.append(info['token'][0][-1])
        #                 logger.info(f'Object details: {info["token"]}')
        #
        #         SCORE_OVERALL.append(score)
        #         avg_score = score / step
        #         AVG_SCORE_EPISODE.append(avg_score)
        #         avg_total_time_units = sum(time_units_each_object) / len(time_units_each_object)
        #         avg_throughput = 1 / avg_total_time_units
        #         AVG_TOTAL_TIME_UNITS.append(avg_total_time_units)
        #         AVG_THROUGHPUT.append(avg_throughput)
        #         JOBS.append(jobs)
        #         TIME_UNITS_EACH_OBJECT.extend(time_units_each_object)
        #
        #         logger.info(f"Episode_no: {curr_episode}")
        #         logger.info(f"jobs: {jobs}")
        #         logger.info(f"time_units_each_object: {time_units_each_object}")
        #         logger.info(f"Steps: {step}")
        #         logger.info(f"Mean Rewards: {avg_score}")
        #         logger.info(f"Total Reward: {score}")
        #         logger.info(f"Avg Time: {avg_total_time_units}")
        #         logger.info(f"Avg throughput: {avg_throughput}")
        #         curr_episode += 1
        #     plt.figure(1)
        #     plt.title('Mean Reward')
        #     plt.xlabel('Episode Run')
        #     plt.ylabel('Reward')
        #     # plt.plot(AVG_THROUGHPUT)
        #     # plt.savefig(f'{plots_save_path}/avg_throughput_{no}.png')
        #     plt.plot(SCORE_OVERALL)
        #     plt.savefig(f'{plots_save_path}/rewards_overall_{no}.png')
        #     plt.figure(2)
        #     plt.title('Mean Steps')
        #     plt.xlabel('Episode Run')
        #     plt.ylabel('Steps')
        #     plt.plot(AVG_TOTAL_TIME_UNITS)
        #     plt.savefig(f'{plots_save_path}/avg_timetaken_{no}.png')
        #     plt.figure(3)
        #     plt.title('Individual Termination Steps')
        #     plt.xlabel('Episode Run')
        #     plt.ylabel('Steps')
        #     plt.plot(TIME_UNITS_EACH_OBJECT)
        #     plt.savefig(f'{plots_save_path}/timetaken_{no}.png')
        #     # Measure Time
        #     time_end = time.time()
        #     time_diff = time_end - time_begin
        #     time_diff_h = int(time_diff / 3600)
        #     time_diff_min = int((time_diff - time_diff_h * 3600) / 60)
        #     time_diff_sec = int(time_diff - time_diff_h * 3600 - time_diff_min * 60)
        #     logger.info(f'Evaluation took {time_diff_h}h, {time_diff_min}m and {time_diff_sec}s.')
        #     logger.debug(f'Evaluation of checkpoint - {400}, configuration - {no} is Complete.')
        # except Exception as e:
        #     print(e, '\n Interrupted the evaluation.')
        if algo == 'PPO':
            agent = ppo.PPOTrainer(config=algo_config, env=ConveyorEnv_A)
        elif algo == 'A3C':
            agent = a3c.A3CTrainer(config=algo_config, env='env_cfms_B')
        else:
            agent = dqn.DQNTrainer(config=algo_config, env=ConveyorEnv_B)
        # agent.restore(f'{checkpoint_path}/checkpoint_{no}/checkpoint-{no}')
        agent.restore(path)
        # agent.restore(f'agents_runs/ConveyorEnv_v4/DQN_best_agents/{checkpoint}/checkpoint-{checkpoint_nr}')
        # logger.info(f"Evaluating algo: PPO, checkpoint_nr: checkpoint_{checkpoint_nr}")
        logger.info(f"Evaluating algo: {algo} with no_of_jobs: {args.no_of_jobs} and init_jobs: {args.init_jobs}")
        curr_episode = 1
        max_episode = 10
        # if no < 15:
        #     env = ConveyorEnv_A(
        #         {'version': 'full', 'final_reward': 1000, 'mask': True, 'no_of_jobs': args.no_of_jobs,
        #          'init_jobs': args.init_jobs})
        # else:
        #     env = ConveyorEnv_B(
        #         {'version': 'full', 'final_reward': 1000, 'mask': True, 'no_of_jobs': args.no_of_jobs,
        #          'init_jobs': args.init_jobs})
        env = ConveyorEnv_B({'version': 'full', 'final_reward': 'B', 'mask': True, 'no_of_jobs': args.no_of_jobs,
                             'init_jobs': args.init_jobs, 'state_extension': args.state_extension})
        time.sleep(10)
        SCORE_OVERALL = []
        AVG_SCORE_EPISODE = []
        JOBS = []
        TIME_UNITS_EACH_OBJECT = []
        AVG_TOTAL_TIME_UNITS = []
        AVG_THROUGHPUT = []
        jobs = []
        time_units_each_object = []
        avg_total_time_units = 0
        avg_throughput = 0
        score_episode = []
        trans_logs = []
        time_begin = time.time()
        while curr_episode <= max_episode:
            logger.info(f"Evaluating episode: {curr_episode}")
            obs = env.reset()
            done = False
            score = 0
            step = 1
            while not done:
                # print(f'step: {step}')
                score_episode.append(score)
                action = agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
                score += reward
                step += 1
                if len(info) > 0:
                    jobs.append(info['token'][0][-2])
                    time_units_each_object.append(info['token'][0][-1])
                    logger.info(f'Object details: {info["token"]}')

            SCORE_OVERALL.append(score)
            avg_score = score / step
            AVG_SCORE_EPISODE.append(avg_score)
            avg_total_time_units = sum(time_units_each_object) / len(time_units_each_object)
            avg_throughput = 1 / avg_total_time_units
            AVG_TOTAL_TIME_UNITS.append(avg_total_time_units)
            AVG_THROUGHPUT.append(avg_throughput)
            JOBS.append(jobs)
            TIME_UNITS_EACH_OBJECT.extend(time_units_each_object)

            logger.info(f"Episode_no: {curr_episode}")
            logger.info(f"jobs: {jobs}")
            logger.info(f"time_units_each_object: {time_units_each_object}")
            logger.info(f"Steps: {step}")
            logger.info(f"Mean Rewards: {avg_score}")
            logger.info(f"Total Reward: {score}")
            logger.info(f"Avg Time: {avg_total_time_units}")
            logger.info(f"Avg throughput: {avg_throughput}")
            curr_episode += 1
        plt.figure(1)
        plt.title('Mean Reward')
        plt.xlabel('Episode Run')
        plt.ylabel('Reward')
        # plt.plot(AVG_THROUGHPUT)
        # plt.savefig(f'{plots_save_path}/avg_throughput_{no}.png')
        plt.plot(SCORE_OVERALL)
        plt.savefig(f'{plots_save_path}/rewards_overall_{no}.png')
        plt.figure(2)
        plt.title('Mean Steps')
        plt.xlabel('Episode Run')
        plt.ylabel('Steps')
        plt.plot(AVG_TOTAL_TIME_UNITS)
        plt.savefig(f'{plots_save_path}/avg_timetaken_{no}.png')
        plt.figure(3)
        plt.title('Individual Termination Steps')
        plt.xlabel('Episode Run')
        plt.ylabel('Steps')
        plt.plot(TIME_UNITS_EACH_OBJECT)
        plt.savefig(f'{plots_save_path}/timetaken_{no}.png')
        # Measure Time
        time_end = time.time()
        time_diff = time_end - time_begin
        time_diff_h = int(time_diff / 3600)
        time_diff_min = int((time_diff - time_diff_h * 3600) / 60)
        time_diff_sec = int(time_diff - time_diff_h * 3600 - time_diff_min * 60)
        logger.info(f'Evaluation took {time_diff_h}h, {time_diff_min}m and {time_diff_sec}s.')
        logger.debug(f'Evaluation of checkpoint - {400}, configuration - {no} is Complete.')


def setup(algo, no_of_jobs, env, timestamp):
    Path(f'./plots/{algo}/{str(no_of_jobs)}').mkdir(parents=True, exist_ok=True)
    plots_save_path = './plots/' + algo + '/' + str(no_of_jobs)
    Path(f'./agents_runs/{env}/{algo}/{str(no_of_jobs)}/{timestamp}').mkdir(parents=True, exist_ok=True)
    agent_save_path = './agents_runs/' + env + '/' + algo + '/' + str(no_of_jobs) + '/' + timestamp
    best_agent_save_path = './agents_runs/' + env + '/' + algo + '/' + str(no_of_jobs) + '/' + timestamp \
                           + '_best_agents'
    # best_agent_save_path = './agents_runs/ConveyorEnv_token_n/PPO_best_agents'
    Path(best_agent_save_path).mkdir(parents=True, exist_ok=True)

    return plots_save_path, agent_save_path, best_agent_save_path


if __name__ == '__main__':
    args = parser.parse_args()
    logger, timestamp = configure_logger()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode, object_store_memory=1000000000)
    register_env("env_cfms_A", lambda _: ConveyorEnv_A({'version': 'full', 'final_reward': args.final_reward,
                                                        'mask': True, 'state_extension': args.state_extension,
                                                        'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs}))
    register_env("env_cfms_B", lambda _: ConveyorEnv_B({'version': 'full', 'final_reward': args.final_reward,
                                                        'mask': True, 'state_extension': args.state_extension,
                                                        'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs}))
    register_env("env_cfms_C", lambda _: ConveyorEnv_C({'version': 'full', 'final_reward': args.final_reward,
                                                        'mask': True, 'state_extension': args.state_extension,
                                                        'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs}))
    register_env("env_cfms_D", lambda _: ConveyorEnv_D({'version': 'full', 'final_reward': args.final_reward,
                                                        'mask': True, 'state_extension': args.state_extension,
                                                        'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs}))
    if not args.state_extension:
        ModelCatalog.register_custom_model(
            "env_cfms_A", TorchParametricActionsModelv2
        )
        ModelCatalog.register_custom_model(
            "env_cfms_B", TorchParametricActionsModelv2
        )
        ModelCatalog.register_custom_model(
            "env_cfms_C", TorchParametricActionsModelv2
        )
        ModelCatalog.register_custom_model(
            "env_cfms_D", TorchParametricActionsModelv2
        )
    else:
        ModelCatalog.register_custom_model(
            "env_cfms_A", TorchParametricActionsModelv3
        )
        ModelCatalog.register_custom_model(
            "env_cfms_B", TorchParametricActionsModelv3
        )
        ModelCatalog.register_custom_model(
            "env_cfms_C", TorchParametricActionsModelv3
        )
        ModelCatalog.register_custom_model(
            "env_cfms_D", TorchParametricActionsModelv3
        )

    if args.algo == 'DQN':
        cfg = {
            "hiddens": [],
            "dueling": False,
        }
    else:
        cfg = {}

    if args.algo == 'PPO':
        config = dict({
            "env": "env_cfms_B",
            "model": {
                "custom_model": "env_cfms_B",
                "vf_share_layers": True,
            },
            "env_config": {
                "version": "full",
                "final_reward": args.final_reward,
                "mask": True,
                'state_extension': args.state_extension,
                "no_of_jobs": args.no_of_jobs,
                "init_jobs": args.init_jobs,
            },
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 0,  # parallelism
            "framework": 'torch',
            "rollout_fragment_length": 125,
            "train_batch_size": 4000,
            # "sgd_minibatch_size": 512,
            # "num_sgd_iter": 20,
            "vf_loss_coeff": 0.0005,
            # "vf_loss_coeff": 0.0001,
            # "lr": tune.grid_search([0.001, 0.0001])
            "lr": 0.0001,
            # "horizon": 32,
            # "timesteps_per_batch": 2048,
        },
            **cfg)
        algo_config = ppo.DEFAULT_CONFIG.copy()
        algo_config.update(config)
        algo_config['model']['fcnet_activation'] = 'relu'
        algo_config['evaluation_interval'] = 100
        # algo_config['evaluation_duration'] = 10
        algo_config["evaluation_parallel_to_training"]: True
    else:
        algo_config = None

    plots_save_path, agent_save_path, best_agent_save_path = setup(args.algo, args.no_of_jobs, args.env, timestamp)

    print("Running manual train loop without Ray Tune")
    # checkpoint_path = train(algo_config)
    evaluate(args.algo, algo_config, plots_save_path)

    ray.shutdown()
