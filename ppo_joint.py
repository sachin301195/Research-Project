import argparse
import platform
import logging
import getpass
import sys
from abc import ABC

import gym

sys.path.append('./conveyor_environment/snakes_master')

from ray.rllib import agents
from util import TorchParametricActionModel, TorchParametricActionsModelv1, TorchParametricActionsModelv2
from util import TorchParametricActionsModelv3, TorchParametricActionModelv4, TorchParametricActionsModelv5
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
from ray.rllib.agents import dqn
from ray.rllib.agents import ppo
from ray.rllib.agents import a3c
from ray.rllib.agents import impala
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
    file_handler = logging.FileHandler(f'./logs/application-{args.algo}-{str(args.no_of_jobs)}' + timestamp + '.log')
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

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="MultiEnv",
    choices=["ConveyorEnv_v1", "ConveyorEnv_v2", "ConveyorEnv_v3", "ConveyorEnv_token_n", "ConveyorEnv_A",
             "ConveyorEnv_B", "ConveyorEnv_C", "ConveyorEnv_D", "MultiEnv"],
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--algo",
    type=str,
    default="PPO",
    choices=["PPO", "SAC", "A2C", "A3C", "DQN", "IMPALA", "APEX", "APEX_DDPG", "PG"],
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
    help="final reward at the end of successful completion of the episode"
)
parser.add_argument(
    "--no-tune",
    default=False,
    type=bool,
    help="Run without Tune using a manual train loop instead. In this case,"
         "use ALGO without TensorBoard.")
parser.add_argument(
    "--state-extension",
    default=False,
    type=bool,
    help="Use the extended form of the state vector or not")
parser.add_argument(
    "--lstm",
    default=False,
    type=bool,
    help="Use LSTM or not")
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
    help="Number of tokens to initialize in an environment. This should be less than or equal to self.no_of_jobs."
)


def setup(algo, no_of_jobs, env, timestamp):
    Path(f'./plots/{algo}/{str(no_of_jobs)}/greed_search').mkdir(parents=True, exist_ok=True)
    plots_save_path = './plots/' + algo + '/' + str(no_of_jobs)
    Path(f'./agents_runs/{env}/{algo}/{str(no_of_jobs)}/{timestamp}').mkdir(parents=True, exist_ok=True)
    agent_save_path = './agents_runs/' + env + '/' + algo + '/' + str(no_of_jobs) + '/' + timestamp
    best_agent_save_path = './agents_runs/' + env + '/' + algo + '/' + str(no_of_jobs) + '/' + timestamp \
                           + '_best_agents'
    # best_agent_save_path = './agents_runs/ConveyorEnv_token_n/PPO_best_agents'
    Path(best_agent_save_path).mkdir(parents=True, exist_ok=True)

    return plots_save_path, agent_save_path, best_agent_save_path


class MultiEnv(gym.Env, ABC):
    def __init__(self, env_config):
        # if env_config.worker_index % 3 == 0:
        #     self.env = ConveyorEnv_A({'version': 'full', 'final_reward': args.final_reward, 'mask': True,
        #                               'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs,
        #                               'state_extension': args.state_extension, })
        #     self.name = "ConveyorEnv_A"
        if env_config.worker_index % 3 == 0:
            self.env = ConveyorEnv_B({'version': 'full', 'final_reward': args.final_reward, 'mask': True,
                                      'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs,
                                      'state_extension': args.state_extension, })
            self.name = 'ConveyorEnv_B'
        elif env_config.worker_index % 3 == 1:
            self.env = ConveyorEnv_C({'version': 'full', 'final_reward': args.final_reward, 'mask': True,
                                      'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs,
                                      'state_extension': args.state_extension, })
            self.name = 'ConveyorEnv_C'
        else:
            self.env = ConveyorEnv_D({'version': 'full', 'final_reward': args.final_reward, 'mask': True,
                                      'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs,
                                      'state_extension': args.state_extension, })
            self.name = 'ConveyorEnv_D'
        # if env_config.vector_index % 4 == 0:
        #     self.env = ConveyorEnv_A({'version': 'full', 'final_reward': args.final_reward, 'mask': True,
        #                               'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs})
        #     self.name = "ConveyorEnv_A"
        # elif env_config.vector_index % 4 == 1:
        #     self.env = ConveyorEnv_B({'version': 'full', 'final_reward': args.final_reward, 'mask': True,
        #                               'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs})
        #     self.name = 'ConveyorEnv_B'
        # elif env_config.vector_index % 4 == 2:
        #     self.env = ConveyorEnv_C({'version': 'full', 'final_reward': args.final_reward, 'mask': True,
        #                               'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs})
        #     self.name = 'ConveyorEnv_C'
        # else:
        #     self.env = ConveyorEnv_D({'version': 'full', 'final_reward': args.final_reward, 'mask': True,
        #                               'no_of_jobs': args.no_of_jobs, 'init_jobs': args.init_jobs})
        #     self.name = 'ConveyorEnv_D'
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


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
    register_env("env_cfms_joint", lambda c: MultiEnv(c))

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
        ModelCatalog.register_custom_model(
            "env_cfms_joint", TorchParametricActionsModelv2
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
        ModelCatalog.register_custom_model(
            "env_cfms_joint", TorchParametricActionsModelv3
        )
    if args.lstm:
        ModelCatalog.register_custom_model(
            "env_cfms_A", TorchParametricActionsModelv5
        )
        ModelCatalog.register_custom_model(
            "env_cfms_B", TorchParametricActionsModelv5
        )
        ModelCatalog.register_custom_model(
            "env_cfms_C", TorchParametricActionsModelv5
        )
        ModelCatalog.register_custom_model(
            "env_cfms_D", TorchParametricActionsModelv5
        )
        ModelCatalog.register_custom_model(
            "env_cfms_joint", TorchParametricActionsModelv5
        )

    if args.algo == 'DQN':
        cfg = {
            "hiddens": [],
            "dueling": False,
        }
    else:
        cfg = {}

    if args.algo == 'PPO' or args.algo == 'A3C':
        config = dict({
            "env": 'env_cfms_joint',
            "model": {
                "custom_model": "env_cfms_joint",
                "vf_share_layers": True,
            },
            "env_config": {
                "version": "full",
                "final_reward": 'B',
                "mask": True,
                "no_of_jobs": args.no_of_jobs,
                "init_jobs": args.init_jobs,
                'state_extension': args.state_extension,
            },
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 33,  # parallelism
            "framework": 'torch',
            "rollout_fragment_length": 125,
            "train_batch_size": 4125,
            # "sgd_minibatch_size": 512,
            # "num_sgd_iter": 20,
            "vf_loss_coeff": 0.0005,
            # "vf_loss_coeff": tune.grid_search([0.0005, 0.0009]),
            # "vf_clip_param": 10,
            # "lr": tune.grid_search([0.001, 0.0001])
            "lr": 0.0001,
            # "optimizer": "SGD",
            # "entropy_coeff": tune.grid_search([tune.uniform(0.0001, 0.001), tune.uniform(0.0001, 0.001),
            #                                    tune.uniform(0.0001, 0.001), tune.uniform(0.0001, 0.001),
            #                                    tune.uniform(0.0001, 0.001)]),
            # "num_envs_per_worker": 4,
            # "horizon": 32,
            # "timesteps_per_batch": 2048,
        },
            **cfg)
        if args.algo == 'PPO':
            algo_config = ppo.DEFAULT_CONFIG.copy()
        elif args.algo == 'A3C':
            algo_config = a3c.DEFAULT_CONFIG.copy()
        else:
            algo_config = dqn.DEFAULT_CONFIG.copy()
        algo_config.update(config)
        algo_config['model']['fcnet_activation'] = 'relu'
        if args.lstm:
            algo_config['model']['use_lstm'] = True
            algo_config['model']['lstm_cell_size'] = 64
        algo_config['evaluation_interval'] = 50
        # algo_config['evaluation_duration'] = 10
        algo_config["evaluation_parallel_to_training"]: True
    else:
        algo_config = None

    stop = {
        "training_iteration": 100 * args.no_of_jobs,
        # "episode_reward_mean": 30 - (40 * args.no_of_jobs * 0.002),
    }
    plots_save_path, agent_save_path, best_agent_save_path = setup(args.algo, args.no_of_jobs, args.env, timestamp)

    # automated run with tune and grid search and Tensorboard
    print("Training with Ray Tune.")
    print('...............................................................................\n'
          '\n\n\t\t\t\t\t\t\t\t Training Starts Here\n\n\n......................................')
    result = tune.run(args.algo, config=algo_config, stop=stop, local_dir=best_agent_save_path, log_to_file=True,
                      checkpoint_at_end=True, checkpoint_freq=50, reuse_actors=False, verbose=3,
                      checkpoint_score_attr='min-episode_len_mean', restore='PPO_CHECKPOINTS/'
                                                                            'PPO_env_cfms_joint_75712_00001_1_'
                                                                            'final_reward=B,vf_loss_coeff='
                                                                            '0.0005_2022-07-16_22-13-02/'
                                                                            'checkpoint_000450/checkpoint-450')
    logger.info(result)
    print('...............................................................................\n'
          '\n\n\t\t\t\t\t\t\t\t Training Ends Here\n\n\n........................................')

    ray.shutdown()
