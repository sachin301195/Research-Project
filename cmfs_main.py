import argparse
import platform
import logging
import getpass
import sys
sys.path.append('./conveyor_environment/snakes_master')

import ray
from ray.rllib import agents
from util import CustomPlot, TorchParametricActionModel
from conveyor_environment.conveyor_environment.envs.conveyor_network_v1 import ConveyorEnv_v1
from conveyor_environment.conveyor_environment.envs.conveyor_network_v0 import ConveyorEnv_v0

import numpy as np
from pathlib import Path
import time
import os
import random

import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print


def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('./logs/application-main-' + timestamp + '.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


logger = configure_logger()

BASE_PATH = '.'
RESULTS_PATH = './results/'
REWARD_RESULTS_PATH = '/reward-results/'
AVG_OVR_EP_PATH = '/avg_over_ep-results/'

torch, nn = try_import_torch()


# def parse_args():
#     """
#     Parse the arguments from shell.
#     :return: args object
#     """
#     if platform.system() == 'Windows':
#         default_base_dir = BASE_PATH
#     else:
#         if getpass.getuser() == 'sachin':
#             default_base_dir = r"/home/sachin/Research-Project"
#         else:
#             default_base_dir = BASE_PATH
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--base-dir',
#                         type=str,
#                         required=False,
#                         default=default_base_dir,
#                         help="BASE_DIR for agents, envs etc(Git-Repo)", )
#     parser.add_argument('--env',
#                         type=str,
#                         required=False,
#                         default=ConveyorEnv_v1,
#                         help="env to train or evaluate an agent. default=ConveyorEnv_v1",
#                         choices=['ConveyorEnv_v1', 'ConveyorEnv_v0'])
#     parser.add_argument('--memory',
#                         type=int,
#                         required=False,
#                         default=30000000000,
#                         help="ray.init(object_store_memory=memory), default=30gb.")
#     # parser.add_argument('--use-action-masking',
#     #                     type=bool,
#     #                     required=False,
#     #                     default=False,
#     #                     help="Whether to mask out unallowed actions or not. default=False")
#     # parser.add_argument('--docker',
#     #                     type=bool,
#     #                     required=False,
#     #                     default=False,
#     #                     help="Determining whether the script runs inside a docker. default=False")
#
#     subparsers = parser.add_subparsers(dest='modes',
#                                        help="train, evaluate, visualize, tune, or export")
#
#     subp_train = subparsers.add_parser('train', help='train a single or multiple agents')
#
#     subp_train.add_argument('--algo',
#                             type=str, required=False, default='APEX',
#                             choices=['DQN', 'APEX', 'PPO', 'A3C', 'IMPALA', 'SAC'],
#                             help="Algorithm to train the agent. default=DQN")
#     subp_train.add_argument('--algo-id', type=str, required=False, default='last',
#                             help="ID to choose the parameters for algorithm in corresponding config-file. default=last")
#     subp_train.add_argument('--test-mode', type=str, required=False,
#                             default='no_test', help="mode whether and when to test the agent",
#                             choices=['no_test', 'during_train_test', 'after_train_test', 'both_test'])
#     subp_train.add_argument('--checkpoint-nr', type=int, required=False, default=0,
#                             help="Whether to continue training with given checkpoint nr or not (=0). Default=0.", )
#     subp_train.add_argument('--debug_metrics', type=bool, required=False, default=False,
#                             help="Print metric values inside environment. default=False")
#
#     subp_train.add_argument('--exp-dir', type=str, required=False, default="",
#                             help="This is to provide custom path to load restore point from.")
#
#     subp_evaluate = subparsers.add_parser('evaluate', help="evaluate agents")
#
#     subp_evaluate.add_argument('--algo',
#                                type=str, action='append',
#                                required=False,
#                                help="Algorithm to evaluate the agent."
#                                     "Choices are from 'DQN', 'APEX', 'PPO', 'A3C', 'IMPALA', 'SAC'")
#     subp_evaluate.add_argument('--non-rl-algos', type=str, action='append', required=False,
#                                help="List of Non-RL-controllers as baseline. Choices are from"
#                                     "'greedy','greedy-wave', 'fixed', 'random', 'adaptive', 'fixed-pattern'")
#     subp_evaluate.add_argument('--checkpoint-nr', type=int, required=False, action='append',
#                                help="Which checkpoint-nr number from trained agents to use.", )
#     subp_evaluate.add_argument('--episodes', type=int, required=False, default=1,
#                                help="Number of episodes for which we want "
#                                     "to run the simulation for evaluation. default = 1")
#     subp_evaluate.add_argument('--verbose', type=bool, required=False, default=False,
#                                help="Plots graph of each episode separately. default = False")
#     subp_evaluate.add_argument('--debug_metrics', type=bool, required=False, default=False,
#                                help="Print metric values inside environment. default=False")
#     subp_evaluate.add_argument('--ci', type=bool, required=False, default=False,
#                                help="Plots Confidence Interval of the evaluation graphs. default=False")
#     subp_evaluate.add_argument('--exp-dir', type=str, required=False, default="",
#                                help="Directory path to load experiment trials from.")
#     subp_evaluate.add_argument('--skip-trials', type=str, action='append', required=False, default=None,
#                                help="Trials to be skipped while evaluating.")
#     subp_evaluate.add_argument('--exp-name', type=str, required=False, default="",
#                                help="Name of the directory to be used to store evaluation results.")
#
#     subp_export = subparsers.add_parser('export', help='export the model of a trained agent')
#     subp_export.add_argument('--algo',
#                              type=str, required=False, default='PPO',
#                              choices=['DQN', 'APEX', 'PPO', 'A3C', 'IMPALA', 'SAC'],
#                              help="Algorithm used for training the agent. default=PPO")
#     subp_export.add_argument('--path-to-checkpoint-and-config', type=str, required=True,
#                              help="The path must contain the checkpoint and config file.", )
#     subp_export.add_argument('--checkpoint-nr', type=int, required=True,
#                              help="Checkpoint number.")
#     subp_export.add_argument('--export-dir', type=str, required=False, default='',
#                              help="The path where to store the policy model.", )
#     subp_export.add_argument('--debug_metrics', type=bool, required=False, default=False,
#                              help="Print metric values inside environment. default=False")
#
#     args = parser.parse_args()
#     args_dict = vars(args)
#     args_dict['algo_config'] = None
#     if not args.modes:
#         parser.print_help()
#         exit(1)
#     return args

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="DQN",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier."
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Weather this script should be run as a test: --stop-reward must "
         "be achieved within --stop-timesteps AND --stop-iters")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=50,
    help="Number of iterations to train")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=0.1,
    help="Reward at which we stop training.")
parser.add_argument(
    "--no-tune",
    type=bool,
    help="Run without Tune using a manual train loop instead. In this case,"
         "use DQN without grid search and no TensorBoard.")
parser.add_argument(
    "--local-mode",
    help="Init Ray in local mode for easier debugging.",
    action="store_true"
)


if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode, object_store_memory=30000000000)

    ModelCatalog.register_custom_model(
        "conveyor_mask", TorchParametricActionModel
    )

    # env = create_env('conveyor_network_v0')

    config = {
        "env": ConveyorEnv_v1,
        "env_config": {
            "version": "trial",
            "final_reward": 2,
            "mask": True
        },
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "conveyor_mask",
            "vf_share_layers": True
        },
        "num_workers": 1,  # parallelism
        "framework": args.framework
    }
    stop = {
        "training_iterations": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward
    }

    if args.no_tune:
        # manual training with train loop using DQN and fixed learning rate
        if args.run != "DQN":
            raise ValueError("Only support --run DQN with __no-time")
        print("Running manual train loop without Ray Tune")
        dqn_config = dqn.DEFAULT_CONFIG.copy()
        dqn_config.update(config)
        dqn_config["lr"] = 1e-3
        trainer = dqn.DQNTrainer(config=dqn_config, env=ConveyorEnv_v0)

        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))

            result.save(CHECKPOINT_ROOT)

            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= args.stop_timesteps or \
                    result["episode_reward_mean"] >= args.stop_reward:
                trainer.stop()
                break

    else:
        # automated run with tune and grid search and Tensorboard
        print("Training with Ray Tune.")
        result = tune.run(args.run, config=config, stop=stop)

        if args.as_test:
            print("Checking if the learning goals are achieved")
            check_learning_achieved((result, args.stop_reward))

    ray.shutdown()
