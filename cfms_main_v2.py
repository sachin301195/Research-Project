import argparse
import platform
import logging
import getpass
import sys
sys.path.append('./conveyor_environment/snakes_master')

import ray
from ray.rllib import agents
from util import CustomPlot, TorchParametricActionModel, TorchParametricActionsModelv1, TorchParametricActionsModelv2
from conveyor_environment.conveyor_environment.envs.conveyor_network_v1 import ConveyorEnv_v1
from conveyor_environment.conveyor_environment.envs.conveyor_network_v0 import ConveyorEnv_v0
from conveyor_environment.conveyor_environment.envs.conveyor_network_v2 import ConveyorEnv_v2
from conveyor_environment.conveyor_environment.envs.conveyor_network_v3 import ConveyorEnv_v3

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
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


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
CHECKPOINT_ROOT = './checkpoints'

torch, nn = try_import_torch()

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
    default=5000,
    help="Number of iterations to train")
parser.add_argument(
    "--no-tune",
    type=bool,
    default=True,
    help="Run without Tune using a manual train loop instead. In this case,"
         "use DQN without grid search and no TensorBoard.")
parser.add_argument(
    "--local-mode",
    help="Init Ray in local mode for easier debugging.",
    action="store_true"
)
parser.add_argument(
    "--env-version",
    type=str,
    default='full',
    help="version of the environment to work on [full/trial]",
)
parser.add_argument(
    "--use-mask",
    type=bool,
    default=True,
    help="use of action masking or not",
)


if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode, object_store_memory=1000000000)
    register_env("env_cfms", lambda _: ConveyorEnv_v3({}))

    if args.use_mask:
        if args.env_version == 'full':
            ModelCatalog.register_custom_model(
                "env_cfms", TorchParametricActionsModelv2
            )
            print('full')
        elif args.env_version == 'trial':
            ModelCatalog.register_custom_model(
                "env_cfms", TorchParametricActionsModelv1
            )
    else:
        ModelCatalog.register_custom_model(
            "env_cfms", TorchParametricActionModel
        )

    if args.run == 'DQN':
        cfg = {
            "hiddens": [],
            "dueling": False,
        }
    else:
        cfg = {}

    config = dict({
        "env": "env_cfms",
        "model": {
            "custom_model": "env_cfms",
        },
        "env_config": {
            "version": "trial1",
            "final_reward": 2,
            "mask": True
        },
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 5,  # parallelism
        "framework": args.framework
        },
        **cfg)

    stop = {
        "training_iteration": args.stop_iters,
    }
    print('going inside if....')
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
            # if result["timesteps_total"] >= args.stop_timesteps or \
            #         result["episode_reward_mean"] >= args.stop_reward:
            #     trainer.stop()
            #     break

    else:
        # automated run with tune and grid search and Tensorboard
        print("Training with Ray Tune.")
        result = tune.run(args.run, config=config, stop=stop)

        if args.as_test:
            print("Checking if the learning goals are achieved")
            check_learning_achieved((result, args.stop_reward))

    ray.shutdown()