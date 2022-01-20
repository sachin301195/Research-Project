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
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
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
from ray.tune.registry import register_env


def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('./logs/application-dqn_main-' + timestamp + '.log')
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
    "--env",
    type=str,
    default="ConveyorEnv_v3",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--algo",
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
    "--stop-timesteps",
    type=int,
    default=100000000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=10,
    help="Reward at which we stop training.")
parser.add_argument(
    "--no-tune",
    default=True,
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

    ray.init(local_mode=args.local_mode, object_store_memory=1000000000)
    register_env("env_cfms", lambda _: ConveyorEnv_v3({}))

    ModelCatalog.register_custom_model(
        "env_cfms", TorchParametricActionsModelv2
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
            "final_reward": 10,
            "mask": True
        },
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 32,  # parallelism
        "framework": 'torch',
        "num_atoms": 1,
        "v_min": -10,
        "v_max": 10,
        "noisy": False,
        "sigma0": 0.5,
        # "dueling": False,
        # "hiddens": [],
        "double_q": True,
        "n_step": 2,
        "target_network_update_freq": 500,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 1000,
            # "temperature": None
        },
        "buffer_size": 50000,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "final_prioritized_replay_beta": 0.4,
        "prioritized_replay_beta_annealing_timesteps": 20000,
        "prioritized_replay_eps": 1.00E-06,
        # "compress_observation": False,
        "before_learn_on_batch": None,
        "training_intensity": None,
        "lr": 5.00E-04,
        "lr_schedule": None,
        "adam_epsilon": 1.00E-08,
        "grad_clip": 40,
        "learning_starts": 1000,
        "rollout_fragment_length": 32,
        "train_batch_size": 128,
        # "num_workers": 32,
        "worker_side_prioritization": False,
        "min_iter_time_s": 30,
        "timesteps_per_iteration": 1000
        },
        **cfg)

    stop = {
        "training_iteration": 5000
    }

    if args.no_tune:
        # manual training with train loop using DQN and fixed learning rate
        if args.run != "DQN":
            raise ValueError("Only support --run DQN with __no-time")
        print("Running manual train loop without Ray Tune")
        Path(f'./agents_runs/{args.env}/{args.algo}').mkdir(parents=True, exist_ok=True)
        agent_save_path = './agents_runs/' + args.env + '/' + args.algo
        best_agent_save_path = './agents_runs/' + args.env + '/' + args.algo + '_best_agents'
        Path(best_agent_save_path).mkdir(parents=True, exist_ok=True)
        dqn_config = dqn.DEFAULT_CONFIG.copy()
        dqn_config.update(config)
        # dqn_config["lr"] = 1e-3
        # # dqn_config['num_sgd_iter'] = 30
        # # dqn_config['sgd_minibatch_size'] = 128
        # dqn_config['model']['fcnet_hiddens'] = [256, 256]
        dqn_config['model']['fcnet_activation'] = 'relu'
        # dqn_config['render_env'] = True
        # dqn_config['timesteps_per_iteration'] = 200000
        print(dqn_config)
        agent = dqn.DQNTrainer(config=dqn_config, env=ConveyorEnv_v3)
        results = []
        episode_data = []
        MAX_TRAINING_EPISODES = 100
        # TIMESTEPS_PER_EPISODE = 5400/5
        run = 1
        best_reward_cum = -10000000
        logger.debug('Start Training.')
        time_begin = time.time()
        episode_save_counter = 0
        while True:
            # print('I am in while')
            logger.info(f"Runs #: {run}")
            run += 1
            results = agent.train()
            logger.info(f"Mean Rewards: {results['episode_reward_mean']}")
            logger.info(f"Episodes this Iteration {results['episodes_this_iter']}")
            logger.info(f"Episodes total {results['episodes_total']}")
            logger.info(f"Timesteps total {results['timesteps_total']}")
            if results['episode_reward_mean'] > best_reward_cum:
                best_reward_cum = results['episode_reward_mean']
                agent.save(best_agent_save_path)
                logger.info('saved new best agent')
            if results['episodes_total'] > episode_save_counter:
                logger.info('saved new agent')
                agent.save(agent_save_path)
                episode_save_counter += 1
                logger.info("Clearing the nohup.out log file")
                os.system("> nohup.out")

            # results['timesteps_total'] >= MAX_TRAINING_EPISODES * TIMESTEPS_PER_EPISODE:
            if results['episodes_total'] > MAX_TRAINING_EPISODES:
                agent.save(agent_save_path)
                logger.info('saved last agent')
                break

        # Measure Time
        time_end = time.time()
        time_diff = time_end - time_begin
        time_diff_h = int(time_diff / 3600)
        time_diff_min = int((time_diff - time_diff_h * 3600) / 60)
        time_diff_sec = int(time_diff - time_diff_h * 3600 - time_diff_min * 60)
        logger.info(f'Training took {time_diff_h}h, {time_diff_min}m and {time_diff_sec}s.')
        logger.debug('Training successful.')
        # for n in range(2):
        #     result = trainer.train()
        #     results.append(result)
        #     print(pretty_print(result))
        #     episode = {
        #         'n': n,
        #         'episode_reward_min': result['episode_reward_min'],
        #         'episode_reward_mean': result['episode_reward_mean'],
        #         'episode_reward_max': result['episode_reward_max'],
        #         'episode_len_mean': result['episode_len_mean']}
        #
        #     episode_data.append(episode)
        #     print(
        #         f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/'
        #         f'{result["episode_reward_mean"]:8.4f}/'
        #         f'{result["episode_reward_max"]:8.4f}')
        #     check_point = trainer.save(CHECKPOINT_ROOT)
        #
        # df = pd.DataFrame(data=episode_data)
        # df.columns.tolist()
        # df.plot(x="n", y=["episode_reward_mean", "episode_reward_min", "episode_reward_max"], secondary_y=True)
        # plt.savefig('output.png')
        # episode_rewards = results[-1]['hist_stats']['episode_reward']
        # df_episode_rewards = pd.DataFrame(data={'episode': range(len(episode_rewards)), 'reward': episode_rewards})
        #
        # df_episode_rewards.plot(x="episode", y="reward")
        # plt.savefig('episode_reward.png')



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
