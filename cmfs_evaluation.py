import argparse
import platform
import logging
import getpass
import sys
sys.path.append('./conveyor_environment/snakes_master')
import matplotlib.pyplot as plt

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
    file_handler = logging.FileHandler('./logs/application-dqn_evaluation-' + timestamp + '.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


TRIAL_LIST = ['checkpoint_000054', 'checkpoint_000006', 'checkpoint_000125', 'checkpoint_000246', 'checkpoint_000284',
              'checkpoint_000290', 'checkpoint_000313', 'checkpoint_000489', 'checkpoint_000947', 'checkpoint_000952 ']
CHECKPOINT_NO = ['000050', '000082', '000100']
core_no = ['50', '82', '100']
# CHECKPOINT_NO = ['000050']
# core_no = ['50']

if platform.system() == 'Windows':
    checkpoint_path = './DQN'
else:
    checkpoint_path = './agents_runs/ConveyorEnv_v3/DQN'

Path(checkpoint_path)


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
        "num_workers": 1,  # parallelism
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
        "timesteps_per_iteration": 10000
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
        env = ConveyorEnv_v3({'version': 'full', 'final_reward': 100, 'mask': True})
        SCORE_OVERALL = []
        JOBS = []
        QUANTITY = []
        TIME_UNITS_EACH_OBJECT = []
        TOTAL_ORDER_COMPLETION_TIME = []
        AVG_ORDER_COMPLETION_TIME = []
        AVG_ORDER_THROUGHPUT = []
        AVG_TOTAL_TIME_UNITS = []
        AVG_THROUGHPUT = []
        time_begin = time.time()
        for idx, no in enumerate(CHECKPOINT_NO):
            agent = dqn.DQNTrainer(config=dqn_config, env=ConveyorEnv_v3)
            # agent.restore(f'{checkpoint_path}/checkpoint_{no}/checkpoint-{no}')
            agent.restore(
                f'agents_runs/ConveyorEnv_v3/DQN_best_agents/checkpoint_{no}/checkpoint-{core_no[idx]}')
            logger.info(f"Evaluating algo: DQN, checkpoint_nr: checkpoint_{no}")
            curr_episode = 1
            max_episode = 1
            run = 1
            best_reward_cum = -10000000
            episode_save_counter = 0
            CustomPlot.plot_figure()
            time.sleep(10)
            n = 1
            while curr_episode <= max_episode:
                # print('I am in while')
                logger.info(f"Evaluating episode: {curr_episode}")
                obs = env.reset()
                done = False
                score = 0
                step = 1
                jobs = []
                quantity = []
                time_units_each_object = []
                total_order_completion_time = []
                avg_order_completion_time = []
                avg_order_throughput = []
                avg_total_time_units = []
                avg_throughput = []
                score_episode = []
                while not done:
                    print(f'step: {step}')
                    action = agent.compute_action(obs)
                    obs, reward, done, info = env.step(action)
                    score += reward
                    step += 1
                avg_reward_per_episode = score / step
                jobs.append(info["jobs"])
                quantity.append(info["quantity"])
                time_units_each_object.append(info["time_units_each_object"])
                total_order_completion_time.append(info["total_order_completion_time"])
                avg_order_completion_time.append(info["avg_order_completion_time"])
                avg_order_throughput.append(info["avg_order_throughput"])
                avg_total_time_units.append(info["avg_total_time_units"])
                avg_throughput.append(info["avg_throughput"])
                score_episode.append(avg_reward_per_episode)
                logger.info(f"Episode_no: {n}")
                logger.info(f"Mean Rewards: {avg_reward_per_episode}")
                logger.info(f"jobs: {jobs}")
                logger.info(f"quantity: {quantity}")
                logger.info(f"time_units_each_object: {time_units_each_object}")
                logger.info(f"total_order_completion_time: {total_order_completion_time}")
                logger.info(f"avg_order_completion_time: {avg_order_completion_time}")
                logger.info(f"avg_order_throughput: {avg_order_throughput}")
                logger.info(f"avg_total_time_units: {avg_total_time_units}")
                logger.info(f"avg_throughput: {avg_throughput}")
                logger.info(f"Timesteps total: {step}")
                n += 1
                curr_episode += 1
            SCORE_OVERALL.append(score_episode)
            JOBS.append(jobs)
            QUANTITY.append(quantity)
            TIME_UNITS_EACH_OBJECT.append(time_units_each_object)
            TOTAL_ORDER_COMPLETION_TIME.append(total_order_completion_time)
            AVG_ORDER_COMPLETION_TIME.append(avg_order_completion_time)
            AVG_ORDER_THROUGHPUT.append(avg_order_throughput)
            AVG_TOTAL_TIME_UNITS.append(avg_total_time_units)
            AVG_THROUGHPUT.append(avg_throughput)

    else:
        # automated run with tune and grid search and Tensorboard
        print("Training with Ray Tune.")
        result = tune.run(args.run, config=config, stop=stop)

        if args.as_test:
            print("Checking if the learning goals are achieved")
            check_learning_achieved((result, args.stop_reward))

    average_throughput = []
    avg_rewards_per_episode = []
    for i in range(len(AVG_THROUGHPUT)):
        a = AVG_THROUGHPUT[i]
        b = SCORE_OVERALL[i]
        for j, k in a, b:
            average_throughput.append(j)
            avg_rewards_per_episode.append(k)
    plt.figure()
    plt.plot(avg_throughput)
    plt.savefig('avg_throughput.png')
    plt.plot(avg_rewards_per_episode)
    plt.savefig('rewards_overall.png')
    # Measure Time
    time_end = time.time()
    time_diff = time_end - time_begin
    time_diff_h = int(time_diff / 3600)
    time_diff_min = int((time_diff - time_diff_h * 3600) / 60)
    time_diff_sec = int(time_diff - time_diff_h * 3600 - time_diff_min * 60)
    logger.info(f'Evaluation took {time_diff_h}h, {time_diff_min}m and {time_diff_sec}s.')
    logger.debug('Evaluation Complete.')

    ray.shutdown()
