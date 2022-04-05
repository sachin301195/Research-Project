"""
With logging and everything
"""

import argparse
import platform
import importlib
import logging

import getpass
import ray
from ray import tune
from ray.rllib import agents
from util import TorchParametricActionModel, TorchParametricActionsModelv1, TorchParametricActionsModelv2
from util import CustomPlot

import numpy as np
import pandas as pd
from pathlib import Path
import time
from collections import Counter
import subprocess
import psutil
import os

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from conveyor_environment.conveyor_environment.envs.conveyor_network_v1 import ConveyorEnv_v1
from conveyor_environment.conveyor_environment.envs.conveyor_network_v0 import ConveyorEnv_v0
from conveyor_environment.conveyor_environment.envs.conveyor_network_v2 import ConveyorEnv_v2
from conveyor_environment.conveyor_environment.envs.conveyor_network_v3 import ConveyorEnv_v3
from conveyor_environment.conveyor_environment.envs.conveyor_network_token_n import ConveyorEnv_token_n

# from config import excelparser_parameter
import experiment.evaluate_experiment as ee

import random
import glob


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
THROUGHPUT_RESULTS_PATH = '/throughput-results/'
ORDER_COMPLETION_TIME_RESULTS_PATH = '/order_completion_time-results/'
REWARD_RESULTS_PATH = '/reward-results/'
CUM_REWARD_RESULTS_PATH = '/cum_reward-results/'
AVG_OVR_EP_PATH = '/avg_over_ep-results/'


def parse_args():
    """
    Parse the arguments from shell.
    :return: args object
    """

    default_base_dir = BASE_PATH

    parser = argparse.ArgumentParser()

    parser.add_argument('--base-dir',
                        type=str,
                        required=False,
                        default=default_base_dir,
                        help="BASE_DIR for agents, envs etc(Git-Repo)", )
    parser.add_argument('--env',
                        type=str,
                        required=False,
                        default='ConveyorEnv_token_n',
                        help="env to train or evaluate an agent. default=one_token",
                        choices=['ConveyorEnv_v0', 'ConveyorEnv_v1', 'ConveyorEnv_v2', 'ConveyorEnv_v3',
                                 'ConveyorEnv_v4', 'ConveyorEnv_token_n'])
    parser.add_argument('--memory',
                        type=int,
                        required=False,
                        default=10000000000,
                        help="ray.init(object_store_memory=memory), default=10gb.")
    parser.add_argument('--use-action-masking',
                        type=bool,
                        required=False,
                        default=True,
                        help="Whether to mask out wrong actions or not. default=False")
    parser.add_argument('--final-reward',
                        type=int,
                        required=False,
                        default=10,
                        help="Final reward during the completion of the process")
    parser.add_argument('--env-version',
                        type=str,
                        required=False,
                        default='full',
                        help="Version of the env to run the optimization",
                        choices=['full', 'trial'])

    subparsers = parser.add_subparsers(dest='modus',
                                       help="train, evaluate, tune, or export")

    subp_train = subparsers.add_parser('train', help='train a single or multiple agents')

    subp_train.add_argument('--algo',
                            type=str, required=False, default='DQN',
                            choices=['DQN', 'APEX', 'PPO', 'A3C', 'IMPALA', 'SAC'],
                            help="Algorithm to train the agent. default=DQN")
    subp_train.add_argument('--test-mode', type=str, required=False,
                            default='no_test', help="mode whether and when to test the agent",
                            choices=['no_test', 'during_train_test', 'after_train_test', 'both_test'])
    subp_train.add_argument('--checkpoint-nr', type=int, required=False, default=0,
                            help="Whether to continue training with given checkpoint nr or not (=0). Default=0.", )
    subp_train.add_argument('--debug_metrics', type=bool, required=False, default=False,
                            help="Print metric values inside environment. default=False")
    subp_train.add_argument('--exp-dir', type=str, required=False, default="",
                            help="This is to provide custom path to load restore point from.")

    subp_evaluate = subparsers.add_parser('evaluate', help="evaluate agents")

    subp_evaluate.add_argument('--algo',
                               type=str, action='append',
                               required=False,
                               help="Algorithm to evaluate the agent."
                                    "Choices are from 'DQN', 'APEX', 'PPO', 'A3C', 'IMPALA', 'SAC'")
    subp_evaluate.add_argument('--non-rl-algos', type=str, action='append', required=False,
                               help="List of Non-RL-controllers as baseline. Choices are from"
                                    "'genetic', 'fixed', 'random'")
    subp_evaluate.add_argument('--checkpoint-nr', type=int, required=False, action='append',
                               help="Which checkpoint-nr number from trained agents to use.", )
    subp_evaluate.add_argument('--episodes', type=int, required=False, default=1,
                               help="Number of episodes for which we want "
                                    "to run the simulation for evaluation. default = 1")
    subp_evaluate.add_argument('--verbose', type=bool, required=False, default=False,
                               help="Plots graph of each episode separately. default = False")
    subp_evaluate.add_argument('--debug_metrics', type=bool, required=False, default=False,
                               help="Print metric values inside environment. default=False")
    subp_evaluate.add_argument('--ci', type=bool, required=False, default=False,
                               help="Plots Confidence Interval of the evaluation graphs. default=False")
    subp_evaluate.add_argument('--exp-dir', type=str, required=False, default="",
                               help="Directory path to load experiment trials from.")
    subp_evaluate.add_argument('--skip-trials', type=str, action='append', required=False, default=None,
                               help="Trials to be skipped while evaluating.")
    subp_evaluate.add_argument('--exp-name', type=str, required=False, default="",
                               help="Name of the directory to be used to store evaluation results.")

    subp_tune = subparsers.add_parser('tune', help='Use this mode to do hyperparameter tuning')

    subp_tune.add_argument('--algo', type=str, required=True,
                           choices=['DQN', 'APEX', 'PPO', 'A3C', 'IMPALA', 'SAC'],
                           help="Algorithm to tune hyperparameters for.")
    subp_tune.add_argument('--checkpoint-nr', type=int, required=False, default=0,
                           help="Checkpoint to resume tuning from")
    subp_tune.add_argument('--print_info', type=bool, required=False, default=False,
                           help="Print observation, reward, done, info from environment. default=False")
    subp_tune.add_argument('--debug_metrics', type=bool, required=False, default=False,
                           help="Print metric values inside environment. default=False")
    subp_tune.add_argument('--exp-id', type=int, required=True,
                           help="Experiment Id to pick corresponding config file.")
    subp_tune.add_argument('--restore-dir', type=str, required=False, default="",
                           help="Directory path to restore experiment checkpoint from.")
    subp_tune.add_argument('--search', type=str, required=False, default="random",
                           help="To switch between grid search or random search.")
    subp_tune.add_argument('--exp-name', type=str, required=False, default="",
                           help="To add name to specific experiment for easy bifurcation.")
    subp_tune.add_argument('--ray-tune', type=bool, required=False, default=False,
                           help="Set this to True in order to use Ray's Tune framework customized for this project.")
    subp_tune.add_argument('--dr', type=bool, required=False, default=False,
                           help="Set this to True in order to train with domain randomization.")
    subp_tune.add_argument('--checkpoint-freq', type=int, required=False, default=0,
                           help="Set the frequency of checkpoint storage using this argument.")

    subp_export = subparsers.add_parser('export', help='export the model of a trained agent')
    subp_export.add_argument('--algo',
                             type=str, required=False, default='PPO',
                             choices=['DQN', 'APEX', 'PPO', 'A3C', 'IMPALA', 'SAC'],
                             help="Algorithm used for training the agent. default=PPO")
    subp_export.add_argument('--path-to-checkpoint-and-config', type=str, required=True,
                             help="The path must contain the checkpoint and config file.", )
    subp_export.add_argument('--checkpoint-nr', type=int, required=True,
                             help="Checkpoint number.")
    subp_export.add_argument('--export-dir', type=str, required=False, default='',
                             help="The path where to store the policy model.", )
    subp_export.add_argument('--debug_metrics', type=bool, required=False, default=False,
                             help="Print metric values inside environment. default=False")

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['algo_config'] = None
    if not args.modus:
        parser.print_help()
        exit(1)

    return args


def init_agent(algo: str, algo_config: dict, env, env_version: str, use_action_masking: bool = False):
    """
    :param algo: algorithm to be trained on or inferred from
    :param algo_config: configuration of algorithm and env_config according to ray[rllib]
    :param env: env class; constructor of this class must take a env_config with type dict
    :param env_version: version of the environment to run [full/trial]
    :param use_action_masking: whether to use action_masking or not
    :return: agent instance
    """
    if use_action_masking:
        register_env("env_cfms", lambda _: ConveyorEnv_token_n({'version': 'full', 'final_reward': 10, 'mask': True,
                                                                'no_of_jobs': 1}))
        if env_version == 'full':
            ModelCatalog.register_custom_model('env_cfms', TorchParametricActionsModelv2)
            print(f'version: {env_version}')
        else:
            ModelCatalog.register_custom_model('env_cfms', TorchParametricActionsModelv1)
        if algo == 'DQN':
            print('algo == DQN')
            cfg = {
                "hiddens": [],
                "dueling": False,
            }
        else:
            cfg = {}
        algo_config_actionmasking = dict({
            "env": "env_cfms",
            "model": {
                "custom_model": "env_cfms",
            },
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": "torch"
            # "num_workers": 32,  # parallelism
        },)
            # **cfg)
        print('before config')
        # print(algo_config['model'])
        algo_config = {**algo_config, **algo_config_actionmasking}
        print(algo_config)
        # algo_config["framework"] = "torch"
    else:
        ModelCatalog.register_custom_model('env_cfms', TorchParametricActionModel)
        algo_config = dict({
            "env": "env_cfms",
            "model": {
                "custom_model": "env_cfms",
            },
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            # "num_workers": 32,  # parallelism
        })
        algo_config['model'] = {**algo_config}
        algo_config["framework"] = "torch"
    config = agents.dqn.DEFAULT_CONFIG.copy()
    config.update(algo_config)

    logger.info(f'Final algo_config passed to training method: {algo_config}')
    print(config)
    print('Outside if')
    if algo == 'DQN':
        print('I came till here...............')
        agent = agents.dqn.DQNTrainer(env=env, config=config)
        print('after agent')
    elif algo == 'APEX':
        agent = agents.dqn.apex.ApexTrainer(env=env, config=algo_config)
    elif algo == 'PPO':
        agent = agents.ppo.ppo.PPOTrainer(env=env, config=algo_config)
    elif algo == 'A3C':
        agent = agents.a3c.a3c.A3CTrainer(env=env, config=algo_config)
    elif algo == 'IMPALA':
        agent = agents.impala.impala.ImpalaTrainer(env=env, config=algo_config)
    elif algo == 'SAC':
        agent = agents.sac.sac.SACTrainer(env=env, config=algo_config)

    return agent


def parse_args_and_init_agent_env(args):

    # init env
    if args.modus == 'train' or args.modus == 'tune' or args.algo_config is None:
        # create env_config
        env_config = {
            'env_config': {
                'version': 'full',
                'final_reward': 10,
                'no_of_jobs': 1,
                'mask': True,
            }
        }
        print('env_config setup...')
    else:
        algo_config = args.algo_config
        env_config = {'env_config': algo_config['env_config']}

    # env_config['env'] = 'env_cfms'
    # env_config['modus'] = args.modus
    # env_config['env_config']['env'] = args.env
    # env_config['env_config']['modus'] = args.modus
    # # if args.modus == 'evaluate':
    # #     env_config['env_config']['ext_e2'] = args.ext_e2
    # if args.modus == 'evaluate' or args.modus == 'export':
    #     env_config['env_config']['skip_worker'] = True
    # else:
    #     env_config['env_config']['skip_worker'] = False
    #
    # if args.modus != 'export':
    #     if args.debug_metrics:
    #         env_config['env_config']['print_metrics'] = True
    #     else:
    #         env_config['env_config']['print_metrics'] = False
    #
    # env_config['env_config']['analysis_logs'] = False
    # env_config['env_config']['algo'] = args.algo

    if args.modus == 'train' or args.modus == 'tune':
        if args.algo == 'DQN':
            print('setting up DQN_Config')
            algo_config = {
                "num_atoms": 1,
                "v_min": -10,
                "v_max": 10,
                "noisy": False,
                "sigma0": 0.5,
                "dueling": False,
                "hiddens": [],
                "double_q": True,
                "n_step": 2,
                "target_network_update_freq": 500,
                "exploration_config": {
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 1,
                    "final_epsilon": 0.02,
                    "epsilon_timesteps": 1000,
                    "temperature": None
                },
                "buffer_size": 50000,
                "prioritized_replay": True,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "final_prioritized_replay_beta": 0.4,
                "prioritized_replay_beta_annealing_timesteps": 20000,
                "prioritized_replay_eps": 1.00E-06,
                "compress_observation": False,
                "before_learn_on_batch": None,
                "training_intensity": None,
                "lr": 5.00E-04,
                "lr_schedule": None,
                "adam_epsilon": 1.00E-08,
                "grad_clip": 40,
                "learning_starts": 1000,
                "rollout_fragment_length": 32,
                "train_batch_size": 128,
                "num_workers": 32,
                "worker_side_prioritization": False,
                "min_iter_time_s": 30,
                "timesteps_per_iteration": 1000
            }

        # if args.algo != 'DQN' and args.algo != 'SAC':
        #     model_dict = parse_algo_parameters(excel_config_path=algo_param_path,
        #                                        param_group='model', desired_id=args.algo_id)
        #     model_dict = {'model': model_dict}
        #     algo_config = {**algo_config, **model_dict}

    if args.modus == 'evaluate' or args.modus == 'export':
        algo_config['num_workers'] = 1

    if args.modus == 'visualize':
        algo_config['num_workers'] = 1

    # init agent
    log_to_driver = True
    print('RAY_INIT_BEFORE')
    ray.init(log_to_driver=log_to_driver, object_store_memory=args.memory)
    print('RAY_INIT_AFTER')
    algo_config = {**algo_config, **env_config, 'log_level': 'INFO'}

    if args.modus == 'export':
        agent = init_agent(args.algo, algo_config, args.env, args.env_version, args.use_action_masking)
    elif args.modus != 'tune' and args.exp_dir == '':
        agent = init_agent(args.algo, algo_config, args.env, args.env_version, args.use_action_masking)
    elif args.modus == 'visualize' and args.exp_dir != '':
        agent = init_agent(args.algo, algo_config, args.env, args.env_version, args.use_action_masking)
    else:
        agent = None

    if args.modus == 'train' and args.checkpoint_nr != 0:
        logger.info(f'Use checkpoint {args.checkpoint_nr} to restore agent from.')
        if args.exp_dir != '':
            agent.restore(f'{args.exp_dir}/checkpoint_{args.checkpoint_nr}'
                          f'/checkpoint-{args.checkpoint_nr.lstrip("0")}')
    print(f'agent: {agent}')

    return agent, env_config, algo_config


def train(args):
    Path(f'./agents_runs/{args.env}/{args.algo}').mkdir(parents=True, exist_ok=True)
    agent_save_path = './agents_runs/' + args.env + '/' + args.algo
    best_agent_save_path = './agents_runs/' + args.env + '/' + args.algo + '_best_agents'
    Path(best_agent_save_path).mkdir(parents=True, exist_ok=True)
    print('before init')
    agent, _, _ = parse_args_and_init_agent_env(args)
    print('After init')
    # train agent
    MAX_TRAINING_EPISODES = 2
    # TIMESTEPS_PER_EPISODE = 5400/5
    run = 1
    best_reward_cum = -10000000
    logger.debug('Start Training.')
    time_begin = time.time()
    episode_save_counter = 0
    while True:
        print('I am in while')
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
    time_diff_min = int((time_diff - time_diff_h*3600)/60)
    time_diff_sec = int(time_diff - time_diff_h*3600 - time_diff_min*60)
    logger.info(f'Training took {time_diff_h}h, {time_diff_min}m and {time_diff_sec}s.')
    logger.debug('Training successful.')


# def evaluate(args):
#     timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
#     if args.exp_name != '':
#         timestamp = timestamp + '_' + args.exp_name
#     THROUGHPUT_PATH = RESULTS_PATH + timestamp + THROUGHPUT_RESULTS_PATH
#     ORDER_COMPLETION_TIME_PATH = RESULTS_PATH + timestamp + ORDER_COMPLETION_TIME_RESULTS_PATH
#     REWARD_PATH = RESULTS_PATH + timestamp + REWARD_RESULTS_PATH
#     CUM_REWARD_PATH = RESULTS_PATH + timestamp + CUM_REWARD_RESULTS_PATH
#     AOE_PATH = RESULTS_PATH + timestamp + AVG_OVR_EP_PATH
#
#     if args.exp_dir != '':
#         trial_info_list = ee.process_exp_checkpoints(args.exp_dir, args.skip_trials)
#         rl_algo_list = ['rl'] * len(trial_info_list)
#     else:
#         trial_info_list = []
#         rl_algo_list = None
#     non_rl_algo_list = args.non_rl_algos
#     if non_rl_algo_list is None:
#         algo_list = rl_algo_list
#     elif rl_algo_list is None:
#         algo_list = non_rl_algo_list
#     else:
#         algo_list = non_rl_algo_list + rl_algo_list
#
#     checkpoint_list = []
#     if non_rl_algo_list is not None:
#         [checkpoint_list.append(0) for i in non_rl_algo_list]
#     trial_non_rl_list = []
#     if non_rl_algo_list is not None:
#         [trial_non_rl_list.append(None) for i in non_rl_algo_list]
#     combined_trial_info_list = trial_non_rl_list + trial_info_list
#     for trial_info in trial_info_list:
#         if trial_info.best_checkpoint:
#             checkpoint_nr = f'_T{trial_info.trial}_Bst_Ch_{trial_info.checkpoint}'
#         else:
#             checkpoint_nr = f'_T{trial_info.trial}_Ch_{trial_info.checkpoint}'
#         checkpoint_list.append(checkpoint_nr)
#
#     throughput_episodes_algos = []
#     order_completion_time_episodes_algos = []
#
#     throughput_episodes_list_algos = []
#     order_completion_time_episodes_list_algos = []
#     reward_lists_algos = []
#     cum_reward_lists_algos = []
#
#     throughput_algo_summary_list = []
#     order_completion_time_algo_summary_list = []
#     reward_algo_summary_list = []
#     cum_reward_algo_summary_list = []
#
#     ma_throughput_algo_summary_list = []
#     ma_order_completion_time_algo_summary_list = []
#     ma_reward_algo_summary_list = []
#     ma_cum_reward_algo_summary_list = []
#
#     max_episode = args.episodes
#     t2 = np.arange(1., max_episode + 1)
#     time.sleep(5)
#
#     algo_list_legend = []
#     algo_node_dict = {}
#     algo_queue_proj_dict = {}
#     algo_travel_time_veh_total = {}
#     algo_travel_time_veh_avg = {}
#     for count, (algo, checkpoint_nr, trial_info)\
#             in enumerate(zip(algo_list, checkpoint_list, combined_trial_info_list)):
#         args_dict = vars(args)
#         config_file_path = None
#         if algo == 'rl':
#             config_file_path = f'{args.exp_dir}/checkpoints/exp_Trial_{trial_info.trial}/{trial_info.config_file}'
#             args_dict['algo_id'] = 0
#             args_dict['mdp_sumo_id'] = 0
#             try:
#                 algo_config = pickle.load(open(f'{config_file_path}', 'rb'))
#             except FileNotFoundError as exception:
#                 config_file_path = f'{args.exp_dir}/{trial_info.config_file}'
#                 algo_config = pickle.load(open(f'{config_file_path}', 'rb'))
#             args_dict['algo'] = algo_config['env_config']['algo']
#             args_dict['algo_config'] = algo_config
#             algo = algo_config['env_config']['algo']
#             if trial_info.best_checkpoint:
#                 checkpoint_nr = f'T{trial_info.trial}_Bst_Ch_{trial_info.checkpoint}'
#             else:
#                 checkpoint_nr = f'T{trial_info.trial}_Ch_{trial_info.checkpoint}'
#         else:
#             args_dict['algo'] = algo
#             args_dict['mdp_sumo_id'] = non_rl_algos_mdp_id_list[count]
#             args_dict['algo_config'] = None
#             args_dict['checkpoint_nr'] = 0
#             checkpoint_nr = str(0)
#
#         algo_list_legend.append(f"{args_dict['algo']}_{checkpoint_nr}")
#         logger.info(f"Evaluating algo: {args_dict['algo']}, checkpoint_nr: {checkpoint_nr}")
#
#         agent, env_config, algo_config = parse_args_and_init_agent_env(args)
#
#         if args.vc_mode:
#             env_module = importlib.import_module('envs.' + args.env + '_wvc')
#         else:
#             env_module = importlib.import_module('envs.' + args.env)
#         env = env_module.TrafficSimulation(env_config['env_config'])
#
#         if args.exp_dir != '' and args_dict['algo_config'] is not None:
#             agent = init_agent(args_dict['algo'], algo_config, env_module.TrafficSimulation,
#                                algo_config['env_config']['mdp_dict']['use_action_masking'])
#             agent.restore(f'{trial_info.path}/checkpoint-{trial_info.checkpoint.lstrip("0")}')
#             logger.info(f"Current MDP Variant: {algo_config['env_config']['mdp_dict']['mdp_variant']}")
#             logger.info(f'Current Checkpoint path: {trial_info.path}/checkpoint-{trial_info.checkpoint.lstrip("0")}')
#
#         curr_episode = 1
#
#         queue_length_episodes = []
#         wait_time_episodes = []
#         speed_episodes = []
#         pedestrian_wait_episodes = []
#         desired_ph_episodes = []
#         actual_ph_episodes = []
#         desired_ph_change_episode = []
#         actual_ph_change_episode = []
#         ki_run_count_episode = []
#         coerced_count_episode = []
#         stopping_incident_episodes = []
#         travel_time_veh_total = {}
#         travel_time_veh_avg = {}
#
#         queue_length_episodes_list = []
#         wait_time_episodes_list = []
#         speed_episodes_list = []
#         pedestrian_wait_episodes_list = []
#         reward_episodes_list = []
#         cum_reward_episodes_list = []
#         stopping_incident_episodes_list = []
#         co2_emission_episodes_list = []
#         co_emission_episodes_list = []
#         hc_emission_episodes_list = []
#         nox_emission_episodes_list = []
#         noise_emission_episodes_list = []
#         pmx_emission_episodes_list = []
#         elec_episodes_list = []
#         fuel_episodes_list = []
#
#         queue_length_episodes_std = []
#         wait_time_episodes_std = []
#         speed_episodes_std = []
#         pedestrian_wait_episodes_std = []
#         stopping_incident_episodes_std = []
#
#         node_data_episodes = NodeData()
#         node_queue_projection_episodes = NodeQueueProjection()
#
#         CustomPlot.plot_figure()
#         time.sleep(10)
#
#         while curr_episode <= max_episode:
#             logger.info(f"Evaluating episode: {curr_episode}")
#             state = None
#             init_state = None
#             state_out = None
#             num_transformers = 0
#             if 'model' in algo_config and 'use_attention' in algo_config['model']\
#                     and algo_config['model']['use_attention'] is True:
#                 num_transformers = algo_config["model"]["attention_num_transformer_units"]
#                 attention_dim = algo_config["model"]["attention_dim"]
#                 init_state = state = [np.zeros([100, attention_dim], np.float32) for _ in range(num_transformers)]
#             elif 'model' in algo_config and 'use_lstm' in algo_config['model']\
#                     and algo_config['model']['use_lstm'] is True:
#                 init_state = state = agent.get_policy().get_initial_state()
#                 prev_action = 0
#                 prev_reward = 0.
#             obs = env.reset()
#             node_data = NodeData()
#             node_queue_projection = NodeQueueProjection()
#             state = init_state
#
#             done = False
#             step = 1
#             queue_length_steps = []
#             wait_time_steps = []
#             speed_steps = []
#             pedestrian_wait_steps = []
#             reward_steps = []
#             cum_reward_steps = []
#             stopping_incident_steps = []
#             co2_emission_steps = []
#             co_emission_steps = []
#             hc_emission_steps = []
#             nox_emission_steps = []
#             noise_emission_steps = []
#             pmx_emission_steps = []
#             elec_steps = []
#             fuel_steps = []
#             while not done:
#                 if 'model' in algo_config and 'use_attention' in algo_config['model']\
#                         and algo_config['model']['use_attention'] is True:
#                     action, state_out, _ = agent.compute_action(obs, state)
#                 elif 'model' in algo_config and 'use_lstm' in algo_config['model']\
#                         and algo_config['model']['use_lstm'] is True:
#                     if algo_config['model']['lstm_use_prev_action'] is True \
#                             and algo_config['model']['lstm_use_prev_reward'] is True:
#                         action, state_out, _ = agent.compute_action(obs, state, prev_action, prev_reward)
#                     elif algo_config['model']['lstm_use_prev_reward'] is True:
#                         action, state_out, _ = agent.compute_action(obs, state, prev_reward)
#                     elif algo_config['model']['lstm_use_prev_action'] is True:
#                         action, state_out, _ = agent.compute_action(obs, state, prev_action)
#                     else:
#                         action, state_out, _ = agent.compute_action(obs, state)
#                 else:
#                     action = agent.compute_action(obs)
#                 obs, reward, done, info = env.step(action)
#
#                 if 'model' in algo_config and 'use_attention' in algo_config['model'] and algo_config['model']['use_attention'] is True:
#                     state = [np.concatenate([state[i], [state_out[i]]], axis=0)[1:] for i in range(num_transformers)]
#                 elif 'model' in algo_config and 'use_lstm' in algo_config['model']\
#                         and algo_config['model']['use_lstm'] is True:
#                     state = state_out
#                     prev_action = action
#                     prev_reward = reward
#
#                 if not args.ph_only:
#
#                     queue_length_nodes_mean = env.getQueueLengthsNodeMean()
#                     queue_length_steps.append(queue_length_nodes_mean)
#
#                     wait_time_nodes_mean = env.getWaitTimeNodeMean()
#                     wait_time_steps.append(wait_time_nodes_mean)
#
#                     speed_nodes_mean = env.getSpeedNodeMean()
#                     speed_steps.append(speed_nodes_mean)
#
#                     pedestrian_wait_nodes_mean = env.getPedestrianWaitNodeMean()
#                     pedestrian_wait_steps.append(pedestrian_wait_nodes_mean)
#
#                     reward_mean = env.getRewardMean()
#                     reward_steps.append(reward_mean)
#
#                     cum_reward_mean = env.getCumRewardMean()
#                     cum_reward_steps.append(cum_reward_mean)
#
#                     stopping_incident_sum = env.getStoppingIncidents()
#                     stopping_incident_steps.append(stopping_incident_sum)
#
#                     curr_node_data = env.getNodeData()
#
#                     co2_emission_steps.append(np.sum(curr_node_data.CO2Emission))
#                     co_emission_steps.append(np.sum(curr_node_data.COEmission))
#                     hc_emission_steps.append(np.sum(curr_node_data.HCEmission))
#                     nox_emission_steps.append(np.sum(curr_node_data.NOxEmission))
#                     noise_emission_steps.append(np.sum(curr_node_data.NoiseEmission))
#                     pmx_emission_steps.append(np.sum(curr_node_data.PMxEmission))
#                     elec_steps.append(np.sum(curr_node_data.ElectricityConsumption))
#                     fuel_steps.append(np.sum(curr_node_data.FuelConsumption))
#
#                     if node_data.lanes_in is None:
#                         node_data.lanes_in = curr_node_data.lanes_in_wo_bikes
#                         node_data.cur_wave = np.array(curr_node_data.cur_wave, dtype=float)
#                         node_data.cur_wait = np.array(curr_node_data.cur_wait, dtype=float)
#                         node_data.cur_cum_wait = np.array(curr_node_data.cur_cum_wait, dtype=float)
#                         node_data.cur_queue = np.array(curr_node_data.cur_queue, dtype=float)
#                         filtered_avg_speed = [0 if x == -1 else x for x in curr_node_data.cur_avg_speed]
#                         node_data.cur_avg_speed = np.array(filtered_avg_speed, dtype=float)
#                         bike_wait_time = np.sum(list(curr_node_data.bike_wait_times.values()))
#                         node_data.ns_ew_pedestrian_wait_time = np.array([curr_node_data.ns_pedestrian_wait_time
#                                                                          + bike_wait_time,
#                                                                          curr_node_data.ew_pedestrian_wait_time],
#                                                                         dtype=float)
#                         node_data.stopping_incidents = np.array(curr_node_data.stopping_incidents, dtype=float)
#
#                         # Emission Data
#                         node_data.CO2Emission = np.array(curr_node_data.CO2Emission, dtype=float)
#                         node_data.COEmission = np.array(curr_node_data.COEmission, dtype=float)
#                         node_data.HCEmission = np.array(curr_node_data.HCEmission, dtype=float)
#                         node_data.NOxEmission = np.array(curr_node_data.NOxEmission, dtype=float)
#                         node_data.NoiseEmission = np.array(curr_node_data.NoiseEmission, dtype=float)
#                         node_data.PMxEmission = np.array(curr_node_data.PMxEmission, dtype=float)
#
#                         # Consumption Data
#                         node_data.ElectricityConsumption = np.array(curr_node_data.ElectricityConsumption, dtype=float)
#                         node_data.FuelConsumption = np.array(curr_node_data.FuelConsumption, dtype=float)
#
#                         if args.ext_e2:
#                             node_data.lanes_in_ext = curr_node_data.lanes_in_ext
#                             node_data.cur_queue_ext = np.array(curr_node_data.cur_queue_ext, dtype=float)
#                             node_queue_projection.queue_south = [curr_node_data.cur_queue[0]
#                                                                  + curr_node_data.cur_queue[1]
#                                                                  + curr_node_data.cur_queue_ext[0]]
#                             node_queue_projection.queue_north = [curr_node_data.cur_queue[2]
#                                                                  + curr_node_data.cur_queue[3]
#                                                                  + curr_node_data.cur_queue_ext[1]
#                                                                  + curr_node_data.cur_queue_ext[2]]
#                             node_queue_projection.queue_west = [curr_node_data.cur_queue[4]
#                                                                 + curr_node_data.cur_queue[5]
#                                                                 + curr_node_data.cur_queue_ext[3]]
#                             node_queue_projection.queue_east = [curr_node_data.cur_queue[6]
#                                                                 + curr_node_data.cur_queue[7]
#                                                                 + curr_node_data.cur_queue_ext[4]]
#
#                             node_queue_projection.queue_total = [np.sum(curr_node_data.cur_queue)
#                                                                  + np.sum(curr_node_data.cur_queue_ext)]
#
#                         else:
#                             node_queue_projection.queue_south = [curr_node_data.cur_queue[0]
#                                                                  + curr_node_data.cur_queue[1]]
#                             node_queue_projection.queue_north = [curr_node_data.cur_queue[2]
#                                                                  + curr_node_data.cur_queue[3]]
#                             node_queue_projection.queue_west = [curr_node_data.cur_queue[4]
#                                                                 + curr_node_data.cur_queue[5]]
#                             node_queue_projection.queue_east = [curr_node_data.cur_queue[6]
#                                                                 + curr_node_data.cur_queue[7]]
#
#                             node_queue_projection.queue_total = [np.sum(curr_node_data.cur_queue)]
#                     else:
#                         node_data.cur_wave += np.array(curr_node_data.cur_wave)
#                         node_data.cur_wait += np.array(curr_node_data.cur_wait)
#                         node_data.cur_cum_wait += np.array(curr_node_data.cur_cum_wait)
#                         node_data.cur_queue += np.array(curr_node_data.cur_queue)
#                         filtered_avg_speed = [0 if x == -1 else x for x in curr_node_data.cur_avg_speed]
#                         node_data.cur_avg_speed += np.array(filtered_avg_speed)
#                         bike_wait_time = np.sum(list(curr_node_data.bike_wait_times.values()))
#                         node_data.ns_ew_pedestrian_wait_time += np.array([curr_node_data.ns_pedestrian_wait_time
#                                                                           + bike_wait_time,
#                                                                          curr_node_data.ew_pedestrian_wait_time])
#                         node_data.stopping_incidents += np.array(curr_node_data.stopping_incidents)
#
#                         # Emission Data
#                         node_data.CO2Emission += np.array(curr_node_data.CO2Emission)
#                         node_data.COEmission += np.array(curr_node_data.COEmission)
#                         node_data.HCEmission += np.array(curr_node_data.HCEmission)
#                         node_data.NOxEmission += np.array(curr_node_data.NOxEmission)
#                         node_data.NoiseEmission += np.array(curr_node_data.NoiseEmission)
#                         node_data.PMxEmission += np.array(curr_node_data.PMxEmission)
#
#                         # Consumption Data
#                         node_data.ElectricityConsumption += np.array(curr_node_data.ElectricityConsumption)
#                         node_data.FuelConsumption += np.array(curr_node_data.FuelConsumption)
#
#                         if args.ext_e2:
#                             node_data.cur_queue_ext += np.array(curr_node_data.cur_queue_ext, dtype=float)
#                             node_queue_projection.queue_south.append(curr_node_data.cur_queue[0]
#                                                                      + curr_node_data.cur_queue[1]
#                                                                      + curr_node_data.cur_queue_ext[0])
#                             node_queue_projection.queue_north.append(curr_node_data.cur_queue[2]
#                                                                      + curr_node_data.cur_queue[3]
#                                                                      + curr_node_data.cur_queue_ext[1]
#                                                                      + curr_node_data.cur_queue_ext[2])
#                             node_queue_projection.queue_west.append(curr_node_data.cur_queue[4]
#                                                                     + curr_node_data.cur_queue[5]
#                                                                     + curr_node_data.cur_queue_ext[3])
#                             node_queue_projection.queue_east.append(curr_node_data.cur_queue[6]
#                                                                     + curr_node_data.cur_queue[7]
#                                                                     + curr_node_data.cur_queue_ext[4])
#
#                             node_queue_projection.queue_total.append(np.sum(curr_node_data.cur_queue)
#                                                                      + np.sum(curr_node_data.cur_queue_ext))
#
#                         else:
#                             node_queue_projection.queue_south.append(curr_node_data.cur_queue[0]
#                                                                      + curr_node_data.cur_queue[1])
#                             node_queue_projection.queue_north.append(curr_node_data.cur_queue[2]
#                                                                      + curr_node_data.cur_queue[3])
#                             node_queue_projection.queue_west.append(curr_node_data.cur_queue[4]
#                                                                     + curr_node_data.cur_queue[5])
#                             node_queue_projection.queue_east.append(curr_node_data.cur_queue[6]
#                                                                     + curr_node_data.cur_queue[7])
#
#                             node_queue_projection.queue_total.append(np.sum(curr_node_data.cur_queue))
#
#                 step += 1
#
#             if not args.ph_only:
#
#                 travel_time_data = env.getTravelTimeData()
#                 # print(f'Travel Time Data: {travel_time_data}')
#                 for lane, lane_veh_data in travel_time_data.items():
#                     total_travel_time = 0
#                     for veh, travel_time in lane_veh_data.items():
#                         total_travel_time += travel_time
#                     # print(f'Lane: {lane}, Total_travel_time: {total_travel_time}')
#                     # print(f'Lane: {lane}, Veh_len: {len(lane_veh_data)}')
#                     if lane in travel_time_veh_total:
#                         travel_time_veh_total[lane].append(total_travel_time)
#                     else:
#                         travel_time_veh_total[lane] = [total_travel_time]
#                     if lane in travel_time_veh_avg:
#                         if len(lane_veh_data) != 0:
#                             travel_time_veh_avg[lane].append(total_travel_time / len(lane_veh_data))
#                         else:
#                             travel_time_veh_avg[lane].append(0)
#                     else:
#                         if len(lane_veh_data) != 0:
#                             travel_time_veh_avg[lane] = [total_travel_time / len(lane_veh_data)]
#                         else:
#                             travel_time_veh_avg[lane] = [0]
#                 env.clearTravelTimeData()
#
#                 eu.compute_episode_mean(node_data, node_data_episodes, timesteps, args.ext_e2)
#                 eu.compute_queue_episode_list(node_queue_projection, node_queue_projection_episodes)
#
#             if ((env_config['env_config']['env'] == 'owl322' or env_config['env_config']['env'] == 'meta')
#                     and max_episode < 10) or args.ph_only:
#                 desired_ph_list, actual_ph_list = env.getPhaseCounts()
#                 desired_ph_episodes += desired_ph_list
#                 actual_ph_episodes += actual_ph_list
#                 actual_phase_durations_dict = env.getActualPhaseDurationDict()
#                 for key, value in actual_phase_durations_dict.items():
#                     if key in actual_phase_durations_algos_dict:
#                         actual_phase_durations_algos_dict[key].extend(value)
#                     else:
#                         actual_phase_durations_algos_dict[key] = value
#
#                 desired_phase_durations_dict = env.getDesiredPhaseDurationDict()
#                 for key, value in desired_phase_durations_dict.items():
#                     if key in desired_phase_durations_algos_dict:
#                         desired_phase_durations_algos_dict[key].extend(value)
#                     else:
#                         desired_phase_durations_algos_dict[key] = value
#
#                 desired_ph_change_count, actual_ph_change_count, ki_run_count, coerced_count = env.getPhaseChangeCounts()
#                 desired_ph_change_episode.append(desired_ph_change_count)
#                 actual_ph_change_episode.append(actual_ph_change_count)
#                 ki_run_count_episode.append(ki_run_count)
#                 coerced_count_episode.append(coerced_count)
#
#                 if args.verbose:
#                     PH_DETAILS_V_PATH = f'{RESULTS_PATH + timestamp}/Phase_details/{algo+checkpoint_nr}'
#                     Path(PH_DETAILS_V_PATH).mkdir(parents=True, exist_ok=True)
#                     ph_v_file = f'{PH_DETAILS_V_PATH}/Ph_Details-{str(curr_episode)}.txt'
#                     with open(ph_v_file, "w") as text_file:
#                         print(f"Desired Phase : {sorted(Counter(desired_ph_list).items())}", file=text_file)
#                         print(f"Actual Phase: {sorted(Counter(actual_ph_list).items())}", file=text_file)
#                         print(f"Desired Phase Change Count: {desired_ph_change_count}", file=text_file)
#                         print(f"Actual Phase Change Count: {actual_ph_change_count}", file=text_file)
#                         print(f"KI_Run count: {ki_run_count}", file=text_file)
#                         print(f"Coerced count: {coerced_count}", file=text_file)
#                         for key, value in actual_phase_durations_dict.items():
#                             print(f"Actual Phase: {key}, Min_Duration: {np.min(value)},"
#                                   f" Mean_Duration: {np.mean(value)}, Median_Duration: {np.median(value)},"
#                                   f" Max_Duration: {np.max(value)}", file=text_file)
#                         for key, value in desired_phase_durations_dict.items():
#                             print(f"Desired Phase: {key}, Min_Duration: {np.min(value)},"
#                                   f" Mean_Duration: {np.mean(value)}, Median_Duration: {np.median(value)},"
#                                   f" Max_Duration: {np.max(value)}", file=text_file)
#
#             if not args.ph_only:
#                 queue_length_steps_mean = np.mean(queue_length_steps)
#                 queue_length_episodes.append(queue_length_steps_mean)
#                 queue_length_episodes_list.append(queue_length_steps)
#
#                 wait_time_steps_mean = np.mean(wait_time_steps)
#                 wait_time_episodes.append(wait_time_steps_mean)
#                 wait_time_episodes_list.append(wait_time_steps)
#
#                 speed_steps_mean = np.mean(speed_steps)
#                 speed_episodes.append(speed_steps_mean)
#                 speed_episodes_list.append(speed_steps)
#
#                 pedestrian_wait_mean = np.mean(pedestrian_wait_steps)
#                 pedestrian_wait_episodes.append(pedestrian_wait_mean)
#                 pedestrian_wait_episodes_list.append(pedestrian_wait_steps)
#
#                 reward_episodes_list.append(reward_steps)
#                 cum_reward_episodes_list.append(cum_reward_steps)
#
#                 stopping_inc_mean = np.mean(stopping_incident_steps)
#                 stopping_incident_episodes.append(stopping_inc_mean)
#                 stopping_incident_episodes_list.append(stopping_incident_steps)
#
#                 co2_emission_episodes_list.append(co2_emission_steps)
#                 co_emission_episodes_list.append(co_emission_steps)
#                 hc_emission_episodes_list.append(hc_emission_steps)
#                 nox_emission_episodes_list.append(nox_emission_steps)
#                 noise_emission_episodes_list.append(noise_emission_steps)
#                 pmx_emission_episodes_list.append(pmx_emission_steps)
#                 elec_episodes_list.append(elec_steps)
#                 fuel_episodes_list.append(fuel_steps)
#
#                 queue_length_steps_std = np.std(queue_length_steps)
#                 queue_length_episodes_std.append(queue_length_steps_std)
#
#                 wait_time_steps_std = np.std(wait_time_steps)
#                 wait_time_episodes_std.append(wait_time_steps_std)
#
#                 speed_steps_std = np.std(speed_steps)
#                 speed_episodes_std.append(speed_steps_std)
#
#                 pedestrian_wait_std = np.std(pedestrian_wait_steps)
#                 pedestrian_wait_episodes_std.append(pedestrian_wait_std)
#
#                 stopping_incidents_std = np.std(stopping_incident_steps)
#                 stopping_incident_episodes_std.append(stopping_incidents_std)
#
#                 if args.verbose:
#
#                     t = np.arange(0., len(queue_length_steps))
#
#                     CustomPlot.save_plot(f'{QUEUE_PATH}Queue-Length_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
#                                          'Simulation time (sec)', 'Queue Length (m)', f'Episode :{str(curr_episode)}',
#                                          t, queue_length_steps, args.algo+checkpoint_nr)
#
#                     CustomPlot.save_plot(f'{WAIT_PATH}Wait-Time_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
#                                          'Simulation time (sec)', 'Vehicle Wait Time (s)', f'Episode :{str(curr_episode)}',
#                                          t, wait_time_steps, args.algo+checkpoint_nr)
#
#                     CustomPlot.save_plot(f'{SPEED_PATH}Speed_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
#                                          'Simulation time (sec)', 'Speed (m/s)', f'Episode :{str(curr_episode)}',
#                                          t, speed_steps, args.algo+checkpoint_nr)
#
#                     CustomPlot.save_plot(f'{PEDESTRIAN_PATH}Pedestrian_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
#                                          'Simulation time (sec)', 'Pedestrian Wait Time (s)', f'Episode:{str(curr_episode)}',
#                                          t, pedestrian_wait_steps, args.algo+checkpoint_nr)
#
#                     CustomPlot.save_plot(f'{REWARD_PATH}Reward_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
#                                          'Simulation time (sec)', 'Reward', f'Episode:{str(curr_episode)}',
#                                          t, reward_steps, args.algo + checkpoint_nr)
#
#                     CustomPlot.save_plot(f'{STOPPING_INC_PATH}Stopping_Inc{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
#                                          'Simulation time (sec)', 'Stopping Inc.', f'Episode:{str(curr_episode)}',
#                                          t, stopping_incident_steps, args.algo + checkpoint_nr)
#
#                     CustomPlot.save_plot(f'{CO2EMISSION_PATH}CO2Emission{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
#                         'Simulation time (sec)', 'CO2 Emission', f'Episode:{str(curr_episode)}',
#                         t, co2_emission_steps, args.algo + checkpoint_nr)
#
#                     CustomPlot.save_plot(f'{CO2EMISSION_PATH}Fuel_Consumption{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
#                         'Simulation time (sec)', 'Fuel Consumption', f'Episode:{str(curr_episode)}',
#                         t, fuel_steps, args.algo + checkpoint_nr)
#
#             curr_episode += 1
#
#         if env_config['env_config']['env'] == 'owl322' or env_config['env_config']['env'] == 'meta':
#             PH_DETAILS_PATH = f'{RESULTS_PATH + timestamp}/Phase_details/{algo+checkpoint_nr}'
#             Path(PH_DETAILS_PATH).mkdir(parents=True, exist_ok=True)
#             ph_file = f'{PH_DETAILS_PATH}/Ph_Details-{algo+checkpoint_nr}.txt'
#             with open(ph_file, "w") as text_file:
#                 print(f"Desired Phase : {sorted(Counter(desired_ph_episodes).items())}", file=text_file)
#                 print(f"Actual Phase: {sorted(Counter(actual_ph_episodes).items())}", file=text_file)
#                 print(f"Desired Phase Change Count: {np.sum(desired_ph_change_episode)}", file=text_file)
#                 print(f"Actual Phase Change Count: {np.sum(actual_ph_change_episode)}", file=text_file)
#                 print(f"KI_Run Count: {np.sum(ki_run_count_episode)}", file=text_file)
#                 print(f"Coercion Count: {np.sum(coerced_count_episode)}", file=text_file)
#                 for key, value in actual_phase_durations_algos_dict.items():
#                     print(f"Total: Actual Phase: {key}, Min_Duration: {np.min(value)},"
#                           f" Mean_Duration: {np.mean(value)}, Median_Duration: {np.median(value)},"
#                           f" Max_Duration: {np.max(value)}", file=text_file)
#                 for key, value in desired_phase_durations_algos_dict.items():
#                     print(f"Total: Desired Phase: {key}, Min_Duration: {np.min(value)},"
#                           f" Mean_Duration: {np.mean(value)}, Median_Duration: {np.median(value)},"
#                           f" Max_Duration: {np.max(value)}", file=text_file)
#
#         t = np.arange(0., timesteps)
#
#         if not args.ph_only:
#
#             eu.compute_all_ep_mean(node_data_episodes, max_episode, args.ext_e2)
#             algo_node_dict[algo+'_'+checkpoint_nr] = node_data_episodes
#             algo_travel_time_veh_total[algo+'_'+checkpoint_nr] = travel_time_veh_total
#             algo_travel_time_veh_avg[algo+'_'+checkpoint_nr] = travel_time_veh_avg
#             algo_queue_proj_dict[algo+'_'+checkpoint_nr] = node_queue_projection_episodes
#
#             if args.ci:
#                 queue_length_episodes_std = np.array(queue_length_episodes_std)
#                 queue_length_episodes = np.array(queue_length_episodes)
#                 std_err_que = queue_length_episodes_std / np.sqrt(timesteps)
#                 std_err_que *= 1.96
#                 CustomPlot.save_ci_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_CI_{args.algo}_{checkpoint_nr}.png',
#                                         'Episode', 'Queue Length (m)', 'Queue Length over Episodes',
#                                         t2, queue_length_episodes, queue_length_episodes-std_err_que,
#                                         queue_length_episodes+std_err_que, algo+checkpoint_nr)
#                 std_err_que_algos.append(std_err_que)
#
#                 ma_queue_mean, ma_queue_std = eu.plot_moving_average_over_episode(queue_length_episodes_list, 'Queue Length',
#                                                     QUEUE_PATH, args.algo, checkpoint_nr,
#                                                     queue_length_episodes_list_algos, std_err_que_list_algos)
#
#                 queue_mean, queue_std = eu.compute_metric_mean_std(queue_length_episodes_list)
#                 queue_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Queue Length', queue_mean, queue_std))
#                 ma_queue_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Queue Length', ma_queue_mean, ma_queue_std))
#
#             else:
#                 CustomPlot.save_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_{args.algo}_{checkpoint_nr}.png',
#                                      'Episode', 'Queue Length (m)', 'Queue Length over Episodes',
#                                      t2, queue_length_episodes, algo+checkpoint_nr)
#
#             queue_length_episodes_algos.append(queue_length_episodes)
#
#             if args.ci:
#                 wait_time_episodes_std = np.array(wait_time_episodes_std)
#                 wait_time_episodes = np.array(wait_time_episodes)
#                 std_err_wait = wait_time_episodes_std / np.sqrt(timesteps)
#                 std_err_wait *= 1.96
#                 CustomPlot.save_ci_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_CI_{args.algo}_{checkpoint_nr}.png',
#                                         'Episode', 'Vehicle Wait Time (s)', 'Wait Time over Episodes',
#                                         t2, wait_time_episodes, wait_time_episodes-std_err_wait, wait_time_episodes
#                                         + std_err_wait, algo+checkpoint_nr)
#                 std_err_wait_algos.append(std_err_wait)
#
#                 ma_wait_mean, ma_wait_std = eu.plot_moving_average_over_episode(wait_time_episodes_list, 'Vehicle Wait Time',
#                                                     WAIT_PATH, args.algo, checkpoint_nr,
#                                                     wait_time_episodes_list_algos, std_err_wait_list_algos)
#
#                 wait_mean, wait_std = eu.compute_metric_mean_std(wait_time_episodes_list)
#                 wait_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Vehicle Wait Time', wait_mean, wait_std))
#                 ma_wait_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Vehicle Wait Time', ma_wait_mean, ma_wait_std))
#             else:
#                 CustomPlot.save_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_{args.algo}_{checkpoint_nr}.png',
#                                      'Episode', 'Vehicle Wait Time (s)', 'Wait Time over Episodes',
#                                      t2, wait_time_episodes, algo+checkpoint_nr)
#
#             wait_time_episodes_algos.append(wait_time_episodes)
#
#             CustomPlot.save_plot(f'{SPEED_PATH}Avg-Speed-Over_Episodes_{args.algo}_{checkpoint_nr}.png',
#                                  'Episode', 'Speed (m/s)', 'Avg. Speed over Episodes',
#                                  t2, speed_episodes, algo+checkpoint_nr)
#
#             speed_episodes_algos.append(speed_episodes)
#
#             if args.ci:
#                 ma_speed_mean, ma_speed_std = eu.plot_moving_average_over_episode(speed_episodes_list, 'Speed',
#                                                     SPEED_PATH, args.algo, checkpoint_nr,
#                                                     speed_episodes_list_algos, std_err_avg_speed_algos)
#
#                 speed_mean, speed_std = eu.compute_metric_mean_std(speed_episodes_list)
#                 speed_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Speed', speed_mean, speed_std))
#                 ma_speed_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Speed', ma_speed_mean, ma_speed_std))
#
#             if args.ci:
#                 pedestrian_wait_episodes_std = np.array(pedestrian_wait_episodes_std)
#                 pedestrian_wait_episodes = np.array(pedestrian_wait_episodes)
#                 std_err_pedestrian_wait = pedestrian_wait_episodes_std / np.sqrt(timesteps)
#                 std_err_pedestrian_wait *= 1.96
#                 CustomPlot.save_ci_plot(f'{PEDESTRIAN_PATH}Pedestrian_Wait_time-Over_Episodes_CI_{args.algo}_{checkpoint_nr}.png',
#                                         'Episode', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time over Episodes',
#                                         t2, pedestrian_wait_episodes, pedestrian_wait_episodes-std_err_pedestrian_wait, pedestrian_wait_episodes
#                                         + std_err_pedestrian_wait, algo+checkpoint_nr)
#                 std_err_pedestrian_algos.append(std_err_pedestrian_wait)
#
#                 ma_ped_wait_mean, ma_ped_wait_std = eu.plot_moving_average_over_episode(pedestrian_wait_episodes_list, 'Pedestrian Wait Time',
#                                                     PEDESTRIAN_PATH, args.algo, checkpoint_nr,
#                                                     pedestrian_episodes_list_algos, std_err_pedestrian_list_algos)
#
#                 ped_wait_mean, ped_wait_std = eu.compute_metric_mean_std(pedestrian_wait_episodes_list)
#                 ped_wait_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Pedestrian Wait Time', ped_wait_mean, ped_wait_std))
#                 ma_ped_wait_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Pedestrian Wait Time', ma_ped_wait_mean, ma_ped_wait_std))
#
#             else:
#                 CustomPlot.save_plot(f'{PEDESTRIAN_PATH}Pedestrian_Wait_time-Over_Episodes_{args.algo}_{checkpoint_nr}.png',
#                                      'Episode', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time over Episodes',
#                                      t2, pedestrian_wait_episodes, algo+checkpoint_nr)
#
#             pedestrian_episodes_algos.append(pedestrian_wait_episodes)
#
#             if args.ci:
#                 ma_reward_mean, ma_reward_std = eu.plot_moving_average_over_episode(reward_episodes_list, 'Reward',
#                                                     REWARD_PATH, args.algo, checkpoint_nr,
#                                                     reward_lists_algos, std_err_reward_lists_algos)
#
#                 reward_mean, reward_std = eu.compute_metric_mean_std(reward_episodes_list)
#                 reward_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Reward', reward_mean, reward_std))
#                 ma_reward_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Reward', ma_reward_mean, ma_reward_std))
#
#                 ma_cum_reward_mean, ma_cum_reward_std = eu.plot_moving_average_over_episode(cum_reward_episodes_list, 'Cummulative Reward',
#                                                     CUM_REWARD_PATH, args.algo, checkpoint_nr,
#                                                     cum_reward_lists_algos, std_err_cum_reward_lists_algos)
#
#                 cum_reward_mean, cum_reward_std = eu.compute_metric_mean_std(cum_reward_episodes_list)
#                 cum_reward_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Reward', cum_reward_mean, cum_reward_std))
#                 ma_cum_reward_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Reward', ma_cum_reward_mean, ma_cum_reward_std))
#
#             if args.ci:
#                 stopping_incident_episodes_std = np.array(stopping_incident_episodes_std)
#                 stopping_incident_episodes = np.array(stopping_incident_episodes)
#                 std_err_si = stopping_incident_episodes_std / np.sqrt(timesteps)
#                 std_err_si *= 1.96
#                 CustomPlot.save_ci_plot(f'{STOPPING_INC_PATH}Stopping_Inc-Over_Episodes_CI_{args.algo}_{checkpoint_nr}.png',
#                                         'Episode', 'Stopping Incident', 'Stopping Incident over Episodes',
#                                         t2, stopping_incident_episodes, stopping_incident_episodes-std_err_si,
#                                         stopping_incident_episodes+std_err_si, algo+checkpoint_nr)
#                 std_err_stopping_incident_algos.append(std_err_si)
#
#                 ma_si_mean, ma_si_std = eu.plot_moving_average_over_episode(stopping_incident_episodes_list, 'Stopping Incident',
#                                                     STOPPING_INC_PATH, args.algo, checkpoint_nr,
#                                                     stopping_incident_episodes_list_algos, std_err_stopping_incident_episodes_list_algos)
#
#                 si_mean, si_std = eu.compute_metric_sum_std(stopping_incident_episodes_list)
#                 stopping_incident_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Stopping Incident', si_mean, si_std))
#                 ma_stopping_incident_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Stopping Incident', ma_si_mean, ma_si_std))
#
#             else:
#                 CustomPlot.save_plot(f'{STOPPING_INC_PATH}Stopping_Inc-Over_Episodes_{args.algo}_{checkpoint_nr}.png',
#                                      'Episode', 'Stopping Incident', 'Stopping Incident over Episodes',
#                                      t2, stopping_incident_episodes, algo+checkpoint_nr)
#
#             stopping_incident_episodes_algos.append(stopping_incident_episodes)
#
#             if args.ci:
#                 ma_co2_mean, ma_co2_std = eu.plot_moving_average_over_episode(co2_emission_episodes_list,
#                                                                             'CO2 Emission',
#                                                                             CO2EMISSION_PATH, args.algo, checkpoint_nr,
#                                                                             co2_emission_episodes_list_algos,
#                                                                             std_err_co2_emission_episodes_list_algos)
#
#                 co2_mean, co2_std = eu.compute_metric_mean_std(co2_emission_episodes_list)
#                 CO2Emission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'CO2 Emission', co2_mean, co2_std))
#                 ma_CO2Emission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'CO2 Emission', ma_co2_mean, ma_co2_std))
#
#                 co_mean, co_std = eu.compute_metric_mean_std(co_emission_episodes_list)
#                 # The following method returns list of values averaged over episode.
#                 # The list size is 4200 as per maximum steps of the episode.
#                 # Due to this, the value ma_co_mean has list which is again averaged to get episodic value.
#                 ma_co_mean, _, _ = eu.compute_moving_avg_std_ovr_ep(co_emission_episodes_list)
#                 COEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'CO Emission', co_mean, co_std))
#                 ma_COEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'CO Emission', np.mean(ma_co_mean), np.std(ma_co_mean)))
#
#                 hc_mean, hc_std = eu.compute_metric_mean_std(hc_emission_episodes_list)
#                 ma_hc_mean, _, _ = eu.compute_moving_avg_std_ovr_ep(hc_emission_episodes_list)
#                 HCEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'HC Emission', hc_mean, hc_std))
#                 ma_HCEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'HC Emission', np.mean(ma_hc_mean), np.std(ma_hc_mean)))
#
#                 nox_mean, nox_std = eu.compute_metric_mean_std(nox_emission_episodes_list)
#                 ma_nox_mean, _, _ = eu.compute_moving_avg_std_ovr_ep(nox_emission_episodes_list)
#                 NOxEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'NOx Emission', nox_mean, nox_std))
#                 ma_NOxEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'NOx Emission', np.mean(ma_nox_mean), np.std(ma_nox_mean)))
#
#                 noise_mean, noise_std = eu.compute_metric_mean_std(noise_emission_episodes_list)
#                 ma_noise_mean, _, _ = eu.compute_moving_avg_std_ovr_ep(noise_emission_episodes_list)
#                 NoiseEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'Noise Emission', noise_mean, noise_std))
#                 ma_NoiseEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'Noise Emission', np.mean(ma_noise_mean),
#                                    np.std(ma_noise_mean)))
#
#                 pmx_mean, pmx_std = eu.compute_metric_mean_std(pmx_emission_episodes_list)
#                 ma_pmx_mean, _, _ = eu.compute_moving_avg_std_ovr_ep(pmx_emission_episodes_list)
#                 PMxEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'PMx Emission', pmx_mean, pmx_std))
#                 ma_PMxEmission_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'PMx Emission', np.mean(ma_pmx_mean), np.std(ma_pmx_mean)))
#
#                 elec_mean, elec_std = eu.compute_metric_mean_std(elec_episodes_list)
#                 ma_elec_mean, _, _ = eu.compute_moving_avg_std_ovr_ep(elec_episodes_list)
#                 ElectricityConsumption_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'Electricity Consumption', elec_mean, elec_std))
#                 ma_ElectricityConsumption_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'Electricity Consumption', np.mean(ma_elec_mean),
#                                    np.std(ma_elec_mean)))
#
#                 ma_fuel_mean, ma_fuel_std = eu.plot_moving_average_over_episode(fuel_episodes_list,
#                                                                               'Fuel Consumption',
#                                                                               CO2EMISSION_PATH, args.algo,
#                                                                               checkpoint_nr,
#                                                                               fuel_episodes_list_algos,
#                                                                               std_err_fuel_episodes_list_algos)
#                 fuel_mean, fuel_std = eu.compute_metric_mean_std(fuel_episodes_list)
#                 FuelConsumption_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'Fuel Consumption', fuel_mean, fuel_std))
#                 ma_FuelConsumption_summary_list.append(
#                     eu.AlgoSummary(args.algo, checkpoint_nr, 'Fuel Consumption', ma_fuel_mean, ma_fuel_std))
#
#         ray.shutdown()
#
#     if not args.ph_only:
#
#         metric_summary_dict = {'queue': queue_algo_summary_list, 'wait_time': wait_algo_summary_list,
#                                'ped_wait_time': ped_wait_algo_summary_list, 'speed': speed_algo_summary_list,
#                                'reward': reward_algo_summary_list, 'cum_reward': cum_reward_algo_summary_list,
#                                'stopping_incident': stopping_incident_summary_list,
#                                'algo_travel_time_veh_total': algo_travel_time_veh_total,
#                                'algo_travel_time_veh_avg': algo_travel_time_veh_avg,
#                                'co2': CO2Emission_summary_list,
#                                'co': COEmission_summary_list,
#                                'hc': HCEmission_summary_list,
#                                'nox': NOxEmission_summary_list,
#                                'noise': NoiseEmission_summary_list,
#                                'pmx': PMxEmission_summary_list,
#                                'elec': ElectricityConsumption_summary_list,
#                                'fuel': FuelConsumption_summary_list}
#
#         if args.exp_name != '':
#             eu.save_summary_table(RESULTS_PATH + timestamp, metric_summary_dict, ma=False, file_suffix=args.exp_name)
#         else:
#             eu.save_summary_table(RESULTS_PATH + timestamp, metric_summary_dict)
#
#         ma_metric_summary_dict = {'queue': ma_queue_algo_summary_list, 'wait_time': ma_wait_algo_summary_list,
#                                   'ped_wait_time': ma_ped_wait_algo_summary_list, 'speed': ma_speed_algo_summary_list,
#                                   'reward': ma_reward_algo_summary_list, 'cum_reward': ma_cum_reward_algo_summary_list,
#                                   'stopping_incident': ma_stopping_incident_summary_list,
#                                   'algo_travel_time_veh_total': algo_travel_time_veh_total,
#                                   'algo_travel_time_veh_avg': algo_travel_time_veh_avg,
#                                   'co2': ma_CO2Emission_summary_list,
#                                   'co': ma_COEmission_summary_list,
#                                   'hc': ma_HCEmission_summary_list,
#                                   'nox': ma_NOxEmission_summary_list,
#                                   'noise': ma_NoiseEmission_summary_list,
#                                   'pmx': ma_PMxEmission_summary_list,
#                                   'elec': ma_ElectricityConsumption_summary_list,
#                                   'fuel': ma_FuelConsumption_summary_list}
#
#         if args.exp_name != '':
#             eu.save_summary_table(RESULTS_PATH + timestamp, ma_metric_summary_dict, ma=True, file_suffix=args.exp_name)
#         else:
#             eu.save_summary_table(RESULTS_PATH + timestamp, ma_metric_summary_dict, ma=True)
#
#         if args.exp_name != '':
#             eu.save_detailed_report_table(RESULTS_PATH + timestamp, algo_node_dict, algo_travel_time_veh_total,
#                                           algo_travel_time_veh_avg, args.ext_e2, args.exp_name)
#         else:
#             eu.save_detailed_report_table(RESULTS_PATH + timestamp, algo_node_dict, algo_travel_time_veh_total,
#                                       algo_travel_time_veh_avg, args.ext_e2)
#
#     t = np.arange(0., timesteps)
#
#     if not args.ph_only:
#
#         eu.plot_queue_projections(QUEUE_PROJ_PATH, algo_queue_proj_dict)
#
#         algo_list = algo_list_legend
#         if args.ci:
#             CustomPlot.save_combined_ci_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_CI_Combined.png',
#                                              'Episode', 'Queue Length (m)', 'Queue Length over Episodes',
#                                              t2, queue_length_episodes_algos, algo_list, checkpoint_list, std_err_que_algos)
#
#             CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Queue-Length-Avg_Over_Episodes_CI_Combined.png',
#                                              'Simulation time (s)', 'Queue Length (m)', 'Queue Length over Timesteps',
#                                              t, queue_length_episodes_list_algos, algo_list, checkpoint_list,
#                                              std_err_que_list_algos)
#
#         else:
#             CustomPlot.save_combined_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_Combined.png',
#                                           'Episode', 'Queue Length (m)', 'Queue Length over Episodes',
#                                           t2, queue_length_episodes_algos, algo_list, checkpoint_list)
#
#         if args.ci:
#             CustomPlot.save_combined_ci_plot(f'{STOPPING_INC_PATH}Stopping_Incident-Over_Episodes_CI_Combined.png',
#                                              'Episode', 'Stopping Incident', 'Stopping Incident over Episodes',
#                                              t2, stopping_incident_episodes_algos, algo_list, checkpoint_list, std_err_stopping_incident_algos)
#
#             CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Stopping_Incident-Avg_Over_Episodes_CI_Combined.png',
#                                              'Simulation time (s)', 'Stopping Incident', 'Stopping Incident over Timesteps',
#                                              t, stopping_incident_episodes_list_algos, algo_list, checkpoint_list,
#                                              std_err_stopping_incident_episodes_list_algos)
#
#         else:
#             CustomPlot.save_combined_plot(f'{STOPPING_INC_PATH}Stopping_Incident-Over_Episodes_Combined.png',
#                                           'Episode', 'Stopping Incident', 'Stopping Incident over Episodes',
#                                           t2, stopping_incident_episodes_algos, algo_list, checkpoint_list)
#
#         if args.ci:
#             CustomPlot.save_combined_ci_plot(f'{WAIT_PATH}Wait_time-Over-Episodes_CI_Combined.png',
#                                              'Episode', 'Vehicle Wait Time (s)', 'Wait Time over Episodes',
#                                              t2, wait_time_episodes_algos, algo_list, checkpoint_list, std_err_wait_algos)
#
#             CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Wait_time-Avg-Over-Episodes_CI_Combined.png',
#                                              'Simulation time (s)', 'Vehicle Wait Time (s)', 'Wait Time over Timesteps',
#                                              t, wait_time_episodes_list_algos, algo_list, checkpoint_list,
#                                              std_err_wait_list_algos)
#         else:
#             CustomPlot.save_combined_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_Combined.png',
#                                           'Episode', 'Vehicle Wait Time (s)', 'Wait Time over Episodes',
#                                           t2, wait_time_episodes_algos, algo_list, checkpoint_list)
#
#         CustomPlot.save_combined_plot(f'{SPEED_PATH}Avg-Speed-Over_Episodes_Combined.png',
#                                       'Episode', 'Speed (m/s)', 'Avg. Speed over Episodes',
#                                       t2, speed_episodes_algos, algo_list, checkpoint_list)
#
#         if args.ci:
#             CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Speed-Avg-Over-Episodes_CI_Combined.png',
#                                              'Simulation time (s)', 'Speed (m/s)', 'Avg. Speed over Timesteps',
#                                              t, speed_episodes_list_algos, algo_list, checkpoint_list,
#                                              std_err_avg_speed_algos)
#
#         if args.ci:
#             CustomPlot.save_combined_ci_plot(f'{PEDESTRIAN_PATH}Pedestrian-Wait-Time-Over_Episodes_CI_Combined.png',
#                                              'Episode', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time Over Episodes',
#                                              t2, pedestrian_episodes_algos, algo_list, checkpoint_list, std_err_pedestrian_algos)
#
#             CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Pedestrian-Wait_time-Avg-Over-Episodes_CI_Combined.png',
#                                              'Simulation time (s)', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time over Timesteps',
#                                              t, pedestrian_episodes_list_algos, algo_list, checkpoint_list,
#                                              std_err_pedestrian_list_algos)
#
#             # CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Pedestrian-Wait_time-Avg-Over-Episodes_CI_Combined_WO_L.png',
#             #                                  'Simulation time (s)', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time over Timesteps',
#             #                                  t, pedestrian_episodes_list_algos[0:-1], algo_list[0:-1], checkpoint_list[0:-1],
#             #                                  std_err_pedestrian_list_algos[0:-1])
#
#         else:
#             CustomPlot.save_combined_plot(f'{PEDESTRIAN_PATH}Pedestrian-Wait-Time-Over_Episodes_Combined.png',
#                                           'Episode', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time Over Episodes',
#                                           t2, pedestrian_episodes_algos, algo_list, checkpoint_list)
#
#         CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Reward-Avg-Over-Episodes_CI_Combined.png',
#                                          'Simulation time (s)', 'Reward', 'Reward over Timesteps',
#                                          t, reward_lists_algos, algo_list, checkpoint_list, std_err_reward_lists_algos)
#
#         CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Cum_Reward-Avg-Over-Episodes_CI_Combined.png',
#                                          'Simulation time (s)', 'Cum Reward', 'Reward over Timesteps',
#                                          t, cum_reward_lists_algos, algo_list, checkpoint_list, std_err_cum_reward_lists_algos)
#
#         CustomPlot.save_combined_ci_plot(f'{AOE_PATH}CO2-Avg-Over-Episodes_CI_Combined.png',
#                                          'Simulation time (s)', 'CO2 Emission', 'CO2 Emission over Timesteps',
#                                          t, co2_emission_episodes_list_algos, algo_list, checkpoint_list,
#                                          std_err_co2_emission_episodes_list_algos)
#
#         CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Fuel-Avg-Over-Episodes_CI_Combined.png',
#                                          'Simulation time (s)', 'Fuel Consumption', 'Fuel Consumption over Timesteps',
#                                          t, fuel_episodes_list_algos, algo_list, checkpoint_list,
#                                          std_err_fuel_episodes_list_algos)
#
#         CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Stopping_Incident-Avg_Over_Episodes_CI_Combined.png',
#                                          'Simulation time (s)', 'Stopping Incident', 'Stopping Incident over Timesteps',
#                                          t, stopping_incident_episodes_list_algos, algo_list, checkpoint_list,
#                                          std_err_stopping_incident_episodes_list_algos)
#
#         final_results_dict = {'queue': queue_length_episodes_list_algos,
#                               'wait_time': wait_time_episodes_list_algos,
#                               'speed': speed_episodes_list_algos,
#                               'ped_wait_time': pedestrian_episodes_list_algos,
#                               'reward': reward_lists_algos,
#                               'cum_reward': cum_reward_lists_algos,
#                               'stopping_incident': stopping_incident_episodes_list_algos,
#                               'co2': co2_emission_episodes_list_algos,
#                               'algos': algo_list,
#                               'checkpoints': checkpoint_list}
#
#         with open(f'{AOE_PATH}/final_results_dict.p', 'wb') as f:
#             pickle.dump(final_results_dict, f)


if __name__ == '__main__':
    try:
        args = parse_args()
        time_begin = time.time()
        if args.modus == 'train':
            train(args)
        # elif args.modus == 'evaluate':
        #     evaluate(args)
        # elif args.modus == 'tune':
        #     if args.ray_tune:
        #         args_dict = vars(args)
        #         args_dict['lisa_replicator'] = True
        #         ray_tune(args)
        #     else:
        #         tune(args)
        # elif args.modus == 'export':
        #     export(args)
        # else:
        #     visualize(args)
    except Exception as e:
        logger.exception(e)
        exit(-1)
    finally:
        time_end = time.time()
        logger.info(f"Time Difference: {time_end - time_begin}")
        logger.info("Shutting down the system..")

