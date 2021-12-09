import argparse
import gym
from gym import spaces
import numpy as np
import os
import shutil
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

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="dqn",
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
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
         "use DQN without grid search and no TensorBoard.")
parser.add_argument(
    "--local-mode",
    help="Init Ray in local mode for easier debugging.",
    action="store-true"
)


class Torch_Network(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, true_obs_shape=(40,),
                 action_embed_size=82, *args, **kwargs):
        super(Torch_Network, self).__init__(obs_space, action_space, num_outputs, model_config, name,
                                            *args, **kwargs)
        self.action_embed_model = TorchFC(spaces.Box(0, 1, shape=true_obs_shape), action_space,
                                          action_embed_size, model_config, name + '_action_embedding')
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({"obs": input_dict["obs"]["state"]})
        intent_vector = torch.expand(action_embedding, 1)
        action_logits = torch.sum(avail_actions * intent_vector, 1)
        inf_mask = torch.maximum(torch.log(action_mask), torch.float32.min)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


CHECKPOINT_ROOT = "tmp/dqn/ConveyorEnv"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    ModelCatalog.register_custom_model(
        "conveyor_mask", Torch_Network
    )

    env = create_env('conveyor_network_v0')

    config = {
        "env": "ConveyorEnv_v0",
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
        trainer = dqn.DQNTrainer(config=dqn_config, env=ConveyorEnv)

        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= args.stop_timesteps or \
                    result["episode_reward_mean"] >= args.stop_reward:
                break

    else:
        # automated run with tune and grid search and Tensorboard
        print("Training with Ray Tune.")
        result = tune.run(args.run, config=config, stop=stop)

        if args.as_test:
            print("Checking if the learning goals are achieved")
            check_learning_achieved((result, args.stop_reward))

    ray.shutdown()