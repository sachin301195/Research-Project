import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from pathlib import Path

from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from gym.spaces import Dict, Discrete, Box, Tuple

matplotlib.use('agg')


def configure_logger():
    timestamp = time.strftime('%Y-%m-%d')
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('./logs/application-utils-'+timestamp+'.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s : %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


logger = configure_logger()

torch, nn = try_import_torch()


class TorchParametricActionModel(DQNTorchModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)

        self.action_model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict,
                state,
                seq_lens):
        input_dict['obs'] = input_dict['obs'].float()
        fc_out, _ = self.action_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return self.action_model.value_function()


class CustomPlot:

    @staticmethod
    def save_combined_plot(path, xlabel, ylabel, plot_label, t, y_data_list, algo_list, checkpoint_list):
        fig = plt.figure()
        ax = fig.gca()

        for y, algo, checkpoint in zip(y_data_list, algo_list, checkpoint_list):
            ax.plot(t, y, label=algo)

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_title(plot_label)
        fig.savefig(path, dpi=300)
        plt.close(fig)

    @staticmethod
    def save_combined_ci_plot(path, xlabel, ylabel, plot_label, t, y_data_list, algo_list, checkpoint_list, std_err_list, alpha=0.15):
        fig = plt.figure()
        ax = fig.gca()

        for ydata, std_err, algo, checkpoint in zip(y_data_list, std_err_list, algo_list, checkpoint_list):
            y_lb = ydata-std_err
            y_ub = ydata+std_err
            # y_lb[y_lb < 0] = 0.0
            ax.fill_between(t, y_ub, y_lb, alpha=alpha)
            # ax.plot(t, ydata, label=algo+'_'+str(checkpoint))
            ax.plot(t, ydata, label=algo)

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_title(plot_label)

        # if ylabel == 'Pedestrian Wait Time (s)':
        #     ax.set_ylim([0, 50])
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def save_plot(path, xlabel, ylabel, plot_label, t, y_data, algo):
        plt.plot(t, y_data, label=algo)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(plot_label)
        plt.savefig(path, dpi=300)
        plt.clf()

    @staticmethod
    def save_ci_plot(path, xlabel, ylabel, plot_label, t, ydata, lb, ub, algo, alpha=0.15):
        # lb[lb < 0] = 0.0
        plt.fill_between(t, ub, lb, alpha=alpha)
        plt.plot(t, ydata, label=algo)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(plot_label)
        plt.savefig(path, dpi=300)
        plt.clf()

    @staticmethod
    def plot_figure():
        plt.figure()

    @staticmethod
    def plot_scatter_plot(path, xlabel, ylabel, plot_label, t, ydata):
        plt.scatter(t, ydata)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(plot_label)
        plt.savefig(path, dpi=300)
        plt.clf()

    @staticmethod
    def add_to_scatter_plot(t, ydata, ylabel, index):
        ax = plt.subplot(int('21'+str(index)))
        ax.set_ylabel(ylabel=ylabel)
        plt.scatter(t, ydata)

    @staticmethod
    def save_scatter_plot(path, plot_label, xlabel):
        plt.xlabel(xlabel)
        plt.savefig(path, dpi=300)
        plt.clf()


def plot_learning_curve(x, scores, epsilon, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, epsilon, color='C0')
    ax.set_xlabel('Training Steps', color='C0')
    ax.set_ylabel('Epsilon', color='C0')
    ax.tick_params(axis='x', colors='C0')
    ax.tick_params(axis='y', colors='C0')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 100):(t + 1)])

    ax2.scatter(x, running_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylable('Score', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')

    plt.savefig(filename)
