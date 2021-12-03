import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from util import plot_learning_curve
from gym import spaces
from Conveyor_Network import ConveyorEnv


class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, *n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = F.relu(self.fc3(layer2))
        layer4 = F.relu(self.fc4(layer3))
        actions = self.fc5(layer4)

        return actions


class Agent:
    def __init__(self, input_dims, n_actions, lr=0.0001, gamma=0.99, epsilon=1.0,
                 eps_dec=1e-5, eps_min=0.01):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = n_actions

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions[0], self.input_dims)

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor(obs, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]

        q_next = self.Q.forward(states_).max()

        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()


if __name__ == '__main__':
    env = ConveyorEnv()

    n_episodes = 100000
    scores = []
    eps_history = []

    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.__getitem__(0).n)

    for i in range(n_episodes):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f' % (score, avg_score, agent.epsilon))

    filename = 'naive_dqn.png'
    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
