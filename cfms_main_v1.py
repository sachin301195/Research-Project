from conveyor_environment.conveyor_environment.envs.conveyor_network_v3 import ConveyorEnv_v3
import random
import matplotlib.pyplot as plt

NUM_EPISODES = 1
REWARDS = []
AVG_THROUGHPUT = []
ORDER_THROUGHPUT = []

env = ConveyorEnv_v3({}, version='full', final_reward=10, mask=True)

for n in range(NUM_EPISODES):
    done = False
    score = 0
    obs = env.reset()
    actions = list(obs['action_mask'])

    while not done:
        final_actions = []
        for idx, a in enumerate(actions):
            if a != 0:
                final_actions.append(idx)
        action = random.choice(final_actions)
        obs, reward, done, info = env.step(action)
        score += reward
        actions = list(obs['action_mask'])
        avg_throughput, order_throughput = env.render()
        AVG_THROUGHPUT.append(avg_throughput)
        ORDER_THROUGHPUT.append(order_throughput)
    plt.plot(AVG_THROUGHPUT)
    plt.plot(ORDER_THROUGHPUT)