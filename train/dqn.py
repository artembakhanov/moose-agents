import torch
import torch.nn as nn
import torch.optim as optim
import gym
import environment  # module init needs to run
from prop.algorithms.dqn import Agent
from prop.net.feed_forward import FeedForward


class FCNet(FeedForward):
    def __init__(self, obs_size, n_actions):
        # model is initiated in parent class, set params early.
        self.obs_size = obs_size
        self.n_actions = n_actions
        super(FCNet, self).__init__()

    def model(self):
        # observations -> hidden layer with relu activation -> actions
        return nn.Sequential(
            nn.Linear(self.obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    env = gym.make('GooseGreedyEnv-v0',)
    agent = Agent(
        env=env,
        net=FCNet,
        name="against-greedy",
        learning_rate=1e-5,
        batch_size=128,
        optimizer=optim.Adam,
        loss_cutoff=0.02,
        max_std_dev=17,
        epsilon_decay=3000,
        double=True,
        target_net_update=500,
        eval_every=500,
        logdir="logs",
        dev=device)

    agent.train()

    print("### some stats ###")
    print(f"games played: {env.stats['games_played']}")

