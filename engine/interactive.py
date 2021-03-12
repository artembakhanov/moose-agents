import time

import gym
import torch
import sys


class InteractiveGame:
    def __init__(self, verbose=False, debug=False, first_player=None, net=None):
        self.env = None
        self.model = None
        self.verbose = verbose
        self.debug = debug
        self.first_player = first_player
        self.net = net

    def load_env(self, env):
        self.env = gym.make(env, verbose=self.verbose)

        # seed
        self.env.seed(time.time())

    def load_model(self, policy):
        self.model = self.net(obs_size=self.env.observation_space_n, n_actions=self.env.action_space_n)
        self.model.load_state_dict(torch.load(f"train/policies/{policy}"))
        self.model.eval()

    def play(self):
        # environment
        obs = self.env.reset(human=True)

        while True:
            obs_tensor = torch.Tensor(obs).reshape(self.env.observation_space_n)
            action = torch.argmax(self.model(obs_tensor)).item()

            obs, reward, done, _ = self.env.step(action)

            if done:
                break


class InteractiveGameRNN(InteractiveGame):
    def play(self):
        obs = self.env.reset()

        while True:
            obs_tensor = torch.Tensor(obs)[None, :, :]
            action = torch.argmax(self.model(obs_tensor)).item()

            obs, reward, done, _ = self.env.step(action)

            if done:
                break
