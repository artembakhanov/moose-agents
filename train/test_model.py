import torch

from engine.game import Game
from train.dqn import FCNet
import numpy as np


def payoff(x):
    return Game.vegetation(x) - 5


model = FCNet(9, 3)
model.load_state_dict(torch.load("policies/dqn-cnn"))

print(model(torch.Tensor([[payoff(2), payoff(0), payoff(1), 0, 1, 0, 0, 1, 0]])))
