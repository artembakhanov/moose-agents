import torch

from engine.game import Game
from train.dqn import FCNet, RNNNet
import numpy as np


def payoff(x):
    return Game.vegetation(x) - 5


model = RNNNet(9, 3)
model.load_state_dict(torch.load("policies/lstm_15000"))

model1 = FCNet(9, 3)
model1.load_state_dict(torch.load("policies/test1_45000", map_location='cpu'))

print(model(torch.Tensor([[payoff(2), payoff(0), payoff(1), 0, 1, 0, 0, 1, 0]])))
print(model1(torch.Tensor([[payoff(2), payoff(0), payoff(1), 0, 1, 0, 0, 1, 0]])))
