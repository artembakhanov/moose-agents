import numpy as np

from engine.game import FIELDS
from .base import GooseBaseEnv as BaseEnv
from .rnn_base import GooseBaseRNNEnv


class GooseGreedyEnv(BaseEnv):
    def _player2_policy(self):
        return np.argmax(self._game.fields)


class GooseGreedyRNNEnv(GooseBaseRNNEnv):
    def _player2_policy(self):
        return np.argmax(self._game.fields)
