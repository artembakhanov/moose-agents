import numpy as np

from engine.game import FIELDS
from .base import GooseBaseEnv as BaseEnv


class GooseGreedyEnv(BaseEnv):
    def _player2_policy(self):
        return np.argmax(self._game.fields)
