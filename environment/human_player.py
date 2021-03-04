from engine.game import FIELDS
from .base import GooseBaseEnv as BaseEnv


class GoosaHumanEnv(BaseEnv):
    def _player2_policy(self):
        while True:
            action = input("Enter cell (0, 1 or 2): ")

            if action in list("012"):
                return int(action)
