import abc

import gym
from gym import spaces
from random import choice
from random import randint

from engine.game import Game, Player, FIELDS, A, B, C, ONE_HOT, FIELDS_SET


class GooseBaseEnv(gym.Env):
    environment_name = "Goose Environment"

    def __init__(self, rewards=None, rounds_number=100, verbose=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(3), spaces.Discrete(3), spaces.Box(0, 10, shape=(3,)))
        )

        self._game = Game(verbose=verbose)
        self._player1 = Player(self._game)
        self._player2 = Player(self._game)
        self._done = False
        self._rounds_number = rounds_number

        if rewards is None:
            rewards = dict(
                local=1,
                glob=0
            )

        self.rewards = rewards

        self._game.move(choice(FIELDS), choice(FIELDS))

        self.stats = {
            'games_played': 0
        }

    def step(self, action):
        if self._done:
            self.reset()

        move1 = action
        move2 = self._player2_policy()

        self._game.move(move1, move2)

        reward = self.rewards["local"] * (self._player1.local_score - self._player2.local_score)

        if len(self._game.moves) == self._rounds_number:
            self._done = True

        if self._done:
            reward += self.rewards["glob"] * (self._player1.score - self._player2.local_score)
            self.stats['games_played'] += 1

        return self.state, reward, self._done, {}

    def reset(self):
        self._done = False
        self._game.reset()
        self._game.move(choice(FIELDS), choice(FIELDS))

        return self.state

    def _player2_policy(self):
        """This player is random one"""
        return choice(list(FIELDS_SET - {self._player1.moves[-1], self._player2.moves[-1]}))

    @property
    def observation_space_n(self):
        return 9

    @property
    def action_space_n(self):
        return 3

    @property
    def legal_actions(self):
        return [0, 1, 2]

    @property
    def state(self):
        return list(map(self._game.payoff, FIELDS)) + ONE_HOT[self._player1.moves[-1]] + ONE_HOT[
            self._player2.moves[-1]]

    @property
    def performance_threshold(self):
        return 1
