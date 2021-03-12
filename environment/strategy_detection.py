import random

import numpy as np

from engine.game import FIELDS, FIELDS_SET, ONE_HOT
from environment.base import GooseBaseEnv


def greedy(game, player):
    return np.argmax(game.fields)


def random_greedy(game, player):
    fields = np.array(game.fields)
    return np.random.choice(fields == np.max(fields))


def random_random(game, player):
    return np.random.choice(FIELDS)


def random_(game, player):
    return np.random.choice(list(FIELDS_SET - {game.players[0].moves[-1], game.players[1].moves[-1]}))


def human(game, player):
    while True:
        action = input("Enter cell (0, 1 or 2): ")

        if action in list("012"):
            return int(action)


def random_weighted(game, player):
    fields = np.array(game.fields)

    return np.random.choice(FIELDS, p=fields / np.sum(fields))


class GooseStrategyDetectionEnv(GooseBaseEnv):
    strategies = ["greedy", "random-greedy", "random-random", "random", "random-weighted"]
    strategies_funcs = [
        greedy,
        random_greedy,
        random_random,
        random_,
        random_weighted,

        human,
    ]
    current_strategy = 3
    player2_strategy = random.randint(0, len(strategies) - 1)
    states = np.zeros(90)

    def step(self, action: int):

        if self._done:
            self.reset()

        self.current_strategy = action
        reward = 0
        for i in range(10):
            move1 = self.strategies_funcs[self.current_strategy](self._game, 0)
            move2 = self.strategies_funcs[self.player2_strategy](self._game, 1)
            self._game.move(move1, move2)
            reward += self._player1.local_score

            self.states = np.append(self.states, ONE_HOT[move1])
            self.states = np.append(self.states, ONE_HOT[move2])
            self.states = np.append(self.states, list(map(self._game.payoff, FIELDS)))

            if len(self._game.moves) == self._rounds_number:
                self._done = True

            if self._done:
                reward += self.rewards["glob"] * (self._player1.score - self._player2.local_score)
                self.stats['games_played'] += 1

        return self.state, reward, self._done, {}

    def reset(self, human=False):
        super(GooseStrategyDetectionEnv, self).reset()
        if human:
            self.current_strategy = -1
        else:
            self.current_strategy = 3
        self.player2_strategy = random.randint(0, len(self.strategies) - 1)
        self.states = np.zeros(90)

        self.step(self.current_strategy)
        return self.state

    @property
    def state(self):
        return self.states[-180:]

    @property
    def legal_actions(self):
        return list(range(0, self.action_space_n))

    @property
    def action_space_n(self):
        return len(self.strategies)

    @property
    def observation_space_n(self):
        return 20 * 9
