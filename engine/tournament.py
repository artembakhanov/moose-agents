from itertools import combinations
from collections import Counter

import numpy as np

from engine.game import Game, Player


class Tournament:
    def __init__(self, players):
        self._players = players
        self._counts = Counter(players)
        self._scores = dict(zip(players, [0] * len(players)))
        self._rounds = len(players) * (len(players) - 1) / 2

    def clear(self):
        self._scores = dict(zip(self._players, [0] * len(self._players)))

    def start(self, n=100):
        for i, j in combinations(self._players, 2):
            game = Game()
            player1 = Player(game=game, strategy=i)
            player2 = Player(game=game, strategy=j)

            for k in range(n):
                game.move(player1.move(), player2.move())

            self._scores[i] += player1.score / self._counts[i] / self._rounds
            self._scores[j] += player2.score / self._counts[j] / self._rounds

        return self._scores
