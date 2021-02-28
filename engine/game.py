from math import exp

A, B, C = 0, 1, 2
FIELDS_SET = {A, B, C}
FIELDS = [A, B, C]
ONE_HOT = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]


class Player(object):
    def __init__(self, game):
        self.score = 0
        self.local_score = 0
        self.moves = []
        self._game = game

        if len(self._game.players) == 2:
            raise Exception("No more than two players are allowed")

        self._game.players.append(self)

    def reset(self):
        self.score = 0
        self.moves = []


class Game(object):

    def __init__(self):
        self.fields = {
            A: 1,
            B: 1,
            C: 1
        }

        self.players = [
        ]

        self.moves = []

    def move(self, move1, move2):
        if move1 not in FIELDS_SET or move2 not in FIELDS_SET:
            raise AttributeError("Move should be one of the fields available")

        for field in FIELDS_SET - {move1, move2}:
            self.fields[field] += 1

        if move1 == move2:
            self.fields[move1] = max(self.fields[move1] - 1, 0)
        else:
            self.players[0].score += self.payoff(move1)
            self.players[1].score += self.payoff(move2)
            self.players[0].local_score = self.payoff(move1)
            self.players[1].local_score = self.payoff(move2)

            self.fields[move1] = max(self.fields[move1] - 1, 0)
            self.fields[move2] = max(self.fields[move2] - 1, 0)

        self.players[0].moves.append(move1)
        self.players[1].moves.append(move2)

        self.moves.append((move1, move2))

    def payoff(self, field):
        return Game.vegetation(self.fields[field]) - 5

    def reset(self):
        for player in self.players:
            player.reset()

        self.fields = {
            A: 1,
            B: 1,
            C: 1
        }

        self.moves = []

    @staticmethod
    def vegetation(x):
        return 10 / (1 + 1 / exp(x))
