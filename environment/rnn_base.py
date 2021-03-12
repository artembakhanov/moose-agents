import numpy as np

from environment.base import GooseBaseEnv


class GooseBaseRNNEnv(GooseBaseEnv):
    environment_name = "Goose RNN Environment"

    def __init__(self, rewards=None, rounds_number=100, verbose=False):
        super().__init__(rewards, rounds_number, verbose)

        self.states = np.empty((0, self.observation_space_n), dtype=np.float64)

    def step(self, action):
        if self._done:
            self.reset()

        move1 = action
        move2 = self._player2_policy()

        self._game.move(move1, move2)

        reward = self.rewards[
                     "local"] * self._player1.local_score  # (self._player1.local_score - self._player2.local_score)

        if len(self._game.moves) == self._rounds_number:
            self._done = True

        if self._done:
            reward += self.rewards["glob"] * (self._player1.score - self._player2.local_score)
            self.stats['games_played'] += 1

        self.states = np.append(self.states, [self.state], axis=0)

        result = np.zeros((100, self.observation_space_n))

        # todo: fix this shit with dimensions: self.states[-self._rounds_number:]
        result[:len(self.states[-self._rounds_number:])] = self.states

        return result, reward, self._done, {}

    def reset(self):
        super(GooseBaseRNNEnv, self).reset()

        self.states = np.empty((0, self.observation_space_n))

        result = np.zeros((100, self.observation_space_n))
        result[:len(self.states[-self._rounds_number:])] = self.states

        return result
