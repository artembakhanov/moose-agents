import torch

from engine.interactive import InteractiveGame


class PredictiveInteractive(InteractiveGame):
    def play(self):
        # environment
        obs = self.env.reset()

        while True:
            obs_tensor = torch.Tensor(obs).reshape(self.env.observation_space_n)
            action = torch.argmax(self.model(obs_tensor)).item()

            obs, reward, done, _ = self.env.step(action)

            if done:
                break
