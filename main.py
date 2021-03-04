from engine.interactive import InteractiveGame
from train.dqn import FCNet

game = InteractiveGame(verbose=True, net=FCNet)
game.load_env("GooseHumanEnv-v0")#("GooseHumanEnv-v0") GooseBaseEnv GooseHumanEnv
game.load_model("against-greedy_15000")

game.play()
