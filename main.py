from engine.interactive import InteractiveGame, InteractiveGameRNN
from train.dqn import FCNet, RNNNet

game = InteractiveGame(verbose=True, net=FCNet)
game.load_env("GooseStrategyDetectionEnv-v0")  # ("GooseHumanEnv-v0") GooseBaseEnv GooseHumanEnv
game.load_model("test3_15000")

game.play()
