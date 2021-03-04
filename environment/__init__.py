from gym.envs.registration import register

register(
    id='GooseBaseEnv-v0',
    entry_point='environment.base:GooseBaseEnv',
    reward_threshold=50,
)

register(
    id='GooseHumanEnv-v0',
    entry_point='environment.human_player:GoosaHumanEnv',
    reward_threshold=50,
)
register(
    id='GooseGreedyEnv-v0',
    entry_point='environment.greedy:GooseGreedyEnv',
    reward_threshold=50,
)
