from gym.envs.registration import register

register(
    id='GooseBaseEnv-v0',
    entry_point='environment.base:GooseBaseEnv',
    reward_threshold=50,
)
