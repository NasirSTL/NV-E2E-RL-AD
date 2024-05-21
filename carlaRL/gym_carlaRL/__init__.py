from gymnasium.envs.registration import register

register(
    id='CarlaRL-v0',
    entry_point='gym_carlaRL.envs:CarlaEnv',
)