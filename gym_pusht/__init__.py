from gymnasium.envs.registration import register

register(
    id="gym_pusht/PushT-v0",
    entry_point="gym_pusht.envs:PushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)

register(
    id="gym_pusht/PushT-v1",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 1.0, "scale_high": 1.0},
)

register(
    id="gym_pusht/PushT-v2",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 0.5, "scale_high": 0.5},
)

register(
    id="gym_pusht/PushT-v3",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 2.0, "scale_high": 2.0},
)

register(
    id="gym_pusht/PushT-multiscale",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": [0.5, 1.0, 2.0], "scale_high": [0.5, 1.0, 2.0]},
)