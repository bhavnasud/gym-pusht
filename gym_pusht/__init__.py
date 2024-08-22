from gymnasium.envs.registration import register

# default PushT env
register(
    id="gym_pusht/PushT-v0",
    entry_point="gym_pusht.envs:PushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)

# MM PushT env scale 1
register(
    id="gym_pusht/PushT-v1",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 1.0, "scale_high": 1.0},
)

# T block scale 0.5
register(
    id="gym_pusht/PushT-v2",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 0.5, "scale_high": 0.5},
)

# T block scale 2
register(
    id="gym_pusht/PushT-v3",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 2.0, "scale_high": 2.0},
)

# randomly chooses scale with T shape
register(
    id="gym_pusht/PushT-multiscale",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": [0.5, 1.0, 2.0], "scale_high": [0.5, 1.0, 2.0], "variable_scale": True},
)

# T block with triangle scale 1
register(
    id="gym_pusht/PushT-shape-v1",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 1.0, "scale_high": 1.0, "shape_idx": 1},
)

# L shape scale 1
register(
    id="gym_pusht/PushT-shape-v2",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 1.0, "scale_high": 1.0, "shape_idx": 2},
)

# Hexagon shape scale 1
register(
    id="gym_pusht/PushT-shape-v3",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 1.0, "scale_high": 1.0, "shape_idx": 3},
)

# Randomly chooses shape with scale 1
register(
    id="gym_pusht/PushT-multishape",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": 1.0, "scale_high": 1.0, "shape_idx": [1,2,3], "variable_shape": True},
)

# Randomly chooses scale or shape, not both
register(
    id="gym_pusht/PushT-multiscale-multishape",
    entry_point="gym_pusht.envs:MMPushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "scale_low": [1.0, 2.0, 3.0], "scale_high": [1.0, 2.0, 3.0], "shape_idx": [1,2,3], "variable_scale": True, "variable_shape": True},
)