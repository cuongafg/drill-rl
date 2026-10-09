from environment import DrillEnv


def test_environment_reset_returns_valid_observation():
    env = DrillEnv()

    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)
