from hydra import compose, initialize_config_dir

from configuration import CONFIG_DIR
from environment import DrillEnv


def test_hydra_composes_groups_and_cli_overrides():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                "grid.nx=41",
                "geology.lens_count=8",
                "training=a2c",
            ],
        )

    assert cfg.grid.nx == 41
    assert cfg.geology.lens_count == 8
    assert cfg.training.algorithm == "a2c"


def test_environment_uses_hydra_action_and_observation_settings():
    env = DrillEnv()

    assert env.action_space.n == 5
    assert env.actions == [2, 4, 8, 12, 16]
    assert env.observation_space.high.tolist() == [100.0, 300.0, 100.0]
