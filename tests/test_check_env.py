from gymnasium.utils.env_checker import check_env
from environment import DrillEnv


def test_gymnasium_environment_contract():
    check_env(DrillEnv(), skip_render_check=True)
