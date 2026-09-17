from gymnasium.utils.env_checker import check_env
from environment import DrillEnv

check_env(
    DrillEnv(),
    skip_render_check=True
)

print("check_env passed")
