# src/environment_eval.py
from configuration import project_path
from environment import DrillEnv as TrainingDrillEnv


class DrillEnv(TrainingDrillEnv):
    """Evaluation environment sharing the training observation contract."""

    def __init__(self, config=None):
        super().__init__(config)
        self.pool_paths = [project_path(self.config["eval_pool"])]
