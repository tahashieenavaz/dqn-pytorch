import gymnasium
import ale_py

gymnasium.register_envs(ale_py)


class DQN:
    def __init__(self):
        pass

    def train(self, *, environment_name: str, seed: int):
        pass
