import gymnasium
import ale_py
import baloot
from dqn_pytorch.functions import make_environment

gymnasium.register_envs(ale_py)


class DQN:
    def __init__(self):
        self.device = baloot.acceleration_device()

    def __make_environments(self, environment_name: str, seed: int):
        return gymnasium.vector.SyncVectorEnv(
            [make_environment(environment_name, seed)]
        )

    def __get_environment_info(self, environments):
        action_dimension = environments.single_action_space.n
        observation_shape = environments.single_observation_space.shape
        return action_dimension, observation_shape

    def train(self, *, environment_name: str, seed: int):
        baloot.seed(seed)
        environments = self.__make_environments(
            environment_name=environment_name, seed=seed
        )
        action_dimension, observation_shape = self.__get_environment_info(
            environments=environments
        )
        pass
