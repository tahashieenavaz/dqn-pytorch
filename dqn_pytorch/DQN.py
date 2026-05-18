import torch
import gymnasium
import ale_py
import baloot
import time
from dqn_pytorch.functions import make_environment
from dqn_pytorch.networks import DQNNetwork
from dqn_pytorch.ReplayBuffer import ReplayBuffer

gymnasium.register_envs(ale_py)


class DQN:
    def __init__(self, buffer_size: int = 1_000_000, lr: float = pow(10, -4)):
        for key, value in locals().items():
            if key == "self":
                continue
            setattr(self, key, value)

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

        network = DQNNetwork(action_dimension).to(self.device)
        target_network = DQNNetwork(action_dimension).to(self.device)
        target_network.load_state_dict(network.state_dict())

        if hasattr(torch, "compile"):
            network = torch.compile(network)

        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        buffer = ReplayBuffer(self.buffer_size, observation_shape, self.device)

        observation, _ = environments.reset(seed=seed)
        start_time = time.time()

        pass
