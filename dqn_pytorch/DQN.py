import torch
import gymnasium
import ale_py
import baloot
import time
import numpy
import random
from dqn_pytorch.functions import make_environment, linear_schedule
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
        observations_shape = environments.single_observations_space.shape
        return action_dimension, observations_shape

    def __get_epsilon(self, step: int) -> float:
        return linear_schedule(
            self.max_epsilon,
            self.min_epsilon,
            self.exploration_fraction * self.timesteps,
            step,
        )

    def __get_random_actions(self, action_dimension: int):
        return random.randint(0, action_dimension - 1)

    @torch.inference_mode()
    def __get_wise_actions(self, network: torch.nn.Module, observations: torch.Tensor):
        q_values = network(torch.Tensor(observations).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def __get_actions(
        self,
        epsilon: float,
        action_dimension: int,
        observations: torch.Tensor,
        network: torch.nn.Module,
    ):
        if random.random() < epsilon:
            return self.__get_random_actions(action_dimension=action_dimension)
        else:
            return self.__get_wise_actions(network=network, observations=observations)

    def train(self, *, environment_name: str, seed: int):
        baloot.seed(seed)
        environments = self.__make_environments(
            environment_name=environment_name, seed=seed
        )
        action_dimension, observations_shape = self.__get_environment_info(
            environments=environments
        )

        network = DQNNetwork(action_dimension).to(self.device)
        target_network = DQNNetwork(action_dimension).to(self.device)
        target_network.load_state_dict(network.state_dict())

        if hasattr(torch, "compile"):
            network = torch.compile(network)

        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        buffer = ReplayBuffer(self.buffer_size, observations_shape, self.device)

        observations, _ = environments.reset(seed=seed)
        start_time = time.time()

        for step in range(self.timesteps):
            epsilon = self.__get_epsilon(step=step)
            actions = self.__get_actions(
                action_dimension=action_dimension,
                observations=observations,
                epsilon=epsilon,
                network=network,
            )
            next_observations, rewards, terminations, truncations, infos = (
                environments.step(actions)
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"Step: {step} | Return: {info['episode']['r'][0]:.2f} | SPS: {int(step / (time.time() - start_time))}"
                        )

        duration = time.time() - start_time
        environments.close()
