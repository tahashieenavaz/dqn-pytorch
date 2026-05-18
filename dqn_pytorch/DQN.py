import torch
import torch.nn.functional as F
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
    def __init__(
        self,
        timesteps: int = 500_000,
        buffer_size: int = 1_000_000,
        lr: float = pow(10, -4),
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.001,
        fraction_epsilon: float = 0.1,
        gamma: float = 0.99,
        tau: float = 1.0,
        target_network_frequency: int = 1000,
        batch_size: int = 32,
        train_frequency: int = 4,
        learning_starts: int = 80_000,
    ):
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
        # FIXED: Changed single_observations_space to single_observation_space
        observations_shape = environments.single_observation_space.shape
        return action_dimension, observations_shape

    def __get_epsilon(self, step: int) -> float:
        return linear_schedule(
            self.max_epsilon,
            self.min_epsilon,
            self.fraction_epsilon * self.timesteps,
            step,
        )

    def __get_random_actions(self, action_dimension: int):
        # FIXED: Wrapped in a numpy array to match the shape and type of __get_wise_actions
        return numpy.array([random.randint(0, action_dimension - 1)])

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

    def __log_out_info(self, infos, step, start_time):
        if "final_info" not in infos:
            return

        for info in infos["final_info"]:
            if not (info and "episode" in info):
                continue
            print(
                f"Step: {step} | Return: {info['episode']['r'][0]:.2f} | SPS: {int(step / (time.time() - start_time))}"
            )

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
            self.__log_out_info(step=step, start_time=start_time, infos=infos)

            real_next_obs = next_observations.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            buffer.add(observations, real_next_obs, actions, rewards, terminations)
            observations = next_observations

            if step > self.learning_starts and step % self.train_frequency == 0:
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(
                    self.batch_size
                )

                device_type = (
                    self.device.type
                    if hasattr(self.device, "type")
                    else str(self.device)
                )
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
                ):
                    with torch.no_grad():
                        next_state_actions = network(b_next_obs).argmax(
                            dim=1, keepdim=True
                        )
                        target_max = target_network(b_next_obs).gather(
                            1, next_state_actions
                        )
                        td_target = b_rewards + self.gamma * target_max * (1 - b_dones)

                    old_val = network(b_obs).gather(1, b_actions)
                    loss = F.smooth_l1_loss(old_val, td_target)

                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            if step % self.target_network_frequency == 0:
                for target_param, q_param in zip(
                    target_network.parameters(), network.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * q_param.data + (1.0 - self.tau) * target_param.data
                    )

        duration = time.time() - start_time
        environments.close()
