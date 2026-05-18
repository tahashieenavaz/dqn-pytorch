import numpy
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.observations = numpy.empty((capacity, *obs_shape), dtype=numpy.uint8)
        self.next_observations = numpy.empty((capacity, *obs_shape), dtype=numpy.uint8)
        self.actions = numpy.empty((capacity, 1), dtype=numpy.int64)
        self.rewards = numpy.empty((capacity, 1), dtype=numpy.float32)
        self.dones = numpy.empty((capacity, 1), dtype=numpy.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = obs[0]
        self.next_observations[self.pos] = next_obs[0]
        self.actions[self.pos] = action[0]
        self.rewards[self.pos] = reward[0]
        self.dones[self.pos] = done[0]
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = numpy.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.observations[indices], device=self.device).float(),
            torch.as_tensor(self.actions[indices], device=self.device),
            torch.as_tensor(self.rewards[indices], device=self.device),
            torch.as_tensor(
                self.next_observations[indices], device=self.device
            ).float(),
            torch.as_tensor(self.dones[indices], device=self.device),
        )
